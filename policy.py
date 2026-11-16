"""Task-level reuse: answer an incoming task from a verified record of the same item.

LSH retrieves the verified records with an estimated MinHash similarity to the
task of at least `threshold` (0.5). Then:

  * The data has key fields: the most similar record whose key fields all equal
    the task's is reused, unless another such record has a different answer.
  * The data has no key fields: a record that is the same text as the task up
    to formatting (the same words and numbers in the same order,
    or the same paragraphs in another order) is reused. With
    `no_key_rule="similarity"` the most similar record is reused regardless.

Nothing here knows a dataset. The fields to compare, the key fields, the fields
that hold a record's answer and the shape of an answer are all read off the data:
  * compared text  = the fields a task shares with the history records. A field
                     whose whole value is a date or timestamp is skipped: it
                     says when a record was filed, and a duplicate always
                     arrives later. Dates inside text or inside a nested value
                     are compared like any other number;
  * key fields     = compared fields whose value in every history record is one
                     token-like scalar (no whitespace, not a date), such as a
                     product, an error code or a scenario id;
  * a record's answer = the fields history records have and tasks do not.
"""

import json
import re
import time
from collections import Counter

from minhash import LSHIndex, MinHasher, shingles, tokenize

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}([T ][\d:.]+)?(Z|[+-]\d{2}:?\d{2})?")


def _leaves(value):
    """Every non-null scalar inside a JSON value, including dates and numbers."""
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _leaves(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)
    elif value is not None:
        yield str(value)


def _same_text(entry, words, blocks):
    """Same words in the same order, or the same paragraphs in any order."""
    return entry["words"] == words or entry["blocks"] == blocks


def _same_answer(first, second):
    """True when two answers agree on every key they share (and share at least one)."""
    common = set(first) & set(second)
    return bool(common) and all(json.dumps(first[k], sort_keys=True) ==
                                json.dumps(second[k], sort_keys=True) for k in common)


def _is_short(value):
    """One token-like scalar: no whitespace and not a date. An empty field counts."""
    if value is None:
        return True
    if not isinstance(value, (str, int, float)):
        return False
    text = str(value).strip()
    return not re.search(r"\s", text) and not _DATE.fullmatch(text)


class ReusePolicy:
    """An index of verified records and the rule for reusing one of their answers.

    `lookup(task)` returns `{"answer", "source", "similarity", "layer"}` or None,
    where layer is "key", "exact" or "similarity" (the rule that accepted it).
    `add(task, answer)` indexes a new answer and must only be called after the
    answer passed verification.
    """

    def __init__(self, history, threshold=0.5, no_key_rule="exact", seed=0):
        if no_key_rule not in ("exact", "similarity"):
            raise ValueError("no_key_rule must be 'exact' or 'similarity'")
        self.history = list(history)
        self.threshold = threshold            # similarity a candidate needs
        self.no_key_rule = no_key_rule        # what decides when the data has no key fields
        self.hasher = MinHasher(seed=seed)
        self.history_fields = set().union(*self.history) if self.history else set()
        self.fields = None        # compared fields, fixed by the first task seen
        self.key_fields = []      # short single-token fields, fixed with the compared fields
        self.answer_keys = None   # keys of verified answers seen so far
        self.pointers = {}        # answer key -> history field holding its value
        self.pending = []         # answers added before the index existed
        self.index_build_seconds = 0.0

    # -- public ---------------------------------------------------------------
    def lookup(self, task):
        self._ensure_index(task)
        words, blocks = self._text(task)
        task_key = self._key(task)
        layer = "key" if self.key_fields else self.no_key_rule
        matches = []
        for key, estimate in self.index.search(self._signature(words), self.threshold):
            entry = self.entries[key]         # best first
            if layer == "key" and entry["key"] != task_key:
                continue
            if layer == "exact" and not _same_text(entry, words, blocks):
                continue
            answer = self._answer(entry)
            if answer is not None:
                matches.append((key, answer, estimate))
                if layer != "key":
                    break                     # without key fields the best match is used
        if not matches or any(not _same_answer(matches[0][1], other) for _, other, _ in matches[1:]):
            return None                       # nothing matched, or matching records disagree
        key, answer, estimate = matches[0]
        return {"answer": answer, "source": self.entries[key]["id"],
                "similarity": estimate, "layer": layer}

    def add(self, task, answer):
        """Index a verified answer and learn the shape of answers from it."""
        if self.fields is None:               # nothing looked up yet: index it later
            self.pending.append((task, answer))
            return
        keys = set(answer)
        self.answer_keys = keys if self.answer_keys is None else self.answer_keys & keys
        for key, value in answer.items():     # an answer field that names a history record
            if key not in self.payload and isinstance(value, str):
                for field in self.payload:
                    if value in self.field_values[field]:
                        self.pointers[key] = field
        self._insert(task, task.get("tid", len(self.entries)), dict(answer))

    # -- index ----------------------------------------------------------------
    def _ensure_index(self, task):
        fields = sorted(set(task) & self.history_fields) or sorted(task)
        if fields == self.fields:
            return
        started = time.perf_counter()
        self.fields = fields
        self.payload = sorted(self.history_fields - set(task))
        self.field_values = {f: {str(r[f]) for r in self.history if f in r} for f in self.payload}
        # A plain payload field whose value is different in every record names the record.
        self.id_fields = [f for f in self.payload
                          if len(self.field_values[f]) == len(self.history)
                          and all(isinstance(r.get(f), str) and list(_leaves(r[f]))
                                  and not _DATE.fullmatch(r[f].strip())
                                  for r in self.history)]
        self.key_fields = [f for f in self.fields
                           if all(_is_short(r.get(f)) for r in self.history)
                           and any(r.get(f) is not None for r in self.history)]
        # Two rows per band: at similarity 0.4 a pair still shares a band with
        # probability above 0.99, so candidates near the threshold are found.
        self.index = LSHIndex(bands=64)
        self.entries = {}
        for position, record in enumerate(self.history):
            name = record[self.id_fields[0]] if self.id_fields else position
            self._insert(record, name, None)
        pending, self.pending = self.pending, []
        for old_task, answer in pending:
            self.add(old_task, answer)
        self.index_build_seconds += time.perf_counter() - started

    def _text(self, record):
        """The compared text as (all words in order, the bag of its paragraphs).

        A paragraph is a blank-line-separated part of a string field; any other
        field (a number, a list, a nested object) is one paragraph, so the order
        inside a list still counts.
        """
        blocks = []
        for field in self.fields:
            value = record.get(field)
            if isinstance(value, str) and _DATE.fullmatch(value.strip()):
                continue                      # a date field: when it was filed, not what it says
            parts = re.split(r"\n\s*\n", value) if isinstance(value, str) else [value]
            for part in parts:
                words = tokenize(" ".join(_leaves(part)))
                if words:
                    blocks.append(tuple(words))
        return tuple(w for block in blocks for w in block), Counter(blocks)

    def _key(self, record):
        """The key fields' values, compared like any other tokens (case, identifier hyphens)."""
        return tuple(tuple(tokenize(str(record.get(f)))) if record.get(f) is not None else ()
                     for f in self.key_fields)

    def _signature(self, words):
        return self.hasher.signature(shingles(words))

    def _insert(self, record, name, answer):
        words, blocks = self._text(record)
        key = len(self.entries)
        self.entries[key] = {"id": name, "words": words, "blocks": blocks, "record": record,
                             "answer": answer, "key": self._key(record)}
        self.index.add(key, self._signature(words))

    # -- answers --------------------------------------------------------------
    def _answer(self, entry):
        """The answer object a matched entry stands for, or None when it cannot supply one."""
        if entry["answer"] is not None:       # a verified answer to an earlier task: copy it
            return dict(entry["answer"])
        record = entry["record"]
        nested = [record[f] for f in self.payload if isinstance(record.get(f), dict)]
        if len(nested) == 1:                  # the record stores the answer object itself
            return dict(nested[0])
        if nested or not self.answer_keys:    # no verified example of an answer's shape yet
            return None
        answer = {}
        for key in self.answer_keys:
            field = key if key in self.payload else self.pointers.get(key)
            if field is None or field not in record:
                return None                   # the record lacks a field answers must have
            answer[key] = record[field]
        return answer
