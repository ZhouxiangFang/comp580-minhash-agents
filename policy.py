"""Task-level reuse: answer an incoming task from a verified record of the same item.

MinHash proposes, an exact check disposes. LSH retrieves verified records whose
text is similar to the task; a record's answer is reused only if the two are the
same text up to formatting: exactly the same words and numbers in the same
order, or the same paragraphs in another order. One changed word, number or
identifier blocks reuse, and so does a swap of two words inside a paragraph;
that is what separates a duplicate from a hard negative that shares 90 % of its
shingles.

Nothing here knows a dataset. The fields to compare, the fields that hold a
record's answer and the shape of an answer are all read off the data:
  * compared text  = the fields a task shares with the history records
                     (dates are skipped: they say when a record was filed);
  * a record's answer = the fields history records have and tasks do not.
"""

import re
from collections import Counter

from minhash import LSHIndex, MinHasher, shingles, tokenize

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}([T ][\d:.]+)?(Z|[+-]\d{2}:?\d{2})?")


def _leaves(value):
    """Every scalar inside a JSON value, in order, without dates and nulls."""
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _leaves(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)
    elif value is not None and not _DATE.fullmatch(str(value).strip()):
        yield str(value)


def _same_text(entry, words, blocks):
    """Same words in the same order, or the same paragraphs in any order."""
    return entry["words"] == words or entry["blocks"] == blocks


class ReusePolicy:
    """An index of verified records and the rule for reusing one of their answers.

    `lookup(task)` returns `{"answer", "source", "similarity"}` or None;
    `add(task, answer)` indexes a new answer and must only be called after the
    answer passed verification. `validate=False` drops the exact check and reuses
    on similarity alone; it exists for the threshold ablation, not for real use.
    """

    def __init__(self, history, threshold=0.5, validate=True, seed=0):
        self.history = list(history)
        self.threshold = threshold
        self.validate = validate
        self.hasher = MinHasher(seed=seed)
        self.history_fields = set().union(*self.history) if self.history else set()
        self.fields = None        # compared fields, fixed by the first task seen
        self.answer_keys = None   # keys of verified answers seen so far
        self.pointers = {}        # answer key -> history field holding its value
        self.pending = []         # answers added before the index existed

    # -- public ---------------------------------------------------------------
    def lookup(self, task):
        self._ensure_index(task)
        words, blocks = self._text(task)
        for key, estimate in self.index.search(self._signature(words), self.threshold):
            entry = self.entries[key]
            if self.validate and not _same_text(entry, words, blocks):
                continue                      # similar, but not the same item
            if self.validate and entry["ambiguous"]:
                return None                   # same text is known to carry two answers
            answer = self._answer(entry)
            if answer is not None:
                return {"answer": answer, "source": entry["id"], "similarity": estimate}
        return None

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
        self.fields = fields
        self.payload = sorted(self.history_fields - set(task))
        self.field_values = {f: {str(r[f]) for r in self.history if f in r} for f in self.payload}
        # A plain payload field whose value is different in every record names the record.
        self.id_fields = [f for f in self.payload
                          if len(self.field_values[f]) == len(self.history)
                          and all(isinstance(r.get(f), str) and list(_leaves(r[f]))
                                  for r in self.history)]
        self.index = LSHIndex()
        self.entries = {}
        for position, record in enumerate(self.history):
            name = record[self.id_fields[0]] if self.id_fields else position
            self._insert(record, name, None)
        self._flag_ambiguous()
        pending, self.pending = self.pending, []
        for old_task, answer in pending:
            self.add(old_task, answer)

    def _text(self, record):
        """The compared text as (all words in order, the bag of its paragraphs).

        A paragraph is a blank-line-separated part of a string field; any other
        field (a number, a list, a nested object) is one paragraph, so the order
        inside a list still counts.
        """
        blocks = []
        for field in self.fields:
            value = record.get(field)
            parts = re.split(r"\n\s*\n", value) if isinstance(value, str) else [value]
            for part in parts:
                words = tokenize(" ".join(_leaves(part)))
                if words:
                    blocks.append(tuple(words))
        return tuple(w for block in blocks for w in block), Counter(blocks)

    def _signature(self, words):
        return self.hasher.signature(shingles(words))

    def _insert(self, record, name, answer):
        words, blocks = self._text(record)
        key = len(self.entries)
        self.entries[key] = {"id": name, "words": words, "blocks": blocks, "record": record,
                             "answer": answer, "ambiguous": False}
        self.index.add(key, self._signature(words))

    def _flag_ambiguous(self):
        """Never reuse a history record whose exact twin in history has another answer."""
        for key, entry in self.entries.items():
            for other, _ in self.index.search(self.index.signatures[key], self.threshold):
                twin = self.entries[other]
                if other != key and _same_text(twin, entry["words"], entry["blocks"]) \
                        and self._verdict(twin["record"]) != self._verdict(entry["record"]):
                    entry["ambiguous"] = True

    def _verdict(self, record):
        """What a history record says the answer is, without its id and dates."""
        return {f: record.get(f) for f in self.payload
                if f not in self.id_fields and list(_leaves(record.get(f)))}

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
