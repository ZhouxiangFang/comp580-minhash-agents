"""Content-level reuse: do not send the model tool output it has already been shown.

Every tool output of one agent run goes through `ContentCache.fold`, which compares it
with the earlier outputs of the same run:

  1. The same text, character for character, as an earlier output that was shown in
     full: the output becomes one line naming that earlier output.
  2. Otherwise MinHash over three-word shingles, with digits masked, and LSH find the
     earlier outputs with an estimated Jaccard similarity of at least `threshold`; the
     most recently seen of them is the base. Masking only widens this search, so that
     outputs differing in their numbers are still compared; what the model is sent
     comes from the exact comparison below.
  3. An exact line diff against the base. A run of identical lines becomes one
     reference. A line that takes the place of one line of the base lists its exact
     word changes ("old" -> "new"); any other changed or added line is shown in full
     and marked, and a line of the base that is gone is marked removed.

So every changed number, identifier or setting reaches the model, next to the value
it replaces. A rewrite is used only if it is at most `max_ratio` (half) of the text it
replaces, and a word-change line only if the line can be rebuilt from it exactly;
otherwise the original is kept. Only line-number prefixes added by read_file are
removed before comparison.
"""

import dataclasses
import difflib
import json
import re
import time

from minhash import LSHIndex, MinHasher, shingles, tokenize

_NUMBERED = re.compile(r"^\d+\t")        # read_file prefixes every line with "N<TAB>"
_DIGITS = re.compile(r"\d+")
_PIECES = re.compile(r"\s+|\w+|[^\w\s]")  # words, runs of whitespace, single punctuation marks


def _line_key(line, numbered=False):
    """Remove only a known tool-added line number; preserve the complete content."""
    return _NUMBERED.sub("", line, count=1) if numbered else line


def _cut(text, length):
    return text if len(text) <= length else text[:length] + "…"


def _word_changes(old, new):
    """[(before, after)] that turn line `old` into `new`, or None.

    Each `before` occurs exactly once in the line it is applied to, so replacing the
    pairs in order rebuilds `new`; a change is widened with the unchanged words around
    it until that holds. None if no such list rebuilds `new` exactly.
    """
    a, b = _PIECES.findall(old), _PIECES.findall(new)
    changes = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        while not ("".join(a[i1:i2]).strip() and old.count("".join(a[i1:i2])) == 1):
            if (i1 == 0 or j1 == 0) and i2 == len(a):
                return None
            if i1 > 0 and j1 > 0:
                i1, j1 = i1 - 1, j1 - 1
            if i2 < len(a):
                i2, j2 = i2 + 1, j2 + 1
        changes.append(("".join(a[i1:i2]), "".join(b[j1:j2])))
    line = old
    for before, after in changes:             # rebuild the line the way a reader would
        if line.count(before) != 1:
            return None
        line = line.replace(before, after)
    return changes if line == new else None


class ContentCache:
    """The tool outputs of one trajectory, and the rewrite that references them."""

    def __init__(self, threshold=0.5, max_ratio=0.5, seed=0):
        self.threshold = threshold    # estimated Jaccard an earlier output needs to be the base
        self.max_ratio = max_ratio    # a rewrite must be at most this fraction of what it replaces
        self.hasher = MinHasher(seed=seed)
        self.index = LSHIndex()
        self.outputs = []             # outputs that can be a base, in the order they were seen
        self.shown = {}               # text of an output shown in full -> its label
        self.texts = {}               # label -> the different outputs it has named
        self.stats = {"outputs": 0, "folded": 0, "chars_in": 0, "chars_out": 0, "seconds": 0.0}

    def fold(self, label, text, *, numbered=False):
        """Return `text`, or a shorter exact rewrite of it that references earlier outputs."""
        started = time.perf_counter()
        result = self._fold(label, text, numbered)
        self.stats["outputs"] += 1
        self.stats["folded"] += result is not text
        self.stats["chars_in"] += len(text)
        self.stats["chars_out"] += len(result)
        self.stats["seconds"] += time.perf_counter() - started
        return result

    def unseen(self, label):
        """Note that `label` also named an output this cache never saw: a tool that raised,
        whose error the harness reports to the model itself. No reference uses that label."""
        self.texts.setdefault(label, set()).add(object())

    def _unique(self, label):
        """True if `label` names one output only, so that a reference to it is unambiguous."""
        return len(self.texts[label]) == 1

    def _fold(self, label, text, numbered=False):
        # Two calls can share a label (a truncated command, or the same values under other
        # argument names); an output is referenced only by a label no other output shares.
        self.texts.setdefault(label, set()).add(text)
        if text in self.shown and self._unique(self.shown[text]):
            reference = "[reuse] %s: identical to the output of %s shown earlier." % (
                label, self.shown[text])
            if len(reference) <= self.max_ratio * len(text):
                return reference
        lines = text.split("\n")
        numbered = numbered and all(_NUMBERED.match(line) for line in lines if line)
        keys = [_line_key(line, numbered) for line in lines]
        result = text
        found = shingles(tokenize(_DIGITS.sub("#", "\n".join(keys))))
        if found:
            signature = self.hasher.signature(found)
            hits = [key for key, _ in self.index.search(signature, self.threshold)
                    if self._unique(self.outputs[key]["label"])]
            if hits:
                base = self.outputs[max(hits)]        # the most recently seen
                folded = self._rewrite(label, lines, keys, numbered, base)
                if len(folded) <= self.max_ratio * len(text):
                    result = folded
            self.index.add(len(self.outputs), signature)
            self.outputs.append({"label": label, "keys": keys, "numbered": numbered})
        if result is text:
            self.shown.setdefault(text, label)
        return result

    def _rewrite(self, label, lines, keys, numbered, base):
        """`lines` written as their exact differences from the output `base`."""
        name = base["label"]
        out = ["[reuse] %s differs little from %s, shown earlier: [changed] and [new] lines are "
               "shown in full, ≈ lines list every changed word, [removed] lines of %s are gone, "
               "and every other line is identical to it." % (label, name, name)]
        ops = difflib.SequenceMatcher(None, base["keys"], keys, autojunk=False).get_opcodes()
        for tag, a1, a2, b1, b2 in ops:
            if tag == "equal":
                out.append("[reuse] %s: identical to %s" % (
                    self._here(numbered, b1, b2), self._there(base, a1, a2)))
                continue
            paired = tag == "replace" and a2 - a1 == b2 - b1   # line j takes the place of line a1+k
            if not paired:
                out += ["[removed] %s" % self._there(base, i, i + 1) for i in range(a1, a2)]
            for k, j in enumerate(range(b1, b2)):
                prefix = "%d\t" % (j + 1) if numbered else ""
                line = None
                changes = _word_changes(base["keys"][a1 + k], keys[j]) if paired else None
                if changes:
                    line = "%s[≈ %s: %s]" % (prefix, self._there(base, a1 + k, a1 + k + 1), "; ".join(
                        "%s → %s" % (json.dumps(before, ensure_ascii=False),
                                     json.dumps(after, ensure_ascii=False))
                        for before, after in changes))
                    if len(line) > self.max_ratio * len(lines[j]):
                        line = None
                out.append(line or "%s[%s] %s" % (prefix, "changed" if paired else "new", keys[j]))
        return "\n".join(out)

    @staticmethod
    def _here(numbered, first, end):
        """Lines first..end-1 of the output being rewritten."""
        if numbered:
            return "line %d" % end if end - first == 1 else "lines %d-%d" % (first + 1, end)
        return "1 line" if end - first == 1 else "%d lines" % (end - first)

    @staticmethod
    def _there(output, first, end):
        """Lines first..end-1 of an earlier output, named so that the model can find them."""
        name, keys = output["label"], output["keys"]
        quote = lambda i, length: json.dumps(_cut(keys[i], length), ensure_ascii=False)
        if output["numbered"]:
            if end - first == 1:
                return "line %d of %s (%s)" % (end, name, quote(first, 16))
            return "lines %d-%d of %s" % (first + 1, end, name)
        if end - first == 1:
            return "the line %s of %s" % (quote(first, 60), name)
        return "%s, from the line %s to the line %s" % (name, quote(first, 60), quote(end - 1, 60))


def wrap_tools(tools, cache):
    """Copies of `tools` (objects with `name` and `run`) whose output goes through `cache`."""
    def wrap(tool):
        def run(**kwargs):
            # Argument names too: grep(pattern="X") and grep(regex="X") are different calls.
            label = " ".join([tool.name] + ["%s=%s" % item for item in kwargs.items()])
            label = " ".join(label.split())[:80]  # one line, however the command was written
            try:
                output = tool.run(**kwargs)
            except Exception:
                cache.unseen(label)           # e.g. grep called without its regex argument
                raise
            return cache.fold(label, str(output), numbered=tool.name == "read_file")
        return dataclasses.replace(tool, run=run)
    return [wrap(tool) for tool in tools]
