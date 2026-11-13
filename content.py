"""Content-level reuse: do not send the model text it has already been shown.

Every tool output of one agent run goes through `ContentCache.fold`. MinHash
(over number-masked shingles, so two files that differ only in their numbers
still look alike) finds the earlier output most similar to the new one. An exact
line diff against it then decides what is really repeated: runs of lines that
are identical, character for character, are replaced by one `[reuse]` line that
names where they were shown; every other line is passed through unchanged.

The diff is exact and never masked, so a changed number, identifier or setting
is always a line the model sees in full, and a line that disappeared is listed.
If the folded text is not clearly shorter, the original output is returned.
Nothing here knows a dataset or a tool; it only looks at text.
"""

import dataclasses
import difflib
import re
import time

from minhash import LSHIndex, MinHasher, shingles, tokenize

_NUMBERED = re.compile(r"^\d+\t")        # read_file prefixes every line with "N<TAB>"
_QUOTED = re.compile(r"^(?:> ?)+")       # a quoted reply prefixes every line with "> "


def _line_key(line):
    """A line without its line number, quote marks and trailing spaces; indentation is kept."""
    return _QUOTED.sub("", _NUMBERED.sub("", line)).rstrip()


class ContentCache:
    """The tool outputs of one trajectory, and the fold that references them."""

    def __init__(self, threshold=0.5, max_ratio=0.7, min_lines=8, min_run=3, seed=0):
        self.threshold = threshold    # estimated Jaccard needed to try a diff at all
        self.max_ratio = max_ratio    # a fold must be at most this fraction of the original
        self.min_lines = min_lines    # shorter outputs are never folded or referenced
        self.min_run = min_run        # shortest run of repeated lines worth a reference
        self.hasher = MinHasher(seed=seed)
        self.index = LSHIndex()
        self.outputs = []
        self.stats = {"outputs": 0, "folded": 0, "chars_in": 0, "chars_out": 0, "seconds": 0.0}

    def fold(self, label, text):
        """Return `text`, or a shorter exact rewrite of it that references earlier outputs."""
        started = time.perf_counter()
        result = self._fold(label, text)
        self.stats["outputs"] += 1
        self.stats["folded"] += result is not text
        self.stats["chars_in"] += len(text)
        self.stats["chars_out"] += len(result)
        self.stats["seconds"] += time.perf_counter() - started
        return result

    def _fold(self, label, text):
        lines = text.split("\n")
        if len(lines) < self.min_lines:
            return text
        keys = [_line_key(line) for line in lines]
        signature = self.hasher.signature(shingles(tokenize("\n".join(keys), mask_numbers=True)))
        number = len(self.outputs)
        output = {"label": label, "keys": keys,
                  "numbered": all(_NUMBERED.match(line) for line in lines if line),
                  # where each line was last shown in full: (output number, line index)
                  "origin": [(number, i) for i in range(len(lines))]}
        hits = self.index.search(signature, self.threshold)
        folded = self._rewrite(output, lines, self.outputs[hits[0][0]]) if hits else None
        if folded is None:
            output["origin"] = [(number, i) for i in range(len(lines))]
            folded = text
        self.index.add(number, signature)
        self.outputs.append(output)
        return folded

    def _rewrite(self, output, lines, earlier):
        """The folded text of `lines` against one earlier output, or None if it saves too little."""
        keys, origin = output["keys"], output["origin"]
        matcher = difflib.SequenceMatcher(None, earlier["keys"], keys, autojunk=False)
        blocks = [b for b in matcher.get_matching_blocks() if b.size]
        source = {}                           # new line index -> where that text was shown in full
        for block in blocks:
            for offset in range(block.size):
                source[block.b + offset] = earlier["origin"][block.a + offset]
        missing = {}                          # new line index -> note on lines the earlier output had
        for before, after in zip(blocks, blocks[1:]):
            gap = earlier["keys"][before.a + before.size:after.a]
            here = after.b - (before.b + before.size)
            if len(gap) > here and here:      # fewer lines here: say so, the lines above differ
                missing[after.b] = ("[reuse] note: %s has %d lines at this point, this output "
                                    "has only the %d shown above" % (earlier["label"], len(gap), here))
            elif len(gap) > here:             # lines that are simply gone: list them
                missing[after.b] = ("[reuse] note: %d line(s) of %s are absent here: %s"
                                    % (len(gap), earlier["label"], " | ".join(gap[:6])))

        out, repeated, i = [], 0, 0
        while i < len(lines):
            if i in missing:
                out.append(missing[i])
            j = i                             # extend a run of lines shown consecutively before
            while j + 1 < len(lines) and j + 1 not in missing and i in source \
                    and source.get(j + 1) == (source[i][0], source[i][1] + j + 1 - i):
                j += 1
            if i in source and j - i + 1 >= self.min_run:
                out.append(self._reference(output, lines, i, j, source[i]))
                for line in range(i, j + 1):
                    origin[line] = source[line]
                repeated += j - i + 1
                i = j + 1
            else:
                out.append(lines[i])
                i += 1
        if not repeated:
            return None
        head = ("[reuse] %s: %d of its %d lines repeat earlier tool output and are referenced "
                "below instead of shown again; every other line is shown in full."
                % (output["label"], repeated, len(lines)))
        folded = "\n".join([head] + out)
        return folded if len(folded) <= self.max_ratio * len("\n".join(lines)) else None

    def _reference(self, output, lines, first, last, source):
        """One line standing for lines first..last, which repeat an earlier output exactly."""
        shown = self.outputs[source[0]]
        start, count = source[1], last - first + 1
        here = "lines %d-%d" % (first + 1, last + 1) if output["numbered"] else "%d lines" % count
        if shown["numbered"]:
            there = "lines %d-%d of %s" % (start + 1, start + count, shown["label"])
        else:
            there = "%s, from the line \"%s\" to the line \"%s\"" % (
                shown["label"], shown["keys"][start][:60], shown["keys"][start + count - 1][:60])
        quoted = any(_QUOTED.match(_NUMBERED.sub("", line)) for line in lines[first:last + 1])
        return "[reuse] %s: identical to %s%s" % (here, there, " (quoted here with '>')" if quoted else "")


def wrap_tools(tools, cache):
    """Copies of `tools` (objects with `name` and `run`) whose output goes through `cache`."""
    def wrap(tool):
        def run(**kwargs):
            label = " ".join([tool.name] + [str(value) for value in kwargs.values()])[:80]
            return cache.fold(label, str(tool.run(**kwargs)))
        return dataclasses.replace(tool, run=run)
    return [wrap(tool) for tool in tools]
