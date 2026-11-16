"""MinHash signatures and an LSH index over three-word shingles.

A text becomes a set of three-word shingles; a signature keeps, for each of 128
hash functions, the smallest hash of any shingle. Two signatures agree in a
component with probability equal to the Jaccard similarity of the two sets, so
the fraction of agreeing components estimates it. The LSH index cuts a signature
into bands and files it under each band, so a lookup only meets records that
share at least one whole band.

Only the standard library is used; hashlib supplies the base hash of a shingle.
"""

import hashlib
import random
import re
from collections import defaultdict

NUM_PERM = 128           # hash functions per signature
SHINGLE_WORDS = 3        # words per shingle
PRIME = (1 << 61) - 1    # modulus of the universal hash family
_SIGN = r"[+\-\N{MINUS SIGN}]"
_TIME = r"\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:z|[+\-]\d{2}:?\d{2})?"
_DATE = rf"\d{{4}}-\d{{2}}-\d{{2}}(?:[t ]{_TIME})?"
_DIGITS = r"(?:\d{1,3}(?:,\d{3})+|\d+(?:_\d+)*)"
_NUMBER = rf"{_SIGN}?(?:{_DIGITS}(?:\.\d+)?|\.\d+)(?:e{_SIGN}?\d+)?"
_TOKEN = re.compile(
    rf"{_DATE}|{_TIME}|{_SIGN}?\d+(?:\.\d+){{2,}}|{_NUMBER}"
    r"|(?P<word>[^\W\d]\w*(?:[-.]\w+)*)"  # words and identifiers, including E-1042 and v3.1.4
    rf"|{_SIGN}"                  # also retain a sign separated from its number
)


def tokenize(text):
    """Lower-case tokens, preserving numeric spelling rather than masking values.

    Signed integers, decimals, scientific notation, grouped digits, ISO dates,
    times and dotted versions/IPs stay intact. No float conversion or numeric
    rounding is used: -10 differs from +10, 3.14 from 3 14, and 1e-3 from 1e3.

    An identifier that mixes letters and digits is compared without its hyphens,
    so E-1042 and E1042 are one token (E-1042 and E-1043 are still two). Only
    hyphens inside such identifiers are dropped: numbers, dates, dots and plain
    words keep their spelling.
    """
    tokens = []
    for match in _TOKEN.finditer(str(text).lower()):
        token = match.group()
        if match.lastgroup == "word" and "-" in token and any(c.isdigit() for c in token):
            token = token.replace("-", "")
        tokens.append(token)
    return tokens


def shingles(tokens, k=SHINGLE_WORDS):
    """The set of k-word shingles of a token list (one short shingle if it has < k words)."""
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


class MinHasher:
    """`num_perm` seeded hash functions h(x) = (a*x + b) mod PRIME over 64-bit shingle hashes."""

    def __init__(self, num_perm=NUM_PERM, seed=0):
        rng = random.Random(seed)
        self.num_perm = num_perm
        self.params = [(rng.randrange(1, PRIME), rng.randrange(PRIME)) for _ in range(num_perm)]

    def signature(self, shingle_set):
        """The tuple of per-function minima; an empty set gets a signature that matches nothing."""
        base = [int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")
                for s in shingle_set]
        if not base:
            return tuple(-1 - i for i in range(self.num_perm))
        return tuple(min((a * x + b) % PRIME for x in base) for a, b in self.params)


def similarity(sig_a, sig_b):
    """Estimated Jaccard similarity: the fraction of signature components that agree."""
    return sum(1 for x, y in zip(sig_a, sig_b) if x == y) / len(sig_a)


class LSHIndex:
    """Banded LSH: two signatures become candidates when any one band is identical.

    With `bands` bands of r = num_perm / bands rows, a pair of Jaccard similarity s
    is retrieved with probability 1 - (1 - s**r)**bands. The default (32 bands of 4)
    finds 99 % of pairs at s = 0.6 and 87 % at s = 0.5, and few below 0.3.
    """

    def __init__(self, bands=32, num_perm=NUM_PERM):
        if num_perm % bands:
            raise ValueError("bands must divide num_perm")
        self.bands = bands
        self.rows = num_perm // bands
        self.buckets = defaultdict(list)
        self.signatures = {}

    def _band_keys(self, sig):
        return [(i, sig[i * self.rows:(i + 1) * self.rows]) for i in range(self.bands)]

    def add(self, key, sig):
        self.signatures[key] = sig
        for band in self._band_keys(sig):
            self.buckets[band].append(key)

    def search(self, sig, threshold):
        """[(key, estimated similarity)] for candidates at or above `threshold`, best first."""
        candidates = {key for band in self._band_keys(sig) for key in self.buckets.get(band, ())}
        scored = [(key, similarity(sig, self.signatures[key])) for key in candidates]
        return sorted((hit for hit in scored if hit[1] >= threshold),
                      key=lambda hit: (-hit[1], str(hit[0])))
