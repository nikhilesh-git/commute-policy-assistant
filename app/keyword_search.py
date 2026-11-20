"""BM25 keyword ranking, used by hybrid retrieval (RETRIEVAL_MODE=hybrid).

Why: dense embeddings are good at meaning but weak at exact terms. A question
like "Can my interview candidate use the cab?" shares the words "interview
candidate" with the Guest travel section, but MiniLM ranked that section 6th.
BM25 scores exact word overlap, weighted so rare words count more.

BM25 for a document d and query q:
  score(d, q) = sum over terms t in q of
      idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * len(d) / avg_len))
  idf(t)      = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
k1 limits how much repeating a word helps; b controls length normalisation.
"""

import math
import re
from collections import Counter

K1 = 1.5
B = 0.75

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its me my "
    "of on or the this to was what when where which who will with you your".split()
)


def _stem(token: str) -> str:
    """Tiny plural folding so 'candidate' matches 'candidates'. Deliberately crude."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, documents: list[str]):
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        n = len(self.docs)
        df = Counter(term for doc in self.docs for term in doc)
        self.idf = {t: math.log(1 + (n - k + 0.5) / (k + 0.5)) for t, k in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = set(tokenize(query))
        out = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / self.avg_len))
            out.append(s)
        return out

    def top(self, query: str, n: int) -> list[int]:
        """Indices of the n best documents with a score above zero, best first."""
        scored = [(s, i) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [i for _, i in scored[:n]]
