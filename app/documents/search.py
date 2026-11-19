"""BM25 keyword search over a document's passages.

Plain Python, no model and no extra memory: good enough to find the passages that
mention what the question asks about, and fast for the few thousand passages a
20 MB upload produces.
"""

import math
import re
from collections import Counter

from app.documents.ingest import Passage

K1 = 1.5
B = 0.75

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
        "about",
        "can",
        "could",
        "should",
        "would",
        "please",
        "tell",
        "show",
        "give",
    ]
)


_SUFFIXES = ("izations", "ization", "isations", "isation", "ations", "ation", "ings", "ing",
             "ness", "ments", "ment", "ies", "ed", "es", "ly", "s")  # fmt: skip


def stem(word: str) -> str:
    """A light stemmer: "normalizations", "normalization", "normal" all become "normal"."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())
    return [stem(w) for w in words if w not in _STOPWORDS]


def search(question: str, passages: list[Passage], k: int = 5) -> list[Passage]:
    """The k passages most relevant to the question, best first (none if nothing matches)."""
    query = set(tokenize(question))
    if not query or not passages:
        return []
    docs = [Counter(tokenize(p.text)) for p in passages]
    lengths = [sum(d.values()) for d in docs]
    avg_len = (sum(lengths) / len(lengths)) or 1.0
    df = {t: sum(1 for d in docs if t in d) for t in query}
    n = len(docs)

    scores: list[tuple[float, int]] = []
    for i, (d, length) in enumerate(zip(docs, lengths, strict=True)):
        score = 0.0
        for term in query:
            tf = d.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / avg_len))
        if score > 0:
            scores.append((score, i))
    scores.sort(key=lambda s: (-s[0], s[1]))
    return [passages[i] for _, i in scores[:k]]
