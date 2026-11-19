"""Table RAG: find the right values and rows first, then let the database compute.

Two kinds of retrieval, both feeding the exact SQL engine (never replacing it):

1. Values. Words in the question are fuzzy-matched against the real values of text
   columns ("zarmeen" -> "Zarmeen Khan", "payroll project" -> "D360-M001-Payroll").
   The matches are shown to the planner, and a filter value that matches nothing is
   corrected to the real value afterwards.
2. Topics. A filter {"op": "about", "value": "tax issues"} keeps the rows whose text is
   about a topic by meaning. Keywords shortlist the likely rows, the model checks the
   shortlist, and the database then counts/sums exactly those rows.
"""

import json
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
from cachetools import TTLCache
from rapidfuzz import fuzz, process, utils

from app.documents.search import tokenize
from app.llm.base import ChatMessage, LLMCall, LLMProvider, LLMUnavailableError
from app.query.sql import quote_ident

if TYPE_CHECKING:
    from app.query.executor import ValidatedPlan

MAX_DISTINCT = 20_000  # columns with more distinct values aren't scanned for names
MAX_TOPIC_ROWS = 50_000
VERIFY_ALL = 60  # up to this many distinct texts: the model reads all of them
SHORTLIST = 60
MAX_TEXT = 220  # characters of each text shown to the model
VALUE_SCORE = 88
MAX_HINTS = 12

_cache: TTLCache[str, Any] = TTLCache(maxsize=256, ttl=1800)
_lock = threading.Lock()


def _key(parquet: Path, *parts: str) -> str:
    try:
        stamp = str(parquet.stat().st_mtime_ns)
    except OSError:
        stamp = "0"
    return ":".join([str(parquet), stamp, *parts])


def distinct_values(parquet: Path, column: str) -> list[str]:
    """The distinct non-empty values of a text column (cached; [] if too many)."""
    key = _key(parquet, "distinct", column)
    with _lock:
        if key in _cache:
            return list(_cache[key])
    q = quote_ident(column)
    with duckdb.connect() as con:
        rows = con.execute(
            f"SELECT DISTINCT CAST({q} AS VARCHAR) FROM read_parquet(?) "  # noqa: S608
            f"WHERE {q} IS NOT NULL LIMIT ?",
            [str(parquet), MAX_DISTINCT + 1],
        ).fetchall()
    values = [str(r[0]) for r in rows if str(r[0]).strip()]
    if len(values) > MAX_DISTINCT:
        values = []
    with _lock:
        _cache[key] = values
    return values


# --- 1. values --------------------------------------------------------------------------

_WORD = re.compile(r"[\w\-.@']+", re.UNICODE)
_SKIP = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "by", "count", "data", "do", "does",
        "for", "from", "give", "how", "i", "in", "is", "it", "list", "many", "me", "much",
        "my", "of", "on", "or", "show", "tell", "that", "the", "their", "them", "there",
        "this", "to", "total", "was", "what", "when", "where", "which", "who", "with",
        "you", "all", "any", "per", "each", "vs",
        # Roman Urdu filler words ("Sara ke kitne tasks hain?")
        "ka", "ke", "ki", "ko", "se", "mein", "main", "hai", "hain", "kitne", "kitni",
        "kitna", "kya", "kaun", "konsa", "kis", "batao", "bataen", "dikhao", "wala",
        "wale", "wali", "sab", "aur", "ya", "liye", "kar", "karo", "hua", "hue",
    ]
)  # fmt: skip


def _phrases(question: str) -> list[str]:
    """Words and 2-4 word phrases of the question, without filler words at the ends."""
    words = _WORD.findall(question)
    out: list[str] = []
    for n in (4, 3, 2, 1):
        for i in range(len(words) - n + 1):
            chunk = words[i : i + n]
            if chunk[0].lower() in _SKIP or chunk[-1].lower() in _SKIP:
                continue
            phrase = " ".join(chunk)
            if len(phrase) >= 3 and not phrase.replace(".", "").isdigit():
                out.append(phrase)
    return out


@dataclass
class ValueMatch:
    column: str
    phrase: str
    value: str


def _text_columns(profile: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        c
        for c in profile
        if c.get("kind") == "text" and 0 < int(c.get("distinct_count") or 0) <= MAX_DISTINCT
    ]


def _is_label(values: list[str]) -> bool:
    """Short labels (names, statuses, projects), not free text like descriptions."""
    sample = values[:200]
    return bool(sample) and sum(len(v) for v in sample) / len(sample) <= 60


def match_values(question: str, parquet: Path, profile: list[dict[str, Any]]) -> list[ValueMatch]:
    phrases = _phrases(question)
    if not phrases:
        return []
    found: dict[tuple[str, str], ValueMatch] = {}
    lowered_q = question.lower()
    for col in _text_columns(profile):
        values = distinct_values(parquet, str(col["name"]))
        if not _is_label(values):
            continue
        for phrase in phrases:
            for value, score, _ in process.extract(
                phrase,
                values,
                scorer=fuzz.WRatio,
                processor=utils.default_process,
                limit=3,
                score_cutoff=VALUE_SCORE,
            ):
                if len(value) < 3 or value.lower() in _SKIP:
                    continue
                # Skip values already written exactly in the question: nothing to fix.
                exact = value.lower() in lowered_q
                key = (str(col["name"]), value)
                if key not in found and (not exact or score >= 99):
                    found[key] = ValueMatch(str(col["name"]), phrase, value)
            if len(found) >= MAX_HINTS * 2:
                break
    return list(found.values())[:MAX_HINTS]


def value_hints(question: str, parquet: Path, profile: list[dict[str, Any]]) -> str:
    """Planner notes: real values that match words in the question."""
    matches = match_values(question, parquet, profile)
    if not matches:
        return ""
    lines = ["Values in this table that match words in the question (use them exactly):"]
    for m in matches:
        lines.append(f"- {m.column}: {json.dumps(m.value, ensure_ascii=False)} "
                     f"(for {json.dumps(m.phrase, ensure_ascii=False)})")  # fmt: skip
    return "\n".join(lines)


def closest_values(value: str, values: list[str]) -> list[str]:
    """Real values a filter value probably meant: exact (any case) first, then fuzzy."""
    lowered = value.strip().lower()
    exact = [v for v in values if v.lower() == lowered]
    if exact:
        return exact
    hits = process.extract(
        value, values, scorer=fuzz.WRatio, processor=utils.default_process, limit=6, score_cutoff=86
    )
    if not hits:
        return []
    best = hits[0][1]
    return [v for v, score, _ in hits if score >= best - 2]  # ties: "Sara" -> 2 Saras


# --- 2. topics ----------------------------------------------------------------------------


@dataclass
class TopicMatch:
    rows: list[int]  # file row numbers
    texts: int  # distinct texts in the column
    checked: int  # how many of them the model read
    method: str  # "ai" (model checked) | "keywords" (model unavailable)
    calls: list[LLMCall] = field(default_factory=list)


EXPAND_SYSTEM = """List words and short phrases that a text about the given topic would
likely contain: synonyms, related terms, abbreviations, common misspellings.
Reply with JSON {"keywords": [...]} (at most 12)."""

VERIFY_SYSTEM = """You decide which numbered texts are about a topic. Judge by meaning,
not by exact words. Include every text that is clearly about the topic; leave out texts
that only share a word. Reply with JSON {"matches": [numbers]}; [] if none match."""

_KEYWORDS_SCHEMA = {
    "type": "object",
    "properties": {"keywords": {"type": "array", "items": {"type": "string"}}},
    "required": ["keywords"],
}
_MATCHES_SCHEMA = {
    "type": "object",
    "properties": {"matches": {"type": "array", "items": {"type": "integer"}}},
    "required": ["matches"],
}


def _call(llm: LLMProvider, system: str, user: str, schema: dict[str, Any]) -> tuple[Any, LLMCall]:
    import time

    started = time.perf_counter()
    out = llm.complete([ChatMessage("system", system), ChatMessage("user", user)],
                       json_schema=schema, max_tokens=400)  # fmt: skip
    call = LLMCall("retrieve", llm.name, llm.model, out.prompt_tokens, out.completion_tokens,
                   out.latency_ms or int((time.perf_counter() - started) * 1000), True)  # fmt: skip
    try:
        return json.loads(out.text), call
    except ValueError:
        return {}, call


def _bm25(query: list[str], texts: list[str]) -> list[float]:
    docs = [Counter(tokenize(t)) for t in texts]
    lengths = [sum(d.values()) for d in docs]
    avg = (sum(lengths) / len(lengths)) if lengths else 1.0
    n = len(docs)
    terms = set(query)
    df = {t: sum(1 for d in docs if t in d) for t in terms}
    scores = []
    for d, length in zip(docs, lengths, strict=True):
        s = 0.0
        for t in terms:
            tf = d.get(t, 0)
            if tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / (avg or 1.0)))
        scores.append(s)
    return scores


def _column_texts(parquet: Path, column: str) -> dict[str, list[int]]:
    key = _key(parquet, "texts", column)
    with _lock:
        if key in _cache:
            return dict(_cache[key])
    q = quote_ident(column)
    with duckdb.connect() as con:
        rows = con.execute(
            f"SELECT file_row_number, CAST({q} AS VARCHAR) "  # noqa: S608
            f"FROM read_parquet(?, file_row_number = true) WHERE {q} IS NOT NULL LIMIT ?",
            [str(parquet), MAX_TOPIC_ROWS],
        ).fetchall()
    texts: dict[str, list[int]] = {}
    for row_number, text in rows:
        if str(text).strip():
            texts.setdefault(str(text), []).append(int(row_number))
    with _lock:
        _cache[key] = texts
    return texts


def match_topic(llm: LLMProvider, parquet: Path, column: str, topic: str) -> TopicMatch:
    """Rows of `column` whose text is about `topic`, by meaning."""
    texts = _column_texts(parquet, column)
    uniques = list(texts)
    calls: list[LLMCall] = []
    words = tokenize(topic)
    try:
        if len(uniques) <= VERIFY_ALL:
            shortlist = uniques
        else:
            data, call = _call(llm, EXPAND_SYSTEM, f"Topic: {topic}", _KEYWORDS_SCHEMA)
            calls.append(call)
            extra = [str(k) for k in data.get("keywords", [])][:12]
            query = words + [t for k in extra for t in tokenize(k)]
            scores = _bm25(query, uniques)
            ranked = sorted(range(len(uniques)), key=lambda i: -scores[i])
            shortlist = [uniques[i] for i in ranked[:SHORTLIST] if scores[i] > 0]
        if not shortlist:
            return TopicMatch([], len(uniques), 0, "ai", calls)
        listing = "\n".join(f"{i}. {t[:MAX_TEXT]}" for i, t in enumerate(shortlist, start=1))
        data, call = _call(llm, VERIFY_SYSTEM, f"Topic: {topic}\n\nTexts:\n{listing}",
                           _MATCHES_SCHEMA)  # fmt: skip
        calls.append(call)
        picked = {int(n) for n in data.get("matches", []) if isinstance(n, int | float)}
        chosen = [shortlist[n - 1] for n in sorted(picked) if 1 <= n <= len(shortlist)]
        method, checked = "ai", len(shortlist)
    except LLMUnavailableError:
        # No model: plain keyword matching on the topic's own words.
        scores = _bm25(words, uniques)
        chosen = [t for t, s in zip(uniques, scores, strict=True) if s > 0]
        method, checked = "keywords", 0
    rows = sorted(r for t in chosen for r in texts[t])
    return TopicMatch(rows, len(uniques), checked, method, calls)


def describe_topic_match(column: str, topic: str, m: TopicMatch) -> str:
    rows = f"{len(m.rows):,} row{'s' if len(m.rows) != 1 else ''}"
    if m.method == "keywords":
        return (f"“{topic}” matched {rows} of {column} by keywords (the AI was "
                "unavailable to check meaning).")  # fmt: skip
    if m.checked >= m.texts:
        return f"“{topic}”: the AI read all {m.texts:,} {column} texts; {rows} match."
    return (f"“{topic}”: the AI checked the {m.checked} most likely of "
            f"{m.texts:,} {column} texts; {rows} match.")  # fmt: skip


# --- applying both to a validated plan --------------------------------------------------


@dataclass
class Retrieval:
    plan: "ValidatedPlan"
    notes: list[str] = field(default_factory=list)  # shown under the answer
    topic_rows: dict[int, list[int]] = field(default_factory=dict)  # filter index -> rows
    calls: list[LLMCall] = field(default_factory=list)


def apply_retrieval(
    llm: LLMProvider,
    vp: "ValidatedPlan",
    parquet: Path,
    *,
    topic_rows: dict[int, list[int]] | None = None,
) -> Retrieval:
    """Correct filter values to real ones and turn topic filters into row numbers.

    `topic_rows`: rows already matched for this plan (e.g. when re-running an answer),
    so the model isn't asked again.
    """
    from app.query.executor import ValidatedPlan

    out = Retrieval(vp)
    filters = list(vp.plan.filters)
    values = list(vp.filter_values)
    for i, f in enumerate(filters):
        kind = vp.kinds.get(f.column)
        if f.op == "about":
            if topic_rows is not None and i in topic_rows:
                values[i] = topic_rows[i]
            else:
                m = match_topic(llm, parquet, f.column, str(values[i]))
                out.calls += m.calls
                out.notes.append(describe_topic_match(f.column, str(values[i]), m))
                values[i] = m.rows
            out.topic_rows[i] = values[i]
            continue
        if kind != "text" or f.op not in ("=", "!=", "in") or values[i] is None:
            continue
        known = distinct_values(parquet, f.column)
        if not known:
            continue
        wanted = values[i] if isinstance(values[i], list) else [values[i]]
        fixed: list[str] = []
        for w in wanted:
            real = closest_values(str(w), known)
            if not real:
                fixed.append(str(w))
                continue
            if [r.lower() for r in real] != [str(w).lower()]:
                shown = ", ".join(f"\u201c{r}\u201d" for r in real)
                out.notes.append(f"Matched \u201c{w}\u201d to {shown} in {f.column}.")
            fixed += [r for r in real if r not in fixed]
        if f.op == "in" or len(fixed) > 1:
            op = "in" if f.op != "!=" else "!="
            if op == "!=" and len(fixed) > 1:
                continue  # "not any of several": leave as asked
            filters[i] = f.model_copy(update={"op": op, "value": fixed if op == "in" else fixed[0]})
            values[i] = fixed if op == "in" else fixed[0]
        else:
            filters[i] = f.model_copy(update={"value": fixed[0]})
            values[i] = fixed[0]
    plan = vp.plan.model_copy(update={"filters": filters})
    out.plan = ValidatedPlan(plan, vp.kinds, vp.date_range, values)
    return out
