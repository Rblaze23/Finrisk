"""Number guard: every number in LLM-written prose must come from the source facts.

The LLM is only allowed to write prose around numbers it was given. This module extracts every
number from its output and rejects the text if any figure is not supported by the facts.

A written number matches a fact when the fact, rounded to the number of decimals written, equals it
(half-up or banker's rounding; sign ignored). Percent forms match a fact stored as a percentage
or as a fraction. Dates like 2014-10 must appear in the facts as strings. 0, 1 and 100 are always
allowed (reference values such as "a ratio of 1.00" or "100%"). Tokens glued to letters
(R5, Q1, SR11) are not numbers. Numbered-list markers are ignored.

What it does NOT check: whether the interpretation is right. That is why verdicts and conditions
in the report are rule-based and never written by the LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

ALWAYS_ALLOWED = (0.0, 1.0, 100.0)

_DATE = re.compile(r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])\b")
_NUM = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_PERCENT = re.compile(r"\s?(?:%|percent\b|per cent\b)")
_LIST_MARKER = re.compile(r"(?m)^\s*(?:[-*\u2022]\s+)?\d+[.)]\s+")


@dataclass
class ParsedNumber:
    raw: str
    value: float
    decimals: int
    percent: bool
    start: int
    end: int


@dataclass
class GuardResult:
    ok: bool
    n_checked: int
    violations: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            return f"{self.n_checked} numbers verified"
        return "unsupported numbers: " + ", ".join(v["number"] for v in self.violations)


def extract_numbers(text: str) -> tuple[list[ParsedNumber], list[tuple[str, int, int]]]:
    """Return (numbers, dates) found in `text`."""
    text = _LIST_MARKER.sub("", text)
    dates = [(m.group(0), m.start(), m.end()) for m in _DATE.finditer(text)]
    masked = _DATE.sub(lambda m: " " * len(m.group(0)), text)
    nums = []
    for m in _NUM.finditer(masked):
        token = m.group(0)
        digits = token.replace(",", "")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        tail = _PERCENT.match(masked, m.end())
        nums.append(ParsedNumber(raw=token + (tail.group(0) if tail else ""), value=float(digits),
                                 decimals=decimals, percent=bool(tail), start=m.start(), end=m.end()))
    return nums, dates


def collect_allowed(facts) -> tuple[list[float], set[str]]:
    """All numeric values and date strings present anywhere in the facts."""
    values: list[float] = []
    dates: set[str] = set()

    def walk(obj):
        if isinstance(obj, bool) or obj is None:
            return
        if isinstance(obj, (int, float)):
            values.append(float(obj))
        elif isinstance(obj, str):
            nums, ds = extract_numbers(obj)
            values.extend(n.value for n in nums)
            dates.update(d[0] for d in ds)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                walk(v)

    walk(facts)
    return values, dates


def _half_up(x: float, d: int) -> float:
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def _same_at(candidate: float, written: float, d: int) -> bool:
    c, w = abs(candidate), abs(written)
    return abs(_half_up(c, d) - w) < 1e-9 or abs(round(c, d) - w) < 1e-9


def _supported(n: ParsedNumber, allowed: list[float]) -> bool:
    if any(abs(n.value - a) < 1e-12 for a in ALWAYS_ALLOWED):
        return True
    for v in allowed:
        candidates = [v, v * 100.0] if n.percent else [v]
        if any(_same_at(c, n.value, n.decimals) for c in candidates):
            return True
    return False


def check_text(text: str, facts) -> GuardResult:
    """Verify every number and date in `text` against `facts`."""
    allowed, allowed_dates = collect_allowed(facts)
    nums, dates = extract_numbers(text)
    bad = []
    for raw, s, e in dates:
        if raw not in allowed_dates:
            bad.append({"number": raw, "context": text[max(0, s - 30): e + 30].replace("\n", " ")})
    for n in nums:
        if not _supported(n, allowed):
            bad.append({"number": n.raw, "context": text[max(0, n.start - 30): n.end + 30].replace("\n", " ")})
    return GuardResult(ok=not bad, n_checked=len(nums) + len(dates), violations=bad)
