"""Final score per the checklist calculation procedure (page 2).

1. N/A only where allowed.  2. Yes = full weight, No = 0; scales are proportional, top step = full weight.
3./4. N/A criteria are excluded and their weight is redistributed proportionally (max stays 100).
5. Round to two decimals; no applicable weighted criteria -> no score. Severe errors are a separate flag.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

from .checklist import BY_CODE, CODES


class ScoreError(ValueError):
    pass


def check_value(code: str, value) -> None:
    c = BY_CODE.get(code)
    if c is None:
        raise ScoreError(f"unknown criterion {code!r}")
    # bool is a subclass of int in Python: keep True/False and 1/0 apart explicitly.
    ok = any(value is a if isinstance(a, bool) or a is None else (type(value) is int and value == a)
             for a in c.allowed)
    if not ok:
        raise ScoreError(f"{code}: value {value!r} not in {list(c.allowed)}")


def final_score(values: dict) -> float | None:
    """values: criterion code -> value (True/False/None/int). Returns 0..100 rounded to 2 decimals."""
    missing = [code for code in CODES if code not in values]
    if missing:
        raise ScoreError(f"missing criteria: {', '.join(missing)}")
    earned = Fraction(0)
    applicable = 0
    for code in CODES:
        c, v = BY_CODE[code], values[code]
        check_value(code, v)
        if c.weight == 0 or v is None:
            continue
        ratio = Fraction(v, c.scale_max) if c.scale_max else Fraction(1 if v else 0)
        earned += ratio * c.weight
        applicable += c.weight
    if applicable == 0:
        return None
    exact = earned * 100 / applicable
    rounded = (Decimal(exact.numerator) / Decimal(exact.denominator)).quantize(Decimal("0.01"), ROUND_HALF_UP)
    return float(rounded)
