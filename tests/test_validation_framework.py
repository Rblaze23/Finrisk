"""Tests for the results contract, grading and the out-of-time helpers."""
import json

import numpy as np
import pandas as pd

from finrisk.validation import results as R
from finrisk.validation.out_of_time import performance_by_period


def test_grade_higher_is_worse():
    assert R.grade(0.05, 0.10, 0.20) == "pass"
    assert R.grade(0.10, 0.10, 0.20) == "pass"        # boundary is inclusive
    assert R.grade(0.15, 0.10, 0.20) == "warn"
    assert R.grade(0.30, 0.10, 0.20) == "fail"


def test_grade_lower_is_worse():
    assert R.grade(0.40, 0.25, 0.15, higher_is_worse=False) == "pass"
    assert R.grade(0.20, 0.25, 0.15, higher_is_worse=False) == "warn"
    assert R.grade(0.10, 0.25, 0.15, higher_is_worse=False) == "fail"


def test_grade_uncomputable_value_is_warn():
    assert R.grade(float("nan"), 0.1, 0.2) == "warn"
    assert R.grade(None, 0.1, 0.2) == "warn"


def test_results_serialise_to_json_even_with_numpy_nan_and_dataframes():
    r = R.make("m", "t", "x", np.float32(0.5), "pass", 0.1, 0.2, "rule",
               {"arr": np.array([1, 2]), "nan": float("nan"), "n": np.int64(3),
                "df": pd.DataFrame({"a": [1.0, np.nan]}), "when": pd.Timestamp("2015-01-01")})
    d = r.to_dict()
    text = json.dumps(d)                              # must not raise
    back = json.loads(text)
    assert back["details"]["nan"] is None
    assert back["details"]["df"][1]["a"] is None
    assert back["details"]["arr"] == [1, 2]


def test_make_turns_nan_value_into_none():
    assert R.make("m", "t", "x", float("nan"), "warn").value is None


def test_thresholds_are_ordered_sensibly():
    for key, (warn, fail) in R.THRESHOLDS.items():
        if key in R.LOWER_IS_WORSE:
            assert fail < warn, key                   # lower is worse
        else:
            assert warn < fail, key


def test_detail_tests_are_flagged_as_diagnostics():
    assert R.make("m", "stability_detail", "x", 1.0, "warn").__class__ is R.TestResult
    r = R.TestResult("m", "stability_detail", "x", 1.0, "warn")
    assert r.is_detail
    assert not R.TestResult("m", "stability", "x", 1.0, "warn").is_detail


def test_performance_by_period_returns_quarterly_gini():
    rng = np.random.default_rng(0)
    n = 40_000
    dates = pd.date_range("2014-01-01", "2015-12-31", periods=n)
    p = rng.beta(2, 10, n)
    y = (rng.uniform(size=n) < p).astype(int)
    t = performance_by_period(dates, y, p)
    assert len(t) == 8
    assert (t["gini"] > 0.2).all()
