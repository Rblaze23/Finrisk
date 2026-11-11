"""Unit tests for WoE binning and Information Value."""
import numpy as np
import pandas as pd

from finrisk.features.woe import MISSING, OTHER, WoETransformer


def _make(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, n)
    p = 1 / (1 + np.exp(0.05 * (x - 50) + 1.5))      # higher x -> lower default
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"x": x}), y


def test_numeric_woe_is_monotonic_increasing_in_a_protective_variable():
    X, y = _make()
    t = WoETransformer(["x"], []).fit(X, y)
    woe = t.bin_table("x")["woe"].to_numpy()
    assert np.all(np.diff(woe) >= 0)          # higher x -> more "good" -> higher WoE
    assert t.bins_["x"].iv > 0.1


def test_noise_feature_has_low_iv():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"noise": rng.normal(size=20000)})
    y = (rng.uniform(size=20000) < 0.15).astype(int)
    t = WoETransformer(["noise"], []).fit(X, y)
    assert t.bins_["noise"].iv < 0.02


def test_bins_respect_minimum_size():
    X, y = _make()
    t = WoETransformer(["x"], [], min_bin_pct=0.10).fit(X, y)
    assert (t.bin_table("x")["n"] >= 0.10 * len(X)).all()


def test_missing_values_get_their_own_bin():
    X, y = _make()
    X.loc[X.index[:3000], "x"] = np.nan
    t = WoETransformer(["x"], []).fit(X, y)
    assert MISSING in t.bin_table("x")["bin"].tolist()
    out = t.transform(X)
    assert out["x"].notna().all()


def test_missing_at_transform_time_without_missing_in_fit_is_neutral():
    X, y = _make()
    t = WoETransformer(["x"], []).fit(X, y)
    X2 = pd.DataFrame({"x": [np.nan, 10.0]})
    out = t.transform(X2)["x"].to_numpy()
    assert out[0] == 0.0


def test_rare_categories_are_grouped_and_unseen_levels_do_not_crash():
    rng = np.random.default_rng(2)
    n = 10000
    cat = rng.choice(["A", "B", "C", "rare1", "rare2"], n, p=[0.4, 0.3, 0.28, 0.01, 0.01])
    p = np.select([cat == "A", cat == "B", cat == "C"], [0.05, 0.15, 0.30], 0.5)
    y = (rng.uniform(size=n) < p).astype(int)
    X = pd.DataFrame({"c": cat})
    t = WoETransformer([], ["c"], min_bin_pct=0.05).fit(X, y)
    assert OTHER in t.bin_table("c")["bin"].tolist()
    out = t.transform(pd.DataFrame({"c": ["A", "never_seen", None]}))["c"]
    assert out.notna().all()


def test_woe_sign_convention_good_over_bad():
    # category with a very low default rate must have a clearly positive WoE
    n = 10000
    cat = np.array(["safe"] * (n // 2) + ["risky"] * (n // 2))
    rng = np.random.default_rng(3)
    y = np.where(cat == "safe", rng.uniform(size=n) < 0.02, rng.uniform(size=n) < 0.40).astype(int)
    t = WoETransformer([], ["c"]).fit(pd.DataFrame({"c": cat}), y)
    tab = t.bin_table("c").set_index("bin")
    assert tab.loc["safe", "woe"] > 0 > tab.loc["risky", "woe"]
