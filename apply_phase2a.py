#!/usr/bin/env python3
"""
Phase 2a for FinRisk Lab: features, WoE binning, scorecard baseline.

Run from the ROOT of your repo (after apply_phase1.py):

    python apply_phase2a.py            # fills the stub files
    python apply_phase2a.py --force    # overwrite even if you edited them
"""
import sys
from pathlib import Path

ROOT = Path(".").resolve()
FORCE = "--force" in sys.argv

FILES = {}

# ---------------------------------------------------------------------------
FILES["sql/03_features.sql"] = r'''-- 03_features.sql
-- staging.model_sample (loans + split) -> features.loan_features
--
-- Only information known at origination is used. Post-origination columns
-- (payments, recoveries, ...) are deliberately NOT carried over.
-- grade / sub_grade / int_rate are LendingClub's own underwriting outputs:
-- they are kept here but trained with and without (see features/build.py).

DROP TABLE IF EXISTS features.loan_features CASCADE;

CREATE TABLE features.loan_features AS
SELECT
    m.loan_id,
    m.split,
    m.issue_d,
    m.issue_year,
    m.is_default,

    -- application
    m.loan_amnt,
    m.annual_inc,
    m.dti,
    m.emp_length_years,
    m.home_ownership,
    m.verification_status,
    m.purpose,
    m.addr_state,
    m.initial_list_status,

    -- credit bureau
    ((m.fico_range_low + m.fico_range_high) / 2.0)::double precision        AS fico_mid,
    m.delinq_2yrs,
    m.inq_last_6mths,
    m.mths_since_last_delinq,
    m.mths_since_last_record,
    m.open_acc,
    m.pub_rec,
    m.revol_bal,
    m.revol_util,
    m.total_acc,
    m.mort_acc,
    m.pub_rec_bankruptcies,
    m.tot_cur_bal,
    m.total_rev_hi_lim,
    m.avg_cur_bal,
    m.bc_util,
    m.num_bc_sats,

    -- engineered
    ((m.issue_d - m.earliest_cr_line_d) / 365.25)::double precision          AS credit_history_years,
    CASE WHEN m.annual_inc > 0 THEN m.loan_amnt / m.annual_inc END           AS loan_to_income,
    CASE WHEN m.annual_inc > 0 THEN m.revol_bal / m.annual_inc END           AS revol_bal_to_income,

    -- lender-assigned (LendingClub's own risk assessment)
    m.grade,
    m.sub_grade,
    m.int_rate
FROM staging.model_sample m;

ALTER TABLE features.loan_features ADD PRIMARY KEY (loan_id);
CREATE INDEX idx_loan_features_split ON features.loan_features (split);
ANALYZE features.loan_features;
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/features/build.py"] = r'''"""Feature table, feature sets and loaders.

Usage:
    python -m finrisk.features.build        # build features.loan_features + missingness report

In other modules:
    from finrisk.features.build import load_features, get_feature_columns
    train = load_features("train")
    numeric, categorical = get_feature_columns("independent")
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from finrisk.config import get_settings
from finrisk.data.columns import LENDER_ASSIGNED_COLUMNS
from finrisk.db import connect, execute_sql, get_engine

pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 100)

NUMERIC_FEATURES = [
    "loan_amnt", "annual_inc", "dti", "emp_length_years", "fico_mid",
    "delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq",
    "mths_since_last_record", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "mort_acc", "pub_rec_bankruptcies", "tot_cur_bal",
    "total_rev_hi_lim", "avg_cur_bal", "bc_util", "num_bc_sats",
    "credit_history_years", "loan_to_income", "revol_bal_to_income",
]
CATEGORICAL_FEATURES = [
    "home_ownership", "verification_status", "purpose", "addr_state",
    "initial_list_status",
]
LENDER_NUMERIC = ["int_rate"]
LENDER_CATEGORICAL = ["grade", "sub_grade"]

assert set(LENDER_NUMERIC + LENDER_CATEGORICAL) == set(LENDER_ASSIGNED_COLUMNS)

# independent  : what a bank with its own data would have
# with_lender  : adds LendingClub's own grade / sub_grade / int_rate
FEATURE_SETS = {
    "independent": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    "with_lender": (NUMERIC_FEATURES + LENDER_NUMERIC,
                    CATEGORICAL_FEATURES + LENDER_CATEGORICAL),
}


def get_feature_columns(feature_set: str) -> tuple[list[str], list[str]]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {list(FEATURE_SETS)}")
    num, cat = FEATURE_SETS[feature_set]
    return list(num), list(cat)


def build_features() -> None:
    sql = (get_settings().sql_dir / "03_features.sql").read_text(encoding="utf-8")
    execute_sql(sql)


def load_features(split: str | None = None) -> pd.DataFrame:
    """Load features.loan_features (optionally one split) with numeric dtypes enforced."""
    query = "SELECT * FROM features.loan_features"
    params = {}
    if split is not None:
        query += " WHERE split = :split"
        params = {"split": split}
    df = pd.read_sql(text(query), get_engine(), params=params)
    numeric = set(NUMERIC_FEATURES + LENDER_NUMERIC)
    for c in df.columns:
        if c in numeric:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    connect().close()  # friendly error if Docker is down
    print("Building features.loan_features ...")
    build_features()

    df = load_features()
    print(f"\n{len(df):,} loans\n")
    print(df.groupby("split")["is_default"].agg(n_loans="size", default_rate="mean")
          .reindex(["train", "valid", "oot", "monitor"]).round(4).to_string())

    feats = NUMERIC_FEATURES + CATEGORICAL_FEATURES + LENDER_NUMERIC + LENDER_CATEGORICAL
    miss = df[feats].isna().groupby(df["issue_year"]).mean().T * 100
    miss = miss[miss.max(axis=1) > 5].round(1)
    print("\nMissing values (%) by issue year, for features with more than 5% missing anywhere:")
    print(miss.to_string())
    print("\nNext: python -m finrisk.models.scorecard")


if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/features/woe.py"] = r'''"""Binning and Weight of Evidence (WoE) / Information Value (IV).

Convention (Siddiqi):  WoE = ln(%good / %bad)
    -> HIGHER WoE means LOWER risk. A logistic regression of default on WoE
       features therefore has NEGATIVE coefficients.

Numeric variables : quantile pre-bins, then adjacent bins are merged until every
                    bin is large enough and (optionally) the bad rate is monotonic.
Categorical       : rare levels are grouped into OTHER.
Missing values    : get their own bin when present in the fitting data, else WoE 0.
Smoothing         : +0.5 per bin and class, so WoE is always finite.

Fit on the TRAIN split only, then transform any split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MISSING = "__MISSING__"
OTHER = "__OTHER__"


def woe_iv(good, bad) -> tuple[np.ndarray, np.ndarray]:
    """WoE and IV contribution for each bin (arrays of good / bad counts)."""
    good = np.asarray(good, dtype=float)
    bad = np.asarray(bad, dtype=float)
    k = len(good)
    g = (good + 0.5) / (good.sum() + 0.5 * k)
    b = (bad + 0.5) / (bad.sum() + 0.5 * k)
    woe = np.log(g / b)
    return woe, (g - b) * woe


def _numeric_labels(edges: np.ndarray) -> list[str]:
    f = lambda v: f"{v:.4g}"
    if len(edges) == 0:
        return ["all values"]
    labels = [f"<= {f(edges[0])}"]
    labels += [f"({f(a)}, {f(b)}]" for a, b in zip(edges[:-1], edges[1:])]
    labels.append(f"> {f(edges[-1])}")
    return labels


@dataclass
class FeatureBins:
    name: str
    kind: str                              # "numeric" | "categorical"
    table: pd.DataFrame                    # bin, n, n_bad, bad_rate, woe, iv
    edges: np.ndarray | None = None        # numeric: inner bin edges
    woe_values: np.ndarray | None = None   # numeric: WoE per bin
    missing_woe: float = 0.0
    mapping: dict | None = None            # categorical: level -> WoE
    other_woe: float | None = None

    @property
    def iv(self) -> float:
        return float(self.table["iv"].sum())

    def transform(self, x: pd.Series) -> np.ndarray:
        if self.kind == "numeric":
            v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
            idx = np.searchsorted(self.edges, v, side="left")
            out = self.woe_values[idx]
            return np.where(np.isnan(v), self.missing_woe, out)
        s = x.astype(object).where(x.notna(), MISSING).astype(str)
        fill = self.other_woe if self.other_woe is not None else 0.0
        return s.map(self.mapping).fillna(fill).to_numpy(dtype=float)


class WoETransformer:
    def __init__(self, numeric: list[str], categorical: list[str],
                 max_prebins: int = 20, max_bins: int = 8,
                 min_bin_pct: float = 0.05, monotonic: bool = True):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.max_prebins = max_prebins
        self.max_bins = max_bins
        self.min_bin_pct = min_bin_pct
        self.monotonic = monotonic
        self.bins_: dict[str, FeatureBins] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y) -> "WoETransformer":
        yv = np.asarray(y).astype(int)
        if len(yv) != len(X):
            raise ValueError("X and y must have the same length")
        for c in self.numeric:
            self.bins_[c] = self._fit_numeric(c, X[c], yv)
        for c in self.categorical:
            self.bins_[c] = self._fit_categorical(c, X[c], yv)
        return self

    def _merge_bins(self, edges, good, bad, n_total):
        edges, good, bad = list(edges), list(good), list(bad)

        def merge(i):  # merge bin i with bin i+1
            good[i] += good.pop(i + 1)
            bad[i] += bad.pop(i + 1)
            edges.pop(i)

        def rates():
            return [b / (g + b) if g + b > 0 else 0.0 for g, b in zip(good, bad)]

        min_n = self.min_bin_pct * n_total

        # 1) merge bins that are too small, with the neighbour of closest bad rate
        while len(good) > 1:
            sizes = [g + b for g, b in zip(good, bad)]
            i = int(np.argmin(sizes))
            if sizes[i] >= min_n:
                break
            r = rates()
            if i == 0:
                j = 0
            elif i == len(good) - 1:
                j = i - 1
            else:
                j = i - 1 if abs(r[i] - r[i - 1]) <= abs(r[i] - r[i + 1]) else i
            merge(j)

        # 2) enforce a monotonic bad-rate trend
        if self.monotonic:
            while len(good) > 2:
                r = rates()
                corr = np.corrcoef(np.arange(len(r)), r)[0, 1]
                direction = -1.0 if (np.isfinite(corr) and corr < 0) else 1.0
                bad_pair = next((i for i in range(len(r) - 1)
                                 if direction * (r[i + 1] - r[i]) < 0), None)
                if bad_pair is None:
                    break
                merge(bad_pair)

        # 3) cap the number of bins
        while len(good) > self.max_bins:
            r = rates()
            i = int(np.argmin([abs(r[k + 1] - r[k]) for k in range(len(r) - 1)]))
            merge(i)

        return np.array(edges, dtype=float), np.array(good), np.array(bad)

    def _fit_numeric(self, name, x, y) -> FeatureBins:
        v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        miss = np.isnan(v)
        xs, ys = v[~miss], y[~miss]

        if xs.size == 0 or np.unique(xs).size < 2:
            edges = np.array([], dtype=float)
            good = np.array([(ys == 0).sum()], dtype=float)
            bad = np.array([(ys == 1).sum()], dtype=float)
        else:
            probs = np.linspace(0, 1, self.max_prebins + 1)[1:-1]
            edges = np.unique(np.quantile(xs, probs))
            edges = edges[edges < xs.max()]
            idx = np.searchsorted(edges, xs, side="left")
            k = len(edges) + 1
            good = np.bincount(idx, weights=(ys == 0), minlength=k)
            bad = np.bincount(idx, weights=(ys == 1), minlength=k)
            edges, good, bad = self._merge_bins(edges, good, bad, len(v))

        labels = _numeric_labels(edges)
        has_missing = bool(miss.any())
        g_all, b_all = list(good), list(bad)
        if has_missing:
            labels.append(MISSING)
            g_all.append(float((y[miss] == 0).sum()))
            b_all.append(float((y[miss] == 1).sum()))

        woe, iv = woe_iv(g_all, b_all)
        n = np.array(g_all) + np.array(b_all)
        table = pd.DataFrame({
            "bin": labels, "n": n.astype(int), "n_bad": np.array(b_all).astype(int),
            "bad_rate": np.array(b_all) / np.maximum(n, 1), "woe": woe, "iv": iv,
        })
        n_num_bins = len(edges) + 1
        return FeatureBins(
            name=name, kind="numeric", table=table, edges=edges,
            woe_values=woe[:n_num_bins],
            missing_woe=float(woe[-1]) if has_missing else 0.0,
        )

    def _fit_categorical(self, name, x, y) -> FeatureBins:
        s = x.astype(object).where(x.notna(), MISSING).astype(str).to_numpy()
        df = pd.DataFrame({"lvl": s, "bad": y})
        g = df.groupby("lvl")["bad"].agg(n="size", bad="sum")
        g["good"] = g["n"] - g["bad"]
        min_n = self.min_bin_pct * len(df)
        kept = g[g["n"] >= min_n].sort_values("bad", ascending=False)
        rare = g[g["n"] < min_n]

        labels = list(kept.index)
        good = list(kept["good"])
        bad = list(kept["bad"])
        if len(rare):
            labels.append(OTHER)
            good.append(rare["good"].sum())
            bad.append(rare["bad"].sum())

        woe, iv = woe_iv(good, bad)
        n = np.array(good) + np.array(bad)
        table = pd.DataFrame({
            "bin": labels, "n": n.astype(int), "n_bad": np.array(bad).astype(int),
            "bad_rate": np.array(bad) / np.maximum(n, 1), "woe": woe, "iv": iv,
        })
        mapping = {lvl: float(w) for lvl, w in zip(kept.index, woe[:len(kept)])}
        return FeatureBins(
            name=name, kind="categorical", table=table, mapping=mapping,
            other_woe=float(woe[-1]) if len(rare) else None,
        )

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        """Return a DataFrame of WoE values (same column names as the features)."""
        cols = columns if columns is not None else list(self.bins_)
        return pd.DataFrame({c: self.bins_[c].transform(X[c]) for c in cols}, index=X.index)

    # -------------------------------------------------------------- reports
    def iv_table(self) -> pd.DataFrame:
        rows = [{"feature": c, "kind": b.kind, "n_bins": len(b.table), "iv": b.iv}
                for c, b in self.bins_.items()]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    def bin_table(self, feature: str) -> pd.DataFrame:
        return self.bins_[feature].table.copy()


def iv_strength(iv: float) -> str:
    """Common rule-of-thumb labels for Information Value."""
    if iv < 0.02:
        return "useless"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious (check leakage)"
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/validation/discrimination.py"] = r'''"""Discrimination metrics: AUC, Gini, KS."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc_gini_ks(y_true, y_score) -> dict:
    """AUC, Gini (= 2*AUC - 1) and KS for a score where HIGHER = more likely default."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    if y.min() == y.max():
        raise ValueError("y_true contains a single class; discrimination is undefined")
    auc = float(roc_auc_score(y, s))
    fpr, tpr, _ = roc_curve(y, s)
    return {"auc": auc, "gini": 2 * auc - 1, "ks": float(np.max(tpr - fpr))}
'''

# ---------------------------------------------------------------------------
FILES["src/finrisk/models/scorecard.py"] = r'''"""WoE + logistic regression scorecard: the regulated-style baseline.

Steps (all fitted on TRAIN only):
  1. WoE binning of every candidate feature
  2. keep features with Information Value >= min_iv and limited missingness
  3. drop features highly correlated (in WoE space) with a stronger one
  4. fit logistic regression; drop features with a counter-intuitive (positive) sign
  5. convert to scorecard points (base score / base odds / points to double the odds)

Usage:
    python -m finrisk.models.scorecard
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from finrisk.features.build import get_feature_columns, load_features
from finrisk.features.woe import WoETransformer, iv_strength
from finrisk.validation.discrimination import auc_gini_ks

pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 200)


class Scorecard:
    def __init__(self, numeric: list[str], categorical: list[str],
                 min_iv: float = 0.02, max_corr: float = 0.8, max_missing: float = 0.6,
                 C: float = 1.0, min_bin_pct: float = 0.05, max_bins: int = 8,
                 base_score: float = 600.0, base_odds: float = 50.0, pdo: float = 20.0):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.min_iv, self.max_corr, self.max_missing = min_iv, max_corr, max_missing
        self.C = C
        self.min_bin_pct, self.max_bins = min_bin_pct, max_bins
        self.base_score, self.base_odds, self.pdo = base_score, base_odds, pdo
        self.features_: list[str] = []
        self.dropped_: dict[str, str] = {}

    # ---------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y) -> "Scorecard":
        y = np.asarray(y).astype(int)
        self.woe_ = WoETransformer(self.numeric, self.categorical,
                                   max_bins=self.max_bins,
                                   min_bin_pct=self.min_bin_pct).fit(X, y)
        iv = self.woe_.iv_table()
        self.iv_ = iv
        miss = X[list(self.woe_.bins_)].isna().mean()

        candidates = []
        for f, v in zip(iv["feature"], iv["iv"]):
            if v < self.min_iv:
                self.dropped_[f] = f"IV {v:.3f} < {self.min_iv}"
            elif miss[f] > self.max_missing:
                self.dropped_[f] = f"{miss[f]:.0%} missing"
            else:
                candidates.append(f)

        W = self.woe_.transform(X, candidates)

        # correlation pruning (candidates are already ordered by IV)
        corr = W.corr().abs()
        kept: list[str] = []
        for f in candidates:
            clash = next((k for k in kept if corr.loc[f, k] > self.max_corr), None)
            if clash:
                self.dropped_[f] = f"corr {corr.loc[f, clash]:.2f} with {clash}"
            else:
                kept.append(f)

        # sign check: with WoE = ln(good/bad) every coefficient must be negative
        feats = list(kept)
        while feats:
            lr = LogisticRegression(C=self.C, max_iter=1000).fit(W[feats], y)
            coefs = dict(zip(feats, lr.coef_[0]))
            wrong = [f for f, c in coefs.items() if c > 0]
            if not wrong:
                break
            worst = max(wrong, key=lambda f: coefs[f])
            self.dropped_[worst] = f"positive coefficient ({coefs[worst]:.2f})"
            feats.remove(worst)
        if not feats:
            raise RuntimeError("No feature survived selection. Check the data / thresholds.")

        self.features_ = feats
        self.model_ = lr
        self.coef_ = dict(zip(feats, lr.coef_[0]))
        self.factor_ = self.pdo / np.log(2)
        self.offset_ = self.base_score - self.factor_ * np.log(self.base_odds)
        return self

    # ------------------------------------------------------------ predict
    def _woe(self, X):
        return self.woe_.transform(X, self.features_)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of default (PD)."""
        return self.model_.predict_proba(self._woe(X))[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Scorecard points: higher = safer. score = offset - factor * logit(PD)."""
        logit = self.model_.decision_function(self._woe(X))
        return self.offset_ - self.factor_ * logit

    # ------------------------------------------------------------- report
    def scorecard_table(self) -> pd.DataFrame:
        """Points per bin. Total score = sum of points over features."""
        m = len(self.features_)
        base = (self.offset_ - self.factor_ * self.model_.intercept_[0]) / m
        rows = []
        for f in self.features_:
            t = self.woe_.bin_table(f)
            pts = -self.factor_ * self.coef_[f] * t["woe"] + base
            for b, n, br, w, p in zip(t["bin"], t["n"], t["bad_rate"], t["woe"], pts):
                rows.append({"feature": f, "bin": b, "n": n, "bad_rate": br,
                             "woe": w, "points": p})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def evaluate(model: Scorecard, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in splits.items():
        pd_hat = model.predict_proba(df)
        m = auc_gini_ks(df["is_default"], pd_hat)
        rows.append({"split": name, "n": len(df),
                     "observed_dr": df["is_default"].mean(),
                     "mean_pd": pd_hat.mean(), **m})
    return pd.DataFrame(rows)


def main() -> None:
    splits = {s: load_features(s) for s in ["train", "valid", "oot"]}
    print("rows: " + ", ".join(f"{k}={len(v):,}" for k, v in splits.items()))

    results = []
    for fs in ["independent", "with_lender"]:
        num, cat = get_feature_columns(fs)
        sc = Scorecard(num, cat).fit(splits["train"], splits["train"]["is_default"])

        print("\n" + "=" * 78)
        print(f"FEATURE SET: {fs}")
        print("=" * 78)
        iv = sc.iv_.head(12).copy()
        iv["strength"] = iv["iv"].map(iv_strength)
        print("\nTop features by Information Value (train):")
        print(iv.round(4).to_string(index=False))
        print(f"\nFeatures in the scorecard ({len(sc.features_)}): {', '.join(sc.features_)}")
        if sc.dropped_:
            print("Dropped:")
            for f, why in sc.dropped_.items():
                print(f"  - {f}: {why}")

        ev = evaluate(sc, splits)
        ev.insert(0, "feature_set", fs)
        results.append(ev)

    res = pd.concat(results, ignore_index=True)
    print("\n" + "=" * 78)
    print("PERFORMANCE (fit on train; valid = 2014; oot = 2015)")
    print("=" * 78)
    print(res.round(4).to_string(index=False))
    print("\nRead it: gini/ks should fall slightly from train to oot; "
          "mean_pd below observed_dr on oot means the model under-predicts risk.")


if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
FILES["tests/test_woe.py"] = r'''"""Unit tests for WoE binning and Information Value."""
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
'''

# ---------------------------------------------------------------------------
FILES["tests/test_discrimination.py"] = r'''"""Unit tests for AUC / Gini / KS."""
import numpy as np
import pytest

from finrisk.validation.discrimination import auc_gini_ks


def test_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = auc_gini_ks(y, s)
    assert m["auc"] == pytest.approx(1.0)
    assert m["gini"] == pytest.approx(1.0)
    assert m["ks"] == pytest.approx(1.0)


def test_random_score_has_gini_near_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20000)
    s = rng.uniform(size=20000)
    assert abs(auc_gini_ks(y, s)["gini"]) < 0.03


def test_inverted_score_has_negative_gini():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    assert auc_gini_ks(y, s)["gini"] == pytest.approx(-1.0)


def test_single_class_raises():
    with pytest.raises(ValueError):
        auc_gini_ks([1, 1, 1], [0.1, 0.2, 0.3])
'''


# ---------------------------------------------------------------------------
def write(rel: str, content: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not FORCE:
        if len(path.read_text(encoding="utf-8").splitlines()) > 3:
            print(f"  skip    {rel}  (already edited, use --force to overwrite)")
            return
    path.write_text(content, encoding="utf-8")
    print(f"  write   {rel}")


def patch(rel: str, old: str, new: str) -> None:
    path = ROOT / rel
    if not path.exists():
        print(f"  missing {rel}")
        return
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"  ok      {rel}  (already patched)")
    elif old in text:
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        print(f"  patch   {rel}")
    else:
        print(f"  warn    {rel}: expected text not found, skipping this patch")


def main() -> None:
    if not (ROOT / "src" / "finrisk" / "data" / "splits.py").exists():
        sys.exit("Run apply_phase1.py first, from the repo root.")

    print(f"Applying Phase 2a in {ROOT}\n")
    for rel, content in FILES.items():
        write(rel, content)

    # friendly 'is Docker running?' message instead of a long traceback
    imp_old = "from finrisk.db import execute_sql, get_engine"
    imp_new = "from finrisk.db import connect, execute_sql, get_engine"
    patch("src/finrisk/data/clean.py", imp_old, imp_new)
    patch("src/finrisk/data/clean.py",
          "    settings = get_settings()\n    engine = get_engine()\n",
          "    settings = get_settings()\n    connect().close()  # friendly error if Docker is down\n    engine = get_engine()\n")
    patch("src/finrisk/data/splits.py", imp_old, imp_new)
    patch("src/finrisk/data/splits.py",
          "    assign_splits()\n\n    df = summarize()",
          "    connect().close()  # friendly error if Docker is down\n    assign_splits()\n\n    df = summarize()")

    print("\nDone. Next:")
    print("  pytest -q")
    print("  python -m finrisk.features.build")
    print("  python -m finrisk.models.scorecard")


if __name__ == "__main__":
    main()