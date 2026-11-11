"""WoE + logistic regression scorecard: the regulated-style baseline.

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
