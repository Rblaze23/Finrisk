"""Gradient boosting challengers: XGBoost and LightGBM behind one interface.

Design choices (documented for the validation report):
  - Categorical variables are encoded as ordinal codes ranked by TRAIN default rate
    (unseen / rare levels -> NaN). Simple, leakage-free, and SHAP-friendly.
  - Monotonic constraints on variables with a clear economic direction
    (e.g. higher FICO never increases PD). Improves stability and explainability.
  - Early stopping on the validation split (2014). The out-of-time split is never used.
  - No class re-weighting, so predicted probabilities stay on the real PD scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# +1: higher value -> higher PD,  -1: higher value -> lower PD
MONOTONE = {
    "fico_mid": -1, "annual_inc": -1, "credit_history_years": -1,
    "mort_acc": -1, "total_rev_hi_lim": -1,
    "dti": 1, "revol_util": 1, "bc_util": 1, "inq_last_6mths": 1,
    "delinq_2yrs": 1, "pub_rec": 1, "loan_to_income": 1,
    "revol_bal_to_income": 1,
    "int_rate": 1, "grade": 1, "sub_grade": 1,   # grade codes are ordered by default rate
}

DEFAULT_PARAMS = {
    "xgboost": dict(n_estimators=1500, learning_rate=0.05, max_depth=4,
                    min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=10.0, tree_method="hist"),
    "lightgbm": dict(n_estimators=1500, learning_rate=0.05, num_leaves=15,
                     min_child_samples=200, subsample=0.8, subsample_freq=1,
                     colsample_bytree=0.8, reg_lambda=10.0),
}
EARLY_STOPPING_ROUNDS = 50


class TargetOrderedEncoder:
    """Category -> rank of its train default rate (0 = safest). Unseen / rare -> NaN."""

    def __init__(self, columns: list[str], min_count: int = 100):
        self.columns = list(columns)
        self.min_count = min_count
        self.maps_: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y) -> "TargetOrderedEncoder":
        y = pd.Series(np.asarray(y).astype(float), index=X.index)
        for c in self.columns:
            g = y.groupby(X[c]).agg(["mean", "size"])
            g = g[g["size"] >= self.min_count].sort_values("mean")
            self.maps_[c] = {k: float(i) for i, k in enumerate(g.index)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in self.columns:
            out[c] = X[c].map(self.maps_[c]).astype(float)
        return out


class GBMModel:
    def __init__(self, kind: str, numeric: list[str], categorical: list[str],
                 monotone: bool = True, params: dict | None = None, seed: int = 42):
        if kind not in DEFAULT_PARAMS:
            raise ValueError(f"kind must be one of {list(DEFAULT_PARAMS)}")
        self.kind = kind
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.feature_names = self.numeric + self.categorical
        self.monotone = monotone
        self.params = {**DEFAULT_PARAMS[kind], **(params or {})}
        self.seed = seed

    @property
    def constraints(self) -> list[int]:
        if not self.monotone:
            return [0] * len(self.feature_names)
        return [MONOTONE.get(f, 0) for f in self.feature_names]

    def prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Raw feature frame -> numeric matrix the booster sees (also used by SHAP)."""
        enc = self.encoder_.transform(X[self.feature_names])
        return enc.astype(float)

    def fit(self, X, y, X_valid=None, y_valid=None) -> "GBMModel":
        y = np.asarray(y).astype(int)
        self.encoder_ = TargetOrderedEncoder(self.categorical).fit(X[self.feature_names], y)
        Xe = self.prepare(X)
        has_valid = X_valid is not None and y_valid is not None
        if has_valid:
            Xv, yv = self.prepare(X_valid), np.asarray(y_valid).astype(int)

        if self.kind == "xgboost":
            from xgboost import XGBClassifier
            self.model_ = XGBClassifier(
                **self.params, monotone_constraints=tuple(self.constraints),
                eval_metric="auc", random_state=self.seed, n_jobs=-1,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS if has_valid else None,
            )
            self.model_.fit(Xe, y, eval_set=[(Xv, yv)] if has_valid else None, verbose=False)
            self.n_trees_ = int(getattr(self.model_, "best_iteration", self.params["n_estimators"] - 1)) + 1
        else:
            import lightgbm as lgb
            self.model_ = lgb.LGBMClassifier(
                **self.params, monotone_constraints=self.constraints,
                random_state=self.seed, n_jobs=-1, verbose=-1,
            )
            kwargs = {}
            if has_valid:
                kwargs = dict(eval_set=[(Xv, yv)], eval_metric="auc",
                              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
            self.model_.fit(Xe, y, **kwargs)
            self.n_trees_ = int(self.model_.best_iteration_ or self.params["n_estimators"])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of default."""
        return self.model_.predict_proba(self.prepare(X))[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        if self.kind == "xgboost":
            gain = self.model_.get_booster().get_score(importance_type="gain")
            vals = [gain.get(f, 0.0) for f in self.feature_names]
        else:
            vals = list(self.model_.booster_.feature_importance(importance_type="gain"))
        df = pd.DataFrame({"feature": self.feature_names, "gain": vals})
        return df.sort_values("gain", ascending=False).reset_index(drop=True)
