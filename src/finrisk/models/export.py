"""Portable scorecard: the fitted scorecard as plain JSON, scored with numpy/pandas only.

Why: the public demo must run without a database, MLflow, scikit-learn or any model server.
The JSON holds the bins, WoE values, coefficients and scorecard points. A test proves that
it reproduces the original Scorecard exactly.

Usage:
    python -m finrisk.models.export        # champion from the registry -> reports/champion_scorecard.json
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from finrisk.validation.results import clean

MISSING = "__MISSING__"   # same marker as features/woe.py


def _apply_bins(b: dict, x: pd.Series) -> np.ndarray:
    if b["kind"] == "numeric":
        v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        idx = np.searchsorted(np.asarray(b["edges"], dtype=float), v, side="left")
        out = np.asarray(b["woe"], dtype=float)[idx]
        return np.where(np.isnan(v), b["missing_woe"], out)
    s = x.astype(object).where(x.notna(), MISSING).astype(str)
    fill = b["other_woe"] if b["other_woe"] is not None else 0.0
    return s.map(b["mapping"]).fillna(fill).to_numpy(dtype=float)


class PortableScorecard:
    def __init__(self, spec: dict):
        self.spec = spec
        self.features: list[str] = list(spec["features"])
        self.numeric: list[str] = list(spec["numeric"])
        self.categorical: list[str] = list(spec["categorical"])
        self.coef: dict = spec["coef"]
        self.intercept: float = spec["intercept"]
        self.factor: float = spec["factor"]
        self.offset: float = spec["offset"]
        self.mean_woe: dict = spec["mean_woe"]

    # ---------------------------------------------------------------- build / io
    @classmethod
    def from_scorecard(cls, sc, background: pd.DataFrame, model_version: str = "",
                       feature_set: str = "", n_train: int | None = None,
                       train_default_rate: float | None = None) -> "PortableScorecard":
        bins = {}
        for f in sc.features_:
            b = sc.woe_.bins_[f]
            bins[f] = {
                "kind": b.kind,
                "edges": b.edges.tolist() if b.kind == "numeric" else None,
                "woe": b.woe_values.tolist() if b.kind == "numeric" else None,
                "missing_woe": float(b.missing_woe),
                "mapping": b.mapping if b.kind == "categorical" else None,
                "other_woe": b.other_woe,
                "table": b.table[["bin", "n", "bad_rate", "woe"]],
            }
        mean_woe = sc.woe_.transform(background, sc.features_).mean().to_dict()
        spec = {
            "format": 1, "kind": "scorecard", "feature_set": feature_set,
            "model_version": str(model_version),
            "created": datetime.now().isoformat(timespec="seconds"),
            "features": list(sc.features_),
            "numeric": [f for f in sc.features_ if bins[f]["kind"] == "numeric"],
            "categorical": [f for f in sc.features_ if bins[f]["kind"] == "categorical"],
            "intercept": float(sc.model_.intercept_[0]),
            "coef": {f: float(c) for f, c in sc.coef_.items()},
            "factor": float(sc.factor_), "offset": float(sc.offset_),
            "bins": bins, "mean_woe": {f: float(v) for f, v in mean_woe.items()},
            "points_table": sc.scorecard_table(),
            "training": {"n_loans": n_train, "default_rate": train_default_rate},
        }
        return cls(clean(spec))

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.spec, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "PortableScorecard":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # ----------------------------------------------------------------- scoring
    def woe_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f: _apply_bins(self.spec["bins"][f], X[f]) for f in self.features},
                            index=X.index)

    def logit(self, X: pd.DataFrame) -> np.ndarray:
        W = self.woe_frame(X)
        return self.intercept + sum(self.coef[f] * W[f].to_numpy() for f in self.features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of default."""
        return 1.0 / (1.0 + np.exp(-self.logit(X)))

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Scorecard points (higher = safer)."""
        return self.offset - self.factor * self.logit(X)

    @property
    def base_value(self) -> float:
        return self.intercept + sum(self.coef[f] * self.mean_woe[f] for f in self.features)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contribution to the log-odds of default vs the average applicant."""
        W = self.woe_frame(X)
        return pd.DataFrame({f: self.coef[f] * (W[f] - self.mean_woe[f]) for f in self.features},
                            index=X.index)


def applicant_frame(inputs: dict, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """One-row frame for the given feature columns from raw form inputs (derived ones computed)."""
    d = dict(inputs)
    if d.get("fico_score") is not None:
        d["fico_mid"] = d["fico_score"]
    loan, inc = d.get("loan_amnt"), d.get("annual_inc")
    if loan and inc and inc > 0:
        d["loan_to_income"] = loan / inc
    rb = d.get("revol_bal")
    if rb is not None and not (isinstance(rb, float) and np.isnan(rb)) and inc and inc > 0:
        d["revol_bal_to_income"] = rb / inc
    row = {}
    for c in numeric:
        v = d.get(c)
        row[c] = np.nan if v is None else float(v)
    for c in categorical:
        row[c] = d.get(c)
    return pd.DataFrame([row])


def main() -> None:
    from finrisk.config import get_settings
    from finrisk.features.build import load_features
    from finrisk.models import registry
    from finrisk.models.scorecard import Scorecard

    fs = "independent"
    client = registry.setup_mlflow()
    sc = registry.load_model_object("champion", fs)
    if not isinstance(sc, Scorecard):
        raise SystemExit("The champion is not a scorecard; only scorecards can be exported.")
    version = client.get_model_version_by_alias(registry.registered_name(fs), "champion").version

    train = load_features("train")
    ps = PortableScorecard.from_scorecard(
        sc, train.sample(5000, random_state=0), version, fs,
        n_train=len(train), train_default_rate=float(train["is_default"].mean()))

    check = load_features("oot").sample(2000, random_state=1)
    diff = float(np.max(np.abs(ps.predict_proba(check) - sc.predict_proba(check))))
    if diff > 1e-9:
        raise SystemExit(f"Export does not reproduce the model (max PD difference {diff:.2e}).")

    path = get_settings().reports_dir / "champion_scorecard.json"
    ps.save(path)
    print(f"Wrote {path}  ({len(ps.features)} features: {', '.join(ps.features)})")
    print("Verified: identical to the registry scorecard on 2,000 OOT loans (max diff "
          f"{diff:.1e}).")


if __name__ == "__main__":
    main()
