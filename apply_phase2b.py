#!/usr/bin/env python3
"""
Small in-place fix for Phase 2b (run from the repo root):


1. XGBoost explanations: use XGBoost's built-in TreeSHAP so contributions are exactly additive
   (the shap library reports a slightly wrong base value with xgboost 3.x).
2. Reason codes: ignore negligible contributions, and say so when there are none.

Safe to run twice. Does not touch your data or MLflow.
"""
from pathlib import Path

ROOT = Path(".").resolve()


def patch(rel: str, old: str, new: str) -> None:
    path = ROOT / rel
    if not path.exists():
        print(f"  missing {rel}")
        return
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"  ok      {rel}  (already fixed)")
    elif old in text:
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        print(f"  patch   {rel}")
    else:
        print(f"  WARN    {rel}: expected text not found. Send me the file.")


SHAP = "src/finrisk/explain/shap_explainer.py"

patch(
    SHAP,
    '''        if isinstance(model, GBMModel):
            import shap
            self._kind = "tree"
            self._explainer = shap.TreeExplainer(model.model_)
            self.base_value = float(np.ravel(self._explainer.expected_value)[-1])
        elif''',
    '''        if isinstance(model, GBMModel) and model.kind == "xgboost":
            # XGBoost's built-in TreeSHAP (pred_contribs): the same Shapley values, exactly
            # additive, and immune to base-value mismatches between shap and xgboost versions.
            import xgboost as xgb
            self._kind = "xgb"
            self._xgb = xgb
            self._booster = model.model_.get_booster()
            probe = pd.DataFrame([[np.nan] * len(model.feature_names)], columns=model.feature_names)
            self.base_value = float(self._booster.predict(xgb.DMatrix(probe), pred_contribs=True)[0, -1])
        elif isinstance(model, GBMModel):
            import shap
            self._kind = "tree"
            self._explainer = shap.TreeExplainer(model.model_)
            self.base_value = float(np.ravel(self._explainer.expected_value)[-1])
        elif''',
)

patch(
    SHAP,
    '''        Xe = self.model.prepare(X)
        sv = self._explainer.shap_values(Xe, check_additivity=False)''',
    '''        Xe = self.model.prepare(X)
        if self._kind == "xgb":
            raw = self._booster.predict(self._xgb.DMatrix(Xe), pred_contribs=True)
            return pd.DataFrame(raw[:, :-1], index=X.index, columns=self.model.feature_names)
        sv = self._explainer.shap_values(Xe, check_additivity=False)''',
)

patch(
    SHAP,
    '''            for line in reasons_text(reasons):
                print("     " + line)''',
    '''            for line in reasons_text(reasons) or ["(no material risk-increasing factors)"]:
                print("     " + line)''',
)

patch(
    "src/finrisk/explain/reason_codes.py",
    '''def top_reasons(contributions: pd.Series, values: pd.Series, k: int = 5) -> list[dict]:
    """Top-k features that INCREASE the applicant's risk (positive log-odds contribution)."""
    pos = contributions[contributions > 0].sort_values(ascending=False).head(k)''',
    '''def top_reasons(contributions: pd.Series, values: pd.Series, k: int = 5,
                min_impact: float = 0.01) -> list[dict]:
    """Top-k features that INCREASE the applicant's risk (log-odds contribution > min_impact)."""
    pos = contributions[contributions > min_impact].sort_values(ascending=False).head(k)''',
)

print("\nNow run: pytest -q   (expect 19 passed)")
