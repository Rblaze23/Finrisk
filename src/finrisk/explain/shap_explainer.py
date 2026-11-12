"""Local and global explanations for any PD model, in log-odds of default.

    contributions(X)  ->  DataFrame (one column per feature)
    base_value + sum(contributions) = logit(PD)        (positive = increases risk)

  GBMModel  : SHAP TreeExplainer
  Scorecard : exact additive contributions  coef_j * (WoE_j - mean WoE_j)

Demo (needs trained models):  python -m finrisk.explain.shap_explainer
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finrisk.explain.reason_codes import reasons_text, top_reasons
from finrisk.models.gbm import GBMModel
from finrisk.models.risk_bands import assign_risk_band
from finrisk.models.scorecard import Scorecard

pd.set_option("display.width", 160)


class PDExplainer:
    def __init__(self, model, background: pd.DataFrame | None = None):
        self.model = model
        if isinstance(model, GBMModel) and model.kind == "xgboost":
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
        elif isinstance(model, Scorecard):
            if background is None:
                raise ValueError("Scorecard explanations need a background sample (e.g. train)")
            self._kind = "scorecard"
            W = model.woe_.transform(background, model.features_)
            self._means = W.mean()
            self.base_value = float(model.model_.intercept_[0]
                                    + sum(model.coef_[f] * self._means[f] for f in model.features_))
        else:
            raise TypeError(f"Unsupported model type: {type(model).__name__}")

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._kind == "scorecard":
            W = self.model.woe_.transform(X, self.model.features_)
            data = {f: self.model.coef_[f] * (W[f] - self._means[f]) for f in self.model.features_}
            return pd.DataFrame(data, index=X.index)

        Xe = self.model.prepare(X)
        if self._kind == "xgb":
            raw = self._booster.predict(self._xgb.DMatrix(Xe), pred_contribs=True)
            return pd.DataFrame(raw[:, :-1], index=X.index, columns=self.model.feature_names)
        sv = self._explainer.shap_values(Xe, check_additivity=False)
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        return pd.DataFrame(sv, index=X.index, columns=self.model.feature_names)

    def global_importance(self, X: pd.DataFrame) -> pd.Series:
        """Mean absolute contribution per feature."""
        return self.contributions(X).abs().mean().sort_values(ascending=False)


def main() -> None:
    from finrisk.features.build import load_features
    from finrisk.models.registry import load_model_object

    fs = "independent"
    champion = load_model_object("champion", fs)
    challenger = load_model_object("challenger", fs)
    train = load_features("train").sample(5000, random_state=0)
    oot = load_features("oot").sample(20000, random_state=0).reset_index(drop=True)

    ex = {"champion": PDExplainer(champion, train), "challenger": PDExplainer(challenger, train)}
    pds = {k: m.predict_proba(oot) for k, m in [("champion", champion), ("challenger", challenger)]}

    order = np.argsort(pds["champion"])
    picks = {"HIGHEST-risk applicant": order[-1],
             "MEDIAN applicant": order[len(order) // 2],
             "LOWEST-risk applicant": order[0]}

    print(f"champion = {type(champion).__name__}, challenger = {type(challenger).__name__}")
    for title, i in picks.items():
        row = oot.iloc[[i]]
        print("\n" + "=" * 78)
        print(f"{title}  (loan_id {int(row['loan_id'].iloc[0])}, "
              f"actual outcome: {'DEFAULT' if row['is_default'].iloc[0] else 'repaid'})")
        for name in ["champion", "challenger"]:
            p = float(pds[name][i])
            c = ex[name].contributions(row).iloc[0]
            reasons = top_reasons(c, row.iloc[0], k=4)
            print(f"\n  {name}: PD = {p:.1%}  band = {assign_risk_band([p])[0]}")
            for line in reasons_text(reasons) or ["(no material risk-increasing factors)"]:
                print("     " + line)

    print("\n" + "=" * 78)
    print("GLOBAL IMPORTANCE (mean |contribution|, log-odds) on 20,000 OOT loans")
    imp = pd.DataFrame({k: ex[k].global_importance(oot) for k in ex}).fillna(0)
    print(imp.sort_values("challenger", ascending=False).head(12).round(4).to_string())


if __name__ == "__main__":
    main()