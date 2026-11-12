"""Out-of-time performance: does discrimination survive the move to a later period?

  - Gini / KS / AUC / Brier per split
  - relative Gini loss train -> oot and valid -> oot
  - Gini floor (minimum acceptable discriminatory power)
  - Gini by calendar quarter across 2014-2015 (stability inside the test window)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from finrisk.validation.discrimination import auc_gini_ks
from finrisk.validation.results import THRESHOLDS, grade, make


def performance_by_period(dates, y, p, freq: str = "Q", min_n: int = 500) -> pd.DataFrame:
    period = pd.to_datetime(pd.Series(np.asarray(dates))).dt.to_period(freq)
    df = pd.DataFrame({"period": period.to_numpy(), "y": np.asarray(y).astype(int),
                       "p": np.asarray(p, dtype=float)})
    rows = []
    for per, g in df.groupby("period"):
        if len(g) < min_n or g["y"].nunique() < 2:
            continue
        rows.append({"period": str(per), "n": len(g), "observed_dr": g["y"].mean(),
                     "mean_pd": g["p"].mean(), "gini": auc_gini_ks(g["y"], g["p"])["gini"]})
    return pd.DataFrame(rows)


def run(ctx) -> list:
    out, m = [], ctx.model_name
    met = {}
    for s in ["train", "valid", "oot"]:
        d = auc_gini_ks(ctx.y(s), ctx.pds[s])
        d["brier"] = float(brier_score_loss(ctx.y(s), np.clip(ctx.pds[s], 1e-6, 1 - 1e-6)))
        d["n"] = int(len(ctx.y(s)))
        met[s] = d
        out.append(make(m, "out_of_time", f"{s}_gini", d["gini"], "info", details=d))

    w, f = THRESHOLDS["gini_drop_relative"]
    for a, b in [("train", "oot"), ("valid", "oot")]:
        drop = (met[a]["gini"] - met[b]["gini"]) / met[a]["gini"]
        out.append(make(m, "out_of_time", f"gini_drop_{a}_to_{b}", drop, grade(drop, w, f), w, f,
                        f"relative Gini loss <= {w:.0%} pass, <= {f:.0%} warn, else fail",
                        {f"gini_{a}": met[a]["gini"], f"gini_{b}": met[b]["gini"]}))

    wf, ff = THRESHOLDS["gini_floor"]
    out.append(make(m, "out_of_time", "oot_gini_floor", met["oot"]["gini"],
                    grade(met["oot"]["gini"], wf, ff, higher_is_worse=False), wf, ff,
                    f"OOT Gini >= {wf} pass, >= {ff} warn, else fail"))

    both = pd.concat([ctx.splits["valid"], ctx.splits["oot"]], ignore_index=True)
    p_both = np.concatenate([ctx.pds["valid"], ctx.pds["oot"]])
    per = performance_by_period(both["issue_d"], both["is_default"], p_both)
    if len(per):
        worst = float(per["gini"].min())
        out.append(make(m, "out_of_time", "worst_quarter_gini", worst,
                        grade(worst, wf, ff, higher_is_worse=False), wf, ff,
                        "lowest quarterly Gini over 2014-2015, same floor as above",
                        {"by_quarter": per}))
    return out
