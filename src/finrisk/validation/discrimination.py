"""Discrimination metrics: AUC, Gini, KS."""
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
