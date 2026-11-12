"""Master scale: fixed PD cut-offs mapped to risk bands R1 (safest) .. R6 (riskiest)."""
from __future__ import annotations

import numpy as np
import pandas as pd

BAND_EDGES = [0.05, 0.08, 0.12, 0.18, 0.25]
BAND_LABELS = ["R1", "R2", "R3", "R4", "R5", "R6"]


def assign_risk_band(pd_values) -> np.ndarray:
    """Map probabilities of default to risk bands (lower edge inclusive)."""
    idx = np.digitize(np.asarray(pd_values, dtype=float), BAND_EDGES)
    return np.array(BAND_LABELS)[idx]


def band_table() -> pd.DataFrame:
    lows = [0.0] + BAND_EDGES
    highs = BAND_EDGES + [1.0]
    return pd.DataFrame({"band": BAND_LABELS, "pd_from": lows, "pd_to": highs})
