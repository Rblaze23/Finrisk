"""Binning and Weight of Evidence (WoE) / Information Value (IV).

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
