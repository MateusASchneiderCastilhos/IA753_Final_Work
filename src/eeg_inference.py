"""
eeg_inference.py
───────────────────────────────────────────────────────────────────────────────
Inferential layer for the descriptive backbone built by eeg_analysis.py.

Consumes results/backbone/scalars.pkl (the tidy per-epoch band-scalar DataFrame)
and runs paired non-parametric tests across SUBJECT-MEANS:

  1. movement_vs_rest        — each movement class vs rest (paired across subjects)
  2. movement_vs_rest_pooled — pooled movement (4 classes) vs rest
  3. ME_vs_MI                — execution vs imagery, same class (paired across subjects)

Each test:
  • Wilcoxon signed-rank (paired, two-sided) on subject-means
  • matched-pairs rank-biserial correlation as effect size
  • median(group_a), median(group_b), median paired difference

Multiple comparisons are controlled with Benjamini-Hochberg FDR, with the family
defined **per (comparison_type, metric)** — i.e. all class × channel × band tests
of one metric within one comparison type are corrected together.

Aggregation rule (inherited from eeg_analysis)
──────────────────────────────────────────────
Collapse epochs → one mean per subject FIRST, then test across the subject-means.
The pooled "movement" value is the subject mean over all movement-class epochs
(pooled at the epoch level, matching the curve figures).

Statistics are reported but NOT interpreted here; figures stay in eeg_analysis.
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from eeg_analysis import (
    ANALYSIS_CHANNELS, ANALYSIS_CLASSES, MOVEMENT_CLASSES, POOLED_LABEL,
    METRIC_COLUMNS, SCALAR_METRIC_ORDER, CONDITIONS, _BANDS, _log,
)

# Classes carried through the ME-vs-MI comparison (5 analysis classes + pooled).
MEMI_CLASSES: List[str] = ANALYSIS_CLASSES + [POOLED_LABEL]
_COL_TO_METRIC: Dict[str, str] = {v: k for k, v in METRIC_COLUMNS.items()}
_VALUE_COLS: List[str] = list(dict.fromkeys(METRIC_COLUMNS.values()))

# Result-table column order
_RESULT_COLS: List[str] = [
    "comparison_type", "condition", "metric", "class", "channel", "band",
    "group_a", "group_b", "n",
    "median_a", "median_b", "median_diff",
    "statistic", "effect_rrb", "p_raw", "p_fdr", "significant",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Subject-level table
# ═══════════════════════════════════════════════════════════════════════════════

def load_scalars(backbone_dir: Path) -> pd.DataFrame:
    """Load the tidy per-epoch scalar DataFrame persisted by eeg_analysis."""
    return pd.read_pickle(backbone_dir / "scalars.pkl")


def build_subject_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the per-epoch DataFrame to one value per subject, in long form.

    Produces, per (subject, condition, channel, band, metric), the subject mean
    for each of the 5 analysis classes plus a pooled 'movement' class (mean over
    all movement-class epochs). Restricted to the analysis channel subset.

    Returns long-form columns:
        subject, condition, class, channel, band, metric, value
    """
    chan = df["channel"].isin(ANALYSIS_CHANNELS)

    # Per-class subject means (5 analysis classes)
    base = df[df["class"].isin(ANALYSIS_CLASSES) & chan]
    per_class = (base.groupby(["subject", "condition", "class", "channel", "band"],
                              observed=True)[_VALUE_COLS]
                     .mean().reset_index())

    # Pooled movement subject means (epoch-level pooling over the 4 movement classes)
    mv = df[df["class"].isin(MOVEMENT_CLASSES) & chan]
    pooled = (mv.groupby(["subject", "condition", "channel", "band"],
                         observed=True)[_VALUE_COLS]
                .mean().reset_index())
    pooled["class"] = POOLED_LABEL

    subj = pd.concat([per_class, pooled], ignore_index=True)
    subj["class"] = subj["class"].astype(str)
    subj[["subject", "condition", "channel", "band"]] = \
        subj[["subject", "condition", "channel", "band"]].astype(str)

    long = subj.melt(
        id_vars=["subject", "condition", "class", "channel", "band"],
        value_vars=_VALUE_COLS, var_name="_col", value_name="value",
    )
    long["metric"] = long["_col"].map(_COL_TO_METRIC)
    long = long.drop(columns="_col")
    return long


def _series(long: pd.DataFrame, condition: str, metric: str, channel: str,
            band: str, klass: str) -> pd.Series:
    """Subject-indexed value Series for one cell of the design."""
    m = ((long["condition"] == condition) & (long["metric"] == metric) &
         (long["channel"] == channel) & (long["band"] == band) &
         (long["class"] == klass))
    sub = long[m]
    return pd.Series(sub["value"].to_numpy(), index=sub["subject"].to_numpy())


# ═══════════════════════════════════════════════════════════════════════════════
# Paired test + effect size
# ═══════════════════════════════════════════════════════════════════════════════

def _rank_biserial(diff: np.ndarray) -> float:
    """
    Matched-pairs rank-biserial correlation in [-1, 1].

    r = (W+ − W−) / (W+ + W−), where W± are the summed ranks of the positive /
    negative absolute differences (zeros dropped). Positive → group_a > group_b.
    """
    nz = diff[diff != 0]
    if nz.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    w_plus = ranks[nz > 0].sum()
    w_minus = ranks[nz < 0].sum()
    total = w_plus + w_minus
    return float((w_plus - w_minus) / total) if total > 0 else 0.0


def paired_wilcoxon(a: pd.Series, b: pd.Series) -> Dict:
    """
    Paired Wilcoxon signed-rank test of a vs b over their common subjects.

    Returns a dict with n, statistic, p_raw, effect_rrb, median_a, median_b,
    median_diff. Degenerate cases (n < 1, all-zero differences) yield NaN stat/p
    but still report the medians and effect size where defined.
    """
    common = a.index.intersection(b.index)
    a_v = a.loc[common].to_numpy(dtype=float)
    b_v = b.loc[common].to_numpy(dtype=float)
    valid = ~(np.isnan(a_v) | np.isnan(b_v))
    a_v, b_v = a_v[valid], b_v[valid]
    n = a_v.size
    diff = a_v - b_v

    out = {
        "n": n,
        "median_a": float(np.median(a_v)) if n else np.nan,
        "median_b": float(np.median(b_v)) if n else np.nan,
        "median_diff": float(np.median(diff)) if n else np.nan,
        "effect_rrb": _rank_biserial(diff) if n else np.nan,
        "statistic": np.nan,
        "p_raw": np.nan,
    }
    if n >= 1 and np.any(diff != 0):
        try:
            res = stats.wilcoxon(a_v, b_v, zero_method="wilcox",
                                 alternative="two-sided")
            out["statistic"] = float(res.statistic)
            out["p_raw"] = float(res.pvalue)
        except ValueError:
            pass   # leaves NaN (e.g. too few non-zero diffs)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Benjamini-Hochberg FDR
# ═══════════════════════════════════════════════════════════════════════════════

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values; NaNs pass through and are excluded."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    mask = ~np.isnan(p)
    pv = p[mask]
    m = pv.size
    if m == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]   # enforce monotonicity
    adj = np.clip(adj, 0.0, 1.0)
    res = np.empty(m)
    res[order] = adj
    out[mask] = res
    return out


def apply_fdr(results: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Add p_fdr and significant columns, correcting within (comparison_type, metric)."""
    results = results.copy()
    results["p_fdr"] = np.nan
    for _, idx in results.groupby(["comparison_type", "metric"], observed=True).groups.items():
        rows = list(idx)
        results.loc[rows, "p_fdr"] = _bh_fdr(results.loc[rows, "p_raw"].to_numpy())
    results["significant"] = results["p_fdr"] < alpha
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Test loops
# ═══════════════════════════════════════════════════════════════════════════════

def _record(comparison_type: str, condition: str, metric: str, klass: str,
            channel: str, band: str, group_a: str, group_b: str,
            res: Dict) -> Dict:
    row = {
        "comparison_type": comparison_type, "condition": condition,
        "metric": metric, "class": klass, "channel": channel, "band": band,
        "group_a": group_a, "group_b": group_b,
    }
    row.update(res)
    return row


def run_movement_vs_rest(long: pd.DataFrame) -> List[Dict]:
    """Per-class and pooled movement vs rest, paired across subjects, within condition."""
    rows: List[Dict] = []
    conditions = sorted(long["condition"].unique())
    for cond in conditions:
        for metric in SCALAR_METRIC_ORDER:
            for ch in ANALYSIS_CHANNELS:
                for band in _BANDS:
                    rest = _series(long, cond, metric, ch, band, "rest")
                    if rest.empty:
                        continue
                    # per movement class
                    for mcls in MOVEMENT_CLASSES:
                        a = _series(long, cond, metric, ch, band, mcls)
                        if a.empty:
                            continue
                        res = paired_wilcoxon(a, rest)
                        rows.append(_record("movement_vs_rest", cond, metric,
                                            mcls, ch, band, mcls, "rest", res))
                    # pooled movement
                    a = _series(long, cond, metric, ch, band, POOLED_LABEL)
                    if not a.empty:
                        res = paired_wilcoxon(a, rest)
                        rows.append(_record("movement_vs_rest_pooled", cond, metric,
                                            POOLED_LABEL, ch, band, POOLED_LABEL,
                                            "rest", res))
    return rows


def run_me_vs_mi(long: pd.DataFrame) -> List[Dict]:
    """ME vs MI for each class, paired across subjects present in both conditions."""
    rows: List[Dict] = []
    if not {"ME", "MI"}.issubset(set(long["condition"].unique())):
        _log("[infer]   ME_vs_MI skipped — both conditions not present in backbone")
        return rows
    for metric in SCALAR_METRIC_ORDER:
        for klass in MEMI_CLASSES:
            for ch in ANALYSIS_CHANNELS:
                for band in _BANDS:
                    me = _series(long, "ME", metric, ch, band, klass)
                    mi = _series(long, "MI", metric, ch, band, klass)
                    if me.empty or mi.empty:
                        continue
                    res = paired_wilcoxon(me, mi)
                    if res["n"] == 0:
                        continue
                    rows.append(_record("ME_vs_MI", "ME_vs_MI", metric,
                                        klass, ch, band, "ME", "MI", res))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference(
    results_root: Optional[str] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run all paired tests, apply FDR, and persist the results tables.

    Reads results/backbone/scalars.pkl, writes:
        results/tables/inferential_movement_vs_rest.csv
        results/tables/inferential_ME_vs_MI.csv
        results/tables/inferential_all.csv

    Returns the combined results DataFrame (with p_fdr and significant columns).
    """
    project_root = Path(__file__).resolve().parent.parent
    res_root = Path(results_root) if results_root else project_root / "results"
    backbone_dir = res_root / "backbone"
    tbl_root = res_root / "tables"
    tbl_root.mkdir(parents=True, exist_ok=True)

    df = load_scalars(backbone_dir)
    long = build_subject_level(df)
    _log(f"[infer]   subject-level table: {len(long):,} rows | "
         f"conditions={sorted(long['condition'].unique())}")

    rows = run_movement_vs_rest(long) + run_me_vs_mi(long)
    results = pd.DataFrame(rows)
    if results.empty:
        _log("[infer]   no tests produced — check backbone contents")
        return results

    results = apply_fdr(results, alpha=alpha)
    results = results[_RESULT_COLS].sort_values(
        ["comparison_type", "metric", "band", "channel", "class"]
    ).reset_index(drop=True)

    mvr = results[results["comparison_type"].str.startswith("movement_vs_rest")]
    memi = results[results["comparison_type"] == "ME_vs_MI"]
    mvr.to_csv(tbl_root / "inferential_movement_vs_rest.csv", index=False)
    if not memi.empty:
        memi.to_csv(tbl_root / "inferential_ME_vs_MI.csv", index=False)
    results.to_csv(tbl_root / "inferential_all.csv", index=False)

    n_sig = int(results["significant"].sum())
    _log(f"[infer]   {len(results)} tests | {n_sig} significant at FDR<{alpha} | "
         f"saved to {tbl_root}")
    if n_sig:
        cols = ["comparison_type", "condition", "metric", "class", "channel",
                "band", "median_diff", "p_raw", "p_fdr"]
        _log("[infer]   significant findings:\n" +
             results.loc[results["significant"], cols].to_string(index=False))
    return results


if __name__ == "__main__":
    run_inference()
