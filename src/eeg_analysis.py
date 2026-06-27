"""
eeg_analysis.py
───────────────────────────────────────────────────────────────────────────────
Descriptive analysis layer for the preprocessed EEG spectral data.

Consumes the per-run pickles produced by eeg_pipeline.process_single_recording()
(see data_processed/{motor_execution,motor_imagery}/Sxx_{ME,MI}/processed_*.pkl),
pools the ~10 runs per subject, builds a reusable data backbone, and generates
the curve and scalar figures.

Pipeline
────────
A  discover_pickles()        Locate + group per-run pickles by subject
B  load_subject()            Concatenate a subject's runs along the epoch axis
C  build_scalar_dataframe()  Tidy long-form DataFrame of per-epoch band scalars
D  build_curve_arrays()      Subject-mean spectral curves + within-subject 95% CI
E  save_backbone/load_backbone   Persist/reload so the heavy step runs once
F  plot_* / summary_tables() Figures + descriptive tables
G  main_analysis()           Orchestrator

Aggregation rule (applies everywhere)
─────────────────────────────────────
Collapse to ONE mean per subject first, then compute group statistics across the
subject-means. Per-subject CI = variability across that subject's epochs; group
CI = variability across subject-means (never across pooled epochs → avoids
pseudoreplication).

Inferential statistics (Wilcoxon, FDR) are intentionally OUT OF SCOPE here.
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from eeg_pipeline import CODE_TO_EVENT, SPECTRAL_BANDS


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

ANALYSIS_CHANNELS: List[str] = ["C3", "Cz", "C4"]

ANALYSIS_CLASSES: List[str] = [
    "elbow_flexion", "elbow_extension", "hand_close", "hand_open", "rest",
]
MOVEMENT_CLASSES: List[str] = [
    "elbow_flexion", "elbow_extension", "hand_close", "hand_open",
]
# Pooled "movement" pseudo-class is appended for the pooled (Movement vs Rest) figures.
POOLED_LABEL: str = "movement"
CURVE_CLASSES: List[str] = ANALYSIS_CLASSES + [POOLED_LABEL]

CURVE_TYPES: List[str] = ["psd_baseline", "psd_active", "erds_curve"]
CONDITIONS: List[str] = ["ME", "MI"]

_CONDITION_FOLDER: Dict[str, str] = {
    "ME": "motor_execution",
    "MI": "motor_imagery",
}

# Short metric name → column in the scalar DataFrame (active window for PSD metrics).
METRIC_COLUMNS: Dict[str, str] = {
    "erds_band":   "erds_band",
    "median_freq": "medfreq_active",
    "auc":         "auc_active",
    "max_power":   "maxpow_active",
}
# Order in which scalar figures are produced.
SCALAR_METRIC_ORDER: List[str] = ["erds_band", "median_freq", "auc", "max_power"]

_BANDS: List[str] = list(SPECTRAL_BANDS.keys())   # ['alpha', 'beta']

# ── Presentation labels (figures only) ───────────────────────────────────────
# Class → short x-tick abbreviation, in ANALYSIS_CLASSES order.
CLASS_ABBR: Dict[str, str] = {
    "elbow_flexion":   "EF",
    "elbow_extension": "EE",
    "hand_close":      "HC",
    "hand_open":       "HO",
    "rest":            "Rest",
}
CLASS_ABBR_ORDER: List[str] = [CLASS_ABBR[c] for c in ANALYSIS_CLASSES]

# Band → Pascal-case row label.
BAND_LABELS: Dict[str, str] = {"alpha": "Alpha", "beta": "Beta"}
BAND_LABEL_ORDER: List[str] = [BAND_LABELS[b] for b in _BANDS]

# Metric → (figure title, y-axis label).
METRIC_TITLES: Dict[str, str] = {
    "erds_band":   "ERD/ERS",
    "median_freq": "Median Frequency",
    "auc":         "Area Under Curve",
    "max_power":   "Max Power",
}
METRIC_YLABELS: Dict[str, str] = {
    "erds_band":   "ERD/ERS (dB)",
    "median_freq": "Median Frequency (Hz)",
    "auc":         "AUC (µV²)",
    "max_power":   "Max Power (µV²/Hz)",
}


def _log(msg: str) -> None:
    print(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# A — Discover pickles
# ═══════════════════════════════════════════════════════════════════════════════

def _run_sort_key(path: Path) -> int:
    """Natural sort key so run10 follows run9 (not run1)."""
    m = re.search(r"run(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def discover_pickles(processed_root: Path, condition: str) -> Dict[str, List[Path]]:
    """
    Locate every per-run pickle for one condition and group them by subject.

    Returns
    ───────
    Dict[subject_id → sorted List[Path]], where subject_id is the 'Sxx' folder
    name (e.g. 'S02'). Runs are ordered run1 … run10.
    """
    folder = _CONDITION_FOLDER[condition]
    cond_root = processed_root / folder
    if not cond_root.exists():
        return {}

    groups: Dict[str, List[Path]] = {}
    for subj_dir in sorted(cond_root.glob("S*")):
        if not subj_dir.is_dir():
            continue
        subject_id = subj_dir.name.split("_")[0]   # 'S02_ME' → 'S02'
        runs = sorted(subj_dir.glob("*.pkl"), key=_run_sort_key)
        if runs:
            groups[subject_id] = runs
    return groups


# ═══════════════════════════════════════════════════════════════════════════════
# B — Load + concatenate a subject's runs
# ═══════════════════════════════════════════════════════════════════════════════

def _concat_metrics(metrics_list: List[Dict]) -> Dict:
    """Concatenate the nested metrics dicts along the epoch axis (axis 0)."""
    out: Dict = {}
    for band in metrics_list[0]:
        out[band] = {"freqs_band": metrics_list[0][band]["freqs_band"]}
        for window in ("baseline", "active"):
            out[band][window] = {
                metric: np.concatenate([m[band][window][metric] for m in metrics_list], axis=0)
                for metric in metrics_list[0][band][window]
            }
        out[band]["erds_band"] = np.concatenate(
            [m[band]["erds_band"] for m in metrics_list], axis=0
        )
    return out


def load_subject(paths: List[Path]) -> Dict:
    """
    Read every run pickle for a subject and concatenate along the epoch axis.

    The per-run curves (psd_baseline, psd_active, erds_curve), events, and all
    nested band scalars are stacked so the subject is represented as a single
    combined recording of ~60 trials/class.

    freqs and ch_names are asserted identical across runs.
    """
    runs = [pickle.load(open(p, "rb")) for p in paths]

    freqs    = runs[0]["freqs"]
    ch_names = list(runs[0]["ch_names"])
    cond     = runs[0]["condition"]
    for r in runs[1:]:
        if not np.array_equal(r["freqs"], freqs):
            raise ValueError("Inconsistent freqs across runs")
        if list(r["ch_names"]) != ch_names:
            raise ValueError("Inconsistent ch_names across runs")

    combined = {
        "psd_baseline": np.concatenate([r["psd_baseline"] for r in runs], axis=0),
        "psd_active":   np.concatenate([r["psd_active"]   for r in runs], axis=0),
        "erds_curve":   np.concatenate([r["erds_curve"]   for r in runs], axis=0),
        "events":       np.concatenate([r["events"]       for r in runs], axis=0),
        "metrics":      _concat_metrics([r["metrics"] for r in runs]),
        "freqs":        freqs,
        "ch_names":     ch_names,
        "condition":    cond,
    }
    return combined


def load_all_subjects(processed_root: Path) -> Dict[Tuple[str, str], Dict]:
    """
    Load and concatenate every subject for both conditions.

    Returns Dict[(condition, subject_id) → combined recording dict].
    """
    subjects: Dict[Tuple[str, str], Dict] = {}
    for cond in CONDITIONS:
        groups = discover_pickles(processed_root, cond)
        for subj, paths in groups.items():
            subjects[(cond, subj)] = load_subject(paths)
            _log(f"[load]    {cond} {subj}: {len(paths)} runs -> "
                 f"{subjects[(cond, subj)]['events'].shape[0]} epochs")
    return subjects


# ═══════════════════════════════════════════════════════════════════════════════
# Shared statistics helper
# ═══════════════════════════════════════════════════════════════════════════════

def _mean_ci(data: np.ndarray, axis: int = 0, conf: float = 0.95
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    t-based mean and (lo, hi) confidence bounds along `axis`.

    CI = mean ± t(0.5+conf/2, n-1) · SEM. For n < 2 the bounds collapse to the mean.
    NaNs are ignored.
    """
    n = data.shape[axis]
    mean = np.nanmean(data, axis=axis)
    if n < 2:
        return mean, mean.copy(), mean.copy()
    sem = np.nanstd(data, axis=axis, ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.5 + conf / 2.0, n - 1)
    return mean, mean - tcrit * sem, mean + tcrit * sem


def _class_names(events: np.ndarray) -> np.ndarray:
    """Map the event-code column to class-name strings."""
    return np.array([CODE_TO_EVENT[c] for c in events[:, 2]])


# ═══════════════════════════════════════════════════════════════════════════════
# C — Tidy scalar DataFrame
# ═══════════════════════════════════════════════════════════════════════════════

def build_scalar_dataframe(subjects: Dict[Tuple[str, str], Dict]) -> pd.DataFrame:
    """
    Build the master tidy DataFrame of per-epoch band scalars.

    One row per (subject, condition, class, channel, band, epoch_idx). All 7
    classes and all 5 channels are kept; filtering to the analysis subset happens
    at plot time.

    Columns
    ───────
    subject, condition, class, channel, band, epoch_idx,
    auc_baseline, auc_active, medfreq_baseline, medfreq_active,
    maxpow_baseline, maxpow_active, erds_band
    """
    frames: List[pd.DataFrame] = []

    for (cond, subj), rec in subjects.items():
        ch_names = rec["ch_names"]
        n_ch = len(ch_names)
        classes = _class_names(rec["events"])
        n_ep = len(classes)

        for band in _BANDS:
            mb = rec["metrics"][band]
            frame = pd.DataFrame({
                "subject":   subj,
                "condition": cond,
                "class":     np.repeat(classes, n_ch),
                "channel":   np.tile(ch_names, n_ep),
                "band":      band,
                "epoch_idx": np.repeat(np.arange(n_ep), n_ch),
                "auc_baseline":    mb["baseline"]["auc"].reshape(-1),
                "auc_active":      mb["active"]["auc"].reshape(-1),
                "medfreq_baseline": mb["baseline"]["median_freq"].reshape(-1),
                "medfreq_active":   mb["active"]["median_freq"].reshape(-1),
                "maxpow_baseline":  mb["baseline"]["max_power"].reshape(-1),
                "maxpow_active":    mb["active"]["max_power"].reshape(-1),
                "erds_band":        mb["erds_band"].reshape(-1),
            })
            frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    cat_cols = ["subject", "condition", "class", "channel", "band"]
    df[cat_cols] = df[cat_cols].astype("category")
    _log(f"[scalars] DataFrame: {len(df):,} rows × {df.shape[1]} cols | "
         f"subjects={df['subject'].nunique()} conditions={df['condition'].nunique()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# D — Subject-mean curve arrays (+ within-subject CI)
# ═══════════════════════════════════════════════════════════════════════════════

def build_curve_arrays(subjects: Dict[Tuple[str, str], Dict]) -> Dict:
    """
    Build subject-mean spectral curves and within-subject 95% CI bounds.

    For every (condition, curve_type) a triple of arrays shaped
    (n_subjects, n_curve_classes, n_channels, n_freqs) is produced — the subject
    mean across epochs and the within-subject CI bounds across epochs. The
    'movement' pseudo-class pools the four movement classes' epochs before
    averaging.

    Returns
    ───────
    {
      'freqs'    : (n_freqs,),
      'channels' : List[str]      (all 5, in recording order),
      'classes'  : CURVE_CLASSES  (5 analysis classes + 'movement'),
      'subjects' : {condition: [subject_id, ...]},
      'data'     : {(condition, curve_type): {'mean','ci_lo','ci_hi'}},
    }
    """
    # Use the first available recording for axis metadata
    any_rec = next(iter(subjects.values()))
    freqs    = any_rec["freqs"]
    channels = list(any_rec["ch_names"])
    n_freqs  = len(freqs)
    n_ch     = len(channels)
    n_cls    = len(CURVE_CLASSES)

    subj_by_cond: Dict[str, List[str]] = {
        cond: sorted({s for (c, s) in subjects if c == cond}) for cond in CONDITIONS
    }

    data: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

    for cond in CONDITIONS:
        subj_list = subj_by_cond[cond]
        if not subj_list:
            continue
        n_subj = len(subj_list)

        for ctype in CURVE_TYPES:
            mean  = np.full((n_subj, n_cls, n_ch, n_freqs), np.nan)
            ci_lo = np.full_like(mean, np.nan)
            ci_hi = np.full_like(mean, np.nan)

            for si, subj in enumerate(subj_list):
                rec = subjects[(cond, subj)]
                classes = _class_names(rec["events"])
                curve = rec[ctype]                       # (n_ep, n_ch, n_freqs)

                for ci, klass in enumerate(CURVE_CLASSES):
                    if klass == POOLED_LABEL:
                        sel = np.isin(classes, MOVEMENT_CLASSES)
                    else:
                        sel = classes == klass
                    if not sel.any():
                        continue
                    m, lo, hi = _mean_ci(curve[sel], axis=0)
                    mean[si, ci]  = m
                    ci_lo[si, ci] = lo
                    ci_hi[si, ci] = hi

            data[(cond, ctype)] = {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi}

    _log(f"[curves]  built {len(data)} (condition×curve_type) blocks | "
         f"classes={n_cls} channels={n_ch} freqs={n_freqs}")
    return {
        "freqs":    freqs,
        "channels": channels,
        "classes":  CURVE_CLASSES,
        "subjects": subj_by_cond,
        "data":     data,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# E — Persist / reload backbone
# ═══════════════════════════════════════════════════════════════════════════════

def save_backbone(df: pd.DataFrame, curves: Dict, backbone_dir: Path) -> None:
    """Persist the scalar DataFrame (pickle) and curve arrays (compressed npz)."""
    backbone_dir.mkdir(parents=True, exist_ok=True)

    df.to_pickle(backbone_dir / "scalars.pkl")

    flat: Dict[str, np.ndarray] = {
        "freqs":    curves["freqs"],
        "channels": np.array(curves["channels"], dtype=object),
        "classes":  np.array(curves["classes"], dtype=object),
    }
    for cond in CONDITIONS:
        flat[f"subjects__{cond}"] = np.array(curves["subjects"].get(cond, []), dtype=object)
    for (cond, ctype), block in curves["data"].items():
        for stat, arr in block.items():
            flat[f"data__{cond}__{ctype}__{stat}"] = arr

    np.savez_compressed(backbone_dir / "curves.npz", **flat)
    _log(f"[saved]   {backbone_dir / 'scalars.pkl'}  &  {backbone_dir / 'curves.npz'}")


def load_backbone(backbone_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """Reload the persisted backbone, reconstructing the nested curves dict."""
    df = pd.read_pickle(backbone_dir / "scalars.pkl")

    npz = np.load(backbone_dir / "curves.npz", allow_pickle=True)
    curves: Dict = {
        "freqs":    npz["freqs"],
        "channels": list(npz["channels"]),
        "classes":  list(npz["classes"]),
        "subjects": {cond: list(npz[f"subjects__{cond}"]) for cond in CONDITIONS
                     if f"subjects__{cond}" in npz.files},
        "data":     {},
    }
    for key in npz.files:
        if not key.startswith("data__"):
            continue
        _, cond, ctype, stat = key.split("__")
        curves["data"].setdefault((cond, ctype), {})[stat] = npz[key]

    _log(f"[loaded]  backbone from {backbone_dir}")
    return df, curves


# ═══════════════════════════════════════════════════════════════════════════════
# F1 — Curve retrieval / aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def _channel_indices(channels: List[str], wanted: List[str]) -> List[int]:
    return [channels.index(ch) for ch in wanted]


def get_curve(curves: Dict, condition: str, curve_type: str, klass: str,
              level: str, subject: Optional[str] = None
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Retrieve a curve and its 95% CI band for one (condition, curve_type, class).

    level='subject' → that subject's mean + within-subject CI (across epochs),
                      requires `subject`.
    level='group'   → mean across subject-means + CI across subject-means.

    Returns (freqs, mean, lo, hi, n) where mean/lo/hi are (n_channels, n_freqs)
    over ALL recording channels and n is the number of contributing subjects
    (group) or epochs-subjects (1 for subject level).
    """
    block = curves["data"][(condition, curve_type)]
    cls_idx = curves["classes"].index(klass)
    freqs = curves["freqs"]

    if level == "subject":
        subj_list = curves["subjects"][condition]
        si = subj_list.index(subject)
        return (freqs,
                block["mean"][si, cls_idx],
                block["ci_lo"][si, cls_idx],
                block["ci_hi"][si, cls_idx],
                1)

    # group: aggregate the per-subject means across the subject axis
    subj_means = block["mean"][:, cls_idx]            # (n_subj, n_ch, n_freqs)
    n = int(np.sum(~np.all(np.isnan(subj_means), axis=(1, 2))))
    mean, lo, hi = _mean_ci(subj_means, axis=0)
    return freqs, mean, lo, hi, n


# ═══════════════════════════════════════════════════════════════════════════════
# F2 — Curve plotters
# ═══════════════════════════════════════════════════════════════════════════════

_PSD_COLORS = {"psd_baseline": "tab:blue", "psd_active": "tab:red"}
_PSD_LABELS = {"psd_baseline": "baseline (pre-cue)", "psd_active": "active (task)"}


def _add_band_shading(ax) -> None:
    """Shade the alpha and beta bands as light vertical spans."""
    for (lo, hi), color in zip(SPECTRAL_BANDS.values(), ("0.85", "0.92")):
        ax.axvspan(lo, hi, color=color, zorder=0)


def _style_curve_ax(ax, title: str, ylabel: str, shade: bool, zero_line: bool) -> None:
    if shade:
        _add_band_shading(ax)
    if zero_line:
        ax.axhline(0.0, color="k", lw=0.8, ls="--", zorder=1)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


def _class_title(klass: str) -> str:
    """Human-readable Title-case label for a class key (e.g. elbow_flexion → 'Elbow Flexion')."""
    if klass == POOLED_LABEL:
        return "Movement"
    return klass.replace("_", " ").title()


def _plot_psd_grid(curves: Dict, condition: str, rows: List[str], level: str,
                   subject: Optional[str], leg_anchor: Tuple[float, float], title: str):
    """Shared PSD overlay grid: rows×channels, baseline vs active per panel."""
    chans = ANALYSIS_CHANNELS
    ch_idx = _channel_indices(curves["channels"], chans)
    nrow, ncol = len(rows), len(chans)

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.5 * nrow),
                             squeeze=False, sharex=True)
    for ri, klass in enumerate(rows):
        for cj, (ch, cidx) in enumerate(zip(chans, ch_idx)):
            ax = axes[ri][cj]
            for ctype in ("psd_baseline", "psd_active"):
                freqs, mean, lo, hi, n = get_curve(
                    curves, condition, ctype, klass, level, subject)
                ax.plot(freqs, mean[cidx], color=_PSD_COLORS[ctype],
                        lw=1.3, label=_PSD_LABELS[ctype])
                ax.fill_between(freqs, lo[cidx], hi[cidx],
                                color=_PSD_COLORS[ctype], alpha=0.20)
            _style_curve_ax(ax, f"{_class_title(klass)} — {ch}", "PSD (µV²/Hz)",
                            shade=True, zero_line=False)

    # Show the numeric x-tick labels on every subplot (not only the bottom row).
    for ax in axes.flat:
        ax.tick_params(labelbottom=True)

    # Single legend outside the plot area, anchored at the figure's upper-right
    # corner (captured by bbox_inches='tight' on save).
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=True,
               bbox_to_anchor=leg_anchor, fontsize=7,)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_psd_pooled(curves: Dict, condition: str, level: str,
                    subject: Optional[str] = None):
    """PSD Fig 1 — pooled: rows {Movement, Rest} × channels."""
    who = subject if level == "subject" else "group"
    return _plot_psd_grid(curves, condition, [POOLED_LABEL, "rest"], level, subject,
                          (0.995, 1.05), f"PSD baseline vs active — {condition} — {who} (pooled)")


def plot_psd_per_class(curves: Dict, condition: str, level: str,
                       subject: Optional[str] = None):
    """PSD Fig 2 — per-class: rows = 5 classes × channels."""
    who = subject if level == "subject" else "group"
    return _plot_psd_grid(curves, condition, ANALYSIS_CLASSES, level, subject, (0.995, 0.995),
                          f"PSD baseline vs active — {condition} — {who} (per-class)")


def _plot_erds_row(curves: Dict, condition: str, classes: List[str], level: str,
                   subject: Optional[str], leg_anchor: Tuple[float, float], title: str):
    """Shared ERD/ERS single-row grid: 1 × channels, classes overlaid per panel."""
    chans = ANALYSIS_CHANNELS
    ch_idx = _channel_indices(curves["channels"], chans)
    palette = sns.color_palette("tab10", n_colors=len(classes))

    fig, axes = plt.subplots(1, len(chans), figsize=(3.4 * len(chans), 3.0),
                             squeeze=False, sharex=True, sharey=True)
    for cj, (ch, cidx) in enumerate(zip(chans, ch_idx)):
        ax = axes[0][cj]
        for klass, color in zip(classes, palette):
            freqs, mean, lo, hi, n = get_curve(
                curves, condition, "erds_curve", klass, level, subject)
            label = "Movement" if klass == POOLED_LABEL else klass
            ax.plot(freqs, mean[cidx], color=color, lw=1.3, label=label)
            ax.fill_between(freqs, lo[cidx], hi[cidx], color=color, alpha=0.15)
        _style_curve_ax(ax, ch, "ERD/ERS (dB)", shade=True, zero_line=True)
        # if cj == len(chans) - 1:
        #     ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(0.995, 0.995))

    # Single legend outside the plot area, anchored at the figure's upper-right
    # corner (captured by bbox_inches='tight' on save).
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=True,
               bbox_to_anchor=leg_anchor, fontsize=7)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_erds_pooled(curves: Dict, condition: str, level: str,
                     subject: Optional[str] = None):
    """ERD/ERS Fig 1 — pooled: Movement vs Rest overlaid, 1 × channels."""
    who = subject if level == "subject" else "group"
    return _plot_erds_row(curves, condition, [POOLED_LABEL, "rest"], level, subject,
                          (0.995, 0.995), f"ERD/ERS — {condition} — {who} (Movement vs Rest)")


def plot_erds_per_class(curves: Dict, condition: str, level: str,
                        subject: Optional[str] = None):
    """ERD/ERS Fig 2 — per-class: all 5 classes overlaid, 1 × channels."""
    who = subject if level == "subject" else "group"
    return _plot_erds_row(curves, condition, ANALYSIS_CLASSES, level, subject,
                          (0.995, 1.05), f"ERD/ERS — {condition} — {who} (all classes)")


# ═══════════════════════════════════════════════════════════════════════════════
# F3 — Scalar boxplots
# ═══════════════════════════════════════════════════════════════════════════════

def _scalar_plot_frame(df: pd.DataFrame, metric: str, level: str,
                       subject: Optional[str]) -> pd.DataFrame:
    """
    Slice/aggregate the master DataFrame for one scalar boxplot.

    Filters to the analysis class/channel subset. For level='group' the rows are
    collapsed to one value per subject (the subject mean) so each box spans the
    14 subject-means; for level='subject' the raw epoch rows of one subject are
    returned so each box spans that subject's ~60 epochs.
    """
    col = METRIC_COLUMNS[metric]
    sub = df[df["class"].isin(ANALYSIS_CLASSES) & df["channel"].isin(ANALYSIS_CHANNELS)].copy()

    if level == "subject":
        sub = sub[sub["subject"] == subject]
        return sub

    grouped = (sub.groupby(["subject", "condition", "class", "channel", "band"],
                           observed=True)[col]
                  .mean().reset_index())
    return grouped


def plot_scalar_box(df: pd.DataFrame, metric: str, level: str,
                    subject: Optional[str] = None):
    """
    Boxplot for one scalar metric: rows=band × cols=channel, x=class, hue=condition.

    Only the x-axis is shared across facets (each panel autoscales its y). Class
    names are shown as short abbreviations (EF/EE/HC/HO/Rest); band rows and the
    y-axis carry Pascal-case labels; the condition legend sits in the figure's
    upper-right corner.

    Returns the seaborn FacetGrid (`.figure` for saving).
    """
    col = METRIC_COLUMNS[metric]
    data = _scalar_plot_frame(df, metric, level, subject).copy()

    # Cast the facet/hue/x columns to plain strings. This drops the unused
    # categorical levels (e.g. C1, C2) that otherwise make seaborn collapse every
    # box into the first column, and lets us relabel for presentation.
    data["channel"]    = data["channel"].astype(str)
    data["condition"]  = data["condition"].astype(str)
    data["class_abbr"] = data["class"].astype(str).map(CLASS_ABBR)
    data["band_label"] = data["band"].astype(str).map(BAND_LABELS)

    who = subject if level == "subject" else "Group"

    g = sns.catplot(
        data=data, kind="box",
        x="class_abbr", y=col, hue="condition",
        row="band_label", col="channel",
        order=CLASS_ABBR_ORDER, col_order=ANALYSIS_CHANNELS, row_order=BAND_LABEL_ORDER,
        height=2.6, aspect=1.4, fliersize=1.5, linewidth=0.8,
        margin_titles=True, sharex=True, sharey=False,
    )
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.set_axis_labels("Class", METRIC_YLABELS[metric])
    g.set_xticklabels(CLASS_ABBR_ORDER, rotation=0, fontsize=8)
    g.figure.suptitle(f"{METRIC_TITLES[metric]} — {who}", fontsize=12)
    g.figure.tight_layout(rect=(0, 0, 1, 0.96))

    # Place the condition legend inside the figure's upper-right corner (it may
    # overlap the top-right panel — preferred over reserving a column for it).
    if g.legend is not None:
        sns.move_legend(g, loc="upper right", bbox_to_anchor=(0.995, 0.995),
                        title="Condition", frameon=True)
    return g


# ═══════════════════════════════════════════════════════════════════════════════
# F4 — Summary tables
# ═══════════════════════════════════════════════════════════════════════════════

def summary_tables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive summary across subject-means: mean / SD / 95% CI per
    (condition, class, channel, band) for every scalar metric.

    Subjects are the unit of observation (collapse epochs → subject mean first).
    """
    keys = ["condition", "class", "channel", "band"]
    sub = df[df["class"].isin(ANALYSIS_CLASSES) & df["channel"].isin(ANALYSIS_CHANNELS)].copy()
    value_cols = list(dict.fromkeys(METRIC_COLUMNS.values()))

    # 1) collapse epochs -> one value per subject
    subj_means = (sub.groupby(["subject"] + keys, observed=True)[value_cols]
                     .mean().reset_index())

    # 2) long form so each metric is a row, then scalar aggregations across subjects
    long = subj_means.melt(id_vars=["subject"] + keys, value_vars=value_cols,
                           var_name="metric", value_name="value")
    table = (long.groupby(keys + ["metric"], observed=True)["value"]
                 .agg(mean="mean", sd="std", n="count").reset_index())

    # 3) t-based 95% CI from mean/sd/n
    n = table["n"].to_numpy()
    sem = table["sd"].to_numpy() / np.sqrt(np.where(n > 0, n, np.nan))
    tcrit = np.where(n >= 2, stats.t.ppf(0.975, np.where(n >= 2, n - 1, 1)), np.nan)
    half = tcrit * sem
    table["ci_lo"] = table["mean"] - half
    table["ci_hi"] = table["mean"] + half
    return table


# ═══════════════════════════════════════════════════════════════════════════════
# G — Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def _save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main_analysis(
    processed_root: Optional[str] = None,
    results_root: Optional[str] = None,
    rebuild: bool = False,
    per_subject_figs: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run the full descriptive analysis.

    discover → load/build-or-reload backbone → curve figures → scalar figures →
    summary tables. Defaults mirror eeg_pipeline.main_load_preprocess():
        processed_root → <project_root>/data_processed/
        results_root   → <project_root>/results/

    Parameters
    ──────────
    rebuild          : if False and a persisted backbone exists, reload it
                       instead of re-reading every per-run pickle.
    per_subject_figs : also emit per-subject curve/scalar figures (many files).
                       Group-level figures are always produced.

    Returns the (scalar DataFrame, curves dict) backbone.
    """
    project_root = Path(__file__).resolve().parent.parent
    proc_root = Path(processed_root) if processed_root else project_root / "data" / "data_processed"
    res_root  = Path(results_root)   if results_root   else project_root / "results"

    backbone_dir = res_root / "backbone"
    fig_root     = res_root / "figures"
    tbl_root     = res_root / "tables"

    # ── Backbone: build or reload ────────────────────────────────────────────
    have_backbone = (backbone_dir / "scalars.pkl").exists() and (backbone_dir / "curves.npz").exists()
    if have_backbone and not rebuild:
        df, curves = load_backbone(backbone_dir)
    else:
        subjects = load_all_subjects(proc_root)
        if not subjects:
            raise FileNotFoundError(f"No processed pickles found under {proc_root}")
        df = build_scalar_dataframe(subjects)
        curves = build_curve_arrays(subjects)
        save_backbone(df, curves, backbone_dir)

    # ── Curve figures ────────────────────────────────────────────────────────
    curve_plotters = {
        "psd/pooled":       plot_psd_pooled,
        "psd/per_class":    plot_psd_per_class,
        "erds/pooled":      plot_erds_pooled,
        "erds/per_class":   plot_erds_per_class,
    }
    for cond in CONDITIONS:
        if not curves["subjects"].get(cond):
            continue
        for name, fn in curve_plotters.items():
            _save_fig(fn(curves, cond, level="group"),
                      fig_root / name / f"group_{cond}.png")
            if per_subject_figs:
                for subj in curves["subjects"][cond]:
                    _save_fig(fn(curves, cond, level="subject", subject=subj),
                              fig_root / name / "per_subject" / f"{subj}_{cond}.png")

    # ── Scalar figures ───────────────────────────────────────────────────────
    for metric in SCALAR_METRIC_ORDER:
        _save_fig(plot_scalar_box(df, metric, level="group").figure,
                  fig_root / "scalars" / f"group_{metric}.png")
        if per_subject_figs:
            for subj in sorted(df["subject"].unique()):
                _save_fig(plot_scalar_box(df, metric, level="subject", subject=subj).figure,
                          fig_root / "scalars" / "per_subject" / f"{subj}_{metric}.png")

    # ── Summary tables ───────────────────────────────────────────────────────
    tbl_root.mkdir(parents=True, exist_ok=True)
    table = summary_tables(df)
    table.to_csv(tbl_root / "summary_by_group.csv", index=False)
    _log(f"[tables]  {tbl_root / 'summary_by_group.csv'}  ({len(table)} rows)")

    _log("[done]    descriptive analysis complete")
    return df, curves


if __name__ == "__main__":
    main_analysis(
        rebuild=False,
        per_subject_figs=True,
    )
