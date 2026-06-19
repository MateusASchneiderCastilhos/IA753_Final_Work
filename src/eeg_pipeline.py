"""
eeg_pipeline.py
───────────────────────────────────────────────────────────────────────────────
EEG Analysis Pipeline — Steps 0 through 4

Step 0a  load_raw()                  Load GDF, extract events (no pick yet)
Step 0b  preprocess_raw()            Re-ref → filter → notch → pick channels
Step 0c  create_epochs()             Segment, reject artifacts, detrend

Step 1+2 compute_psd_per_epoch()     PSD for every epoch  (MNE compute_psd)
         average_psd_per_event()     Average across epochs per event type

Step 3   process_single_recording()  Full pipeline for one GDF file → .pkl
Step 4   process_subject()           Loop over 10 recordings + aggregate

Utility  subject_avg_to_dataframe()  Convert per-subject result to tidy DataFrame
───────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

import mne
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

EVENT_TO_CODE: Dict[str, int] = {
    "elbow_flexion":   1536,
    "elbow_extension": 1537,
    "supination":      1538,
    "pronation":       1539,
    "hand_close":      1540,
    "hand_open":       1541,
    "rest":            1542,
}

CODE_TO_EVENT: Dict[int, str] = {v: k for k, v in EVENT_TO_CODE.items()}

CHANNELS_OF_INTEREST: List[str] = ["C3", "C1", "Cz", "C2", "C4"]

EOG_CHANNELS: List[str] = ["eog-l", "eog-m", "eog-r"]

EXCLUDE_CHANNELS: List[str] = [
    "thumb_near", "thumb_far", "thumb_index", "index_near", "index_far",
    "index_middle", "middle_near", "middle_far", "middle_ring", "ring_near",
    "ring_far", "ring_little", "litte_near", "litte_far", "thumb_palm",
    "wrist_bend", "roll", "pitch", "gesture",
    "handPosX", "handPosY", "handPosZ",
    "elbowPosX", "elbowPosY", "elbowPosZ",
    "ShoulderAdductio", "ShoulderFlexionE", "ShoulderRotation",
    "Elbow", "ProSupination", "Wrist", "GripPressure",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _event_converter(event_str: str) -> Optional[int]:
    """Convert an annotation string to an integer event code.
    Returns None for unknown annotations so MNE ignores them."""
    valid = {"1536", "1537", "1538", "1539", "1540", "1541", "1542"}
    return int(event_str) if isinstance(event_str, str) and event_str in valid else None


def _log(msg: str) -> None:
    print(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 0a — Load
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw(gdf_file: str) -> Tuple[mne.io.Raw, np.ndarray, Dict]:
    """
    Load a GDF file and extract embedded event annotations.

    Channel picking is intentionally deferred to preprocess_raw() so that
    set_eeg_reference() has access to all EEG channels when computing the
    Common Average Reference.
    """
    raw = mne.io.read_raw_gdf(
        gdf_file,
        eog=EOG_CHANNELS,
        exclude=EXCLUDE_CHANNELS,
        preload=False,
        verbose=False,
    )
    events, event_id = mne.events_from_annotations(
        raw, event_id=_event_converter, verbose=False
    )
    _log(
        f"[load]    {Path(gdf_file).name} | "
        f"{len(raw.ch_names)} ch | "
        f"{raw.n_times} samples ({raw.times[-1]:.1f} s) | "
        f"{len(events)} events"
    )
    return raw, events, event_id


# ═══════════════════════════════════════════════════════════════════════════════
# Step 0b — Preprocess Raw
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_raw(
    raw: mne.io.Raw,
    reference: str = "average",
    l_freq: float = 0.5,
    h_freq: float = 100.0,
    notch_freqs: Optional[List[float]] = None,
) -> mne.io.Raw:
    """
    Preprocess the continuous Raw signal before epoching.

    Operation order (order is critical)
    ─────────────────────────────────────
    1. Load into RAM
    2. Re-reference with ALL EEG channels present
       set_eeg_reference() must precede pick() so that every electrode
       contributes to the Common Average Reference calculation.
    3. Bandpass FIR filter (zero-phase / acausal via filtfilt internally)
       Removes slow electrode drift below l_freq and high-frequency
       noise above h_freq. Applied on the continuous signal to avoid
       edge-effect ringing at epoch boundaries.
    4. Notch FIR filter (zero-phase)
       Suppresses Brazilian power-line interference at 60 Hz and its
       harmonics (120, 180 Hz).
    5. Pick only CHANNELS_OF_INTEREST
       Deferred to last so steps 2–4 use all available channels;
       from here on only the 5 motor-cortex channels are in memory.
    """
    # if notch_freqs is None:
    #     notch_freqs = [60.0, 120.0, 180.0]

    raw.load_data()

    raw.set_eeg_reference(reference, projection=False, verbose=False)

    raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir", phase="zero", verbose=False)

    if notch_freqs is not None:
        raw.notch_filter(freqs=notch_freqs, method="fir", phase="zero", verbose=False)

    raw.pick(CHANNELS_OF_INTEREST)

    _log(
        f"[preproc] ref={reference} | bp={l_freq}–{h_freq} Hz | "
        f"notch={notch_freqs} Hz | ch={raw.ch_names}"
    )
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
# Step 0c — Epoch segmentation
# ═══════════════════════════════════════════════════════════════════════════════

def create_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    tmin: float = -0.5,
    tmax: float = 4.0,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    reject_uv: float = 150.0,
    flat_uv: float = 1.0,
    verbose: bool = False,
) -> mne.Epochs:
    """
    Segment the preprocessed Raw signal into epochs around event markers.

    Parameters
    ──────────
    tmin      : seconds before event onset (negative → pre-stimulus window)
    tmax      : seconds after  event onset
    baseline  : (t_start, t_end) window for baseline subtraction from the
                raw voltage signal.  Two common choices:
                  None        — no baseline correction on the signal; rely
                                on detrend=1 alone. Recommended when the
                                PSD-level comparison (Step 6) will use the
                                rest epoch as its own baseline reference.
                  (None, 0)   — subtract the mean of the pre-stimulus window
                                from each epoch. Useful if the signal has a
                                slow DC drift that detrend=1 does not fully
                                remove.
    reject_uv : upper peak-to-peak threshold (µV).
                Epochs where ANY channel exceeds this are dropped.
                Catches blink residuals and movement transients.
    flat_uv   : lower peak-to-peak threshold (µV).
                Epochs where ANY channel stays below this are dropped.
                Catches disconnected or saturated electrodes.
    detrend=1 : removes the best-fit linear trend from each epoch/channel
                before any spectral analysis. Prevents the 1/f² spectral
                leakage that a linear drift causes at low frequencies.

    Note on reject vs flat
    ───────────────────────
    Both are complementary. The original code had flat={'eeg': 400e-6}
    which would never trigger (400 µV is far above any real EEG amplitude);
    corrected here to 1 µV.
    """
    epochs = mne.Epochs(
        raw,
        events,
        event_id=EVENT_TO_CODE,
        tmin=tmin,
        tmax=tmax,
        picks="eeg",
        baseline=baseline,
        # reject={"eeg": reject_uv * 1e-6} if reject_uv is not None else None, # Check these or other values in some literature
        # flat={"eeg": flat_uv * 1e-6} if flat_uv is not None else None, # Check these or other values in some literature
        detrend=1,
        preload=True,
        verbose=verbose,
    )
    epochs.drop_bad(verbose=verbose)
    _log(
        f"[epochs]  {len(epochs)} kept | "
        f"shape: {epochs.get_data().shape}  (epochs × ch × samples)"
    )
    return epochs


# ═══════════════════════════════════════════════════════════════════════════════
# Steps 1 + 2 — PSD per epoch, then average per event
# ═══════════════════════════════════════════════════════════════════════════════

def compute_psd_per_epoch(
    epochs: mne.Epochs,
    method: str = "welch",
    fmin: float = 1.0,
    fmax: float = 45.0,
    n_per_seg: Optional[int] = None,
    n_overlap: int = 0,
    window: str = "hann",
    verbose: bool = False,
) -> mne.time_frequency.EpochsSpectrum:
    """
    Steps 1 + 2: compute PSD for every epoch via MNE's compute_psd().

    MNE handles both steps atomically:
      Step 1 — accesses the voltage time-series of each epoch internally
      Step 2 — applies the chosen spectral estimator per epoch

    The result is an EpochsSpectrum with shape (n_epochs, n_channels, n_freqs)
    that preserves the event structure of the original Epochs object.

    Welch parameter guidance
    ──────────────────────────
    Frequency resolution:  Δf = sfreq / n_per_seg
    Number of segments K:  ≈ n_times / (n_per_seg − n_overlap)

    Recommended defaults for 256 Hz data, 4.5 s epochs (1152 samples):
      n_per_seg = 512   →  Δf = 0.5 Hz, K ≈ 3 non-overlapping segments
      n_overlap = 256   →  50 % overlap, K ≈ 5 effective segments
                            → variance ∝ 1/(K × n_epochs)

    The overlap here is *within* a single epoch (internal to Welch).
    It is entirely independent of inter-epoch overlap, which must remain
    zero to keep realizations independent for ensemble averaging.

    method='multitaper' is also accepted; in that case n_per_seg and
    n_overlap are ignored and MNE uses Slepian tapers internally.
    """
    spectrum = epochs.compute_psd(
        method=method,
        fmin=fmin,
        fmax=fmax,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        window=window,
        # remove_dc=True, # Why its is not interesting herr but as normalization in analysis process?
        verbose=verbose,
    )
    freqs = spectrum.freqs
    _log(
        f"[psd]     method={method} | "
        f"shape: {spectrum.get_data().shape}  (epochs × ch × freqs) | "
        f"Δf = {freqs[1] - freqs[0]:.3f} Hz | "
        f"range: {freqs[0]:.1f}–{freqs[-1]:.1f} Hz"
    )
    return spectrum


def average_psd_per_event(
    spectrum: mne.time_frequency.EpochsSpectrum,
    epochs: mne.Epochs,
) -> Dict[str, Dict]:
    """
    Average the per-epoch PSD across epochs that share the same event type.

    Implementation note
    ────────────────────
    EpochsSpectrum preserves the event array from the parent Epochs object.
    We use direct numpy indexing on epochs.events[:, 2] (the event-code column)
    rather than relying on EpochsSpectrum.__getitem__ to be robust across MNE
    versions.

    Returns
    ───────
    Dict[event_name → {
        'psd'      : np.ndarray (n_channels, n_freqs) — mean across epochs
        'psd_std'  : np.ndarray (n_channels, n_freqs) — std  across epochs
        'freqs'    : np.ndarray (n_freqs,)
        'ch_names' : List[str]
        'n_epochs' : int
    }]
    """
    psd_all  = spectrum.get_data()   # (n_epochs, n_channels, n_freqs)
    freqs    = spectrum.freqs
    ch_names = list(spectrum.ch_names)

    result: Dict[str, Dict] = {}
    for event_name, event_code in EVENT_TO_CODE.items():
        mask    = epochs.events[:, 2] == event_code
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        psd_event = psd_all[indices]          # (n_event_epochs, n_ch, n_freqs)
        result[event_name] = {
            "psd":      psd_event.mean(axis=0),
            "psd_std":  psd_event.std(axis=0),
            "freqs":    freqs,
            "ch_names": ch_names,
            "n_epochs": len(indices),
        }
        _log(f"  {event_name:<20}  {len(indices):>3} epochs averaged")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Process a single recording
# ═══════════════════════════════════════════════════════════════════════════════

def process_single_recording(
    gdf_file: str,
    output_path: str,
    # epoch window
    tmin: float = -0.5,
    tmax: float = 4.0,
    baseline: Optional[Tuple] = None,
    # artifact rejection
    reject_uv: float = 150.0,
    flat_uv: float = 1.0,
    # preprocessing
    l_freq: float = 0.5,
    h_freq: float = 100.0,
    # PSD
    psd_method: str = "welch",
    fmin: float = 1.0,
    fmax: float = 45.0,
    n_per_seg: Optional[int] = None,
    n_overlap: int = 0,
) -> Dict:
    """
    Full pipeline for one GDF recording.

    load → preprocess → epoch → PSD per epoch → average per event → save

    Saves the per-event *averaged* PSD dict to disk (not raw per-epoch PSDs)
    to keep both disk usage and RAM manageable across 10 recordings per subject.
    After saving, large objects (raw, epochs, spectrum) are explicitly deleted
    to free memory before the next recording is loaded.

    Output file: pickle containing Dict[event_name → {psd, psd_std, freqs, ...}]
    """
    _log(f"\n{'═' * 60}")
    _log(f"Recording : {Path(gdf_file).name}")
    _log(f"{'═' * 60}")

    raw, events, _ = load_raw(gdf_file)
    raw = preprocess_raw(raw, l_freq=l_freq, h_freq=h_freq)
    epochs = create_epochs(
        raw, events,
        tmin=tmin, tmax=tmax,
        baseline=baseline,
        reject_uv=reject_uv, flat_uv=flat_uv,
    )
    spectrum = compute_psd_per_epoch(
        epochs,
        method=psd_method,
        fmin=fmin, fmax=fmax,
        n_per_seg=n_per_seg, n_overlap=n_overlap,
    )
    event_psds = average_psd_per_event(spectrum, epochs)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(event_psds, f)
    _log(f"[saved]   {out}")

    del raw, epochs, spectrum   # free RAM before next recording
    return event_psds


# ═══════════════════════════════════════════════════════════════════════════════
# Steps 3 (loop) + 4 — Subject-level processing and aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def process_subject(
    gdf_files: List[str],
    output_dir: str,
    subject_id: str,
    condition: str,
    **recording_kwargs,
) -> Dict:
    """
    Steps 3 + 4 for one subject under one experimental condition.

    Step 3 — iterates over the (up to 10) recordings sequentially,
              calling process_single_recording() for each and saving
              intermediate per-recording .pkl files. Already-computed
              files are skipped (resume-safe).

    Step 4 — loads all saved per-recording results and computes the
              mean and std of the averaged PSD per event type across
              recordings, producing one aggregate result per subject.

    Directory layout created on disk
    ──────────────────────────────────
    {output_dir}/
    └── {subject_id}/
        └── {condition}/
            ├── recording_01_psd.pkl
            ├── ...
            ├── recording_10_psd.pkl
            └── {subject_id}_{condition}_average_psd.pkl

    Parameters
    ──────────
    gdf_files        : ordered list of GDF paths for this subject/condition
    output_dir       : root directory for all results
    subject_id       : e.g. 'S01'
    condition        : 'motor_execution' | 'motor_imagery'
    **recording_kwargs : forwarded verbatim to process_single_recording()

    Returns
    ───────
    Dict[event_name → {
        'psd_mean'    : (n_channels, n_freqs)  — mean across recordings
        'psd_std'     : (n_channels, n_freqs)  — std  across recordings
        'freqs'       : (n_freqs,)
        'ch_names'    : List[str]
        'n_recordings': int
    }]
    """
    subj_dir = Path(output_dir) / subject_id / condition
    subj_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 3: process each recording (skip if already saved) ──────────────
    recording_paths: List[Path] = []
    for i, gdf_file in enumerate(gdf_files, start=1):
        out = subj_dir / f"recording_{i:02d}_psd.pkl"
        if out.exists():
            _log(f"[skip]    {out.name} — already computed")
        else:
            process_single_recording(gdf_file, str(out), **recording_kwargs)
        recording_paths.append(out)

    # ── Step 4: aggregate across recordings ─────────────────────────────────
    _log(f"\n{'─' * 60}")
    _log(f"[agg]  {subject_id} | {condition} | {len(recording_paths)} recordings")

    all_recordings: List[Dict] = []
    for path in recording_paths:
        with open(path, "rb") as f:
            all_recordings.append(pickle.load(f))

    subject_avg: Dict = {}
    for event_name in EVENT_TO_CODE:
        per_rec = [r[event_name] for r in all_recordings if event_name in r]
        if not per_rec:
            continue
        psds = np.stack([r["psd"] for r in per_rec], axis=0)  # (n_rec, n_ch, n_freqs)
        subject_avg[event_name] = {
            "psd_mean":     psds.mean(axis=0),
            "psd_std":      psds.std(axis=0),
            "freqs":        per_rec[0]["freqs"],
            "ch_names":     per_rec[0]["ch_names"],
            "n_recordings": len(per_rec),
        }
        _log(f"  {event_name:<20}  {len(per_rec):>2} recordings averaged")

    avg_path = subj_dir / f"{subject_id}_{condition}_average_psd.pkl"
    with open(avg_path, "wb") as f:
        pickle.dump(subject_avg, f)
    _log(f"[saved]   {avg_path}")

    return subject_avg


# ═══════════════════════════════════════════════════════════════════════════════
# Utility — Tidy DataFrame for downstream analysis (Steps 5–6)
# ═══════════════════════════════════════════════════════════════════════════════

def subject_avg_to_dataframe(
    subject_avg: Dict,
    subject_id: str,
    condition: str,
) -> pd.DataFrame:
    """
    Convert the subject-level average PSD dict to a tidy (long-format) DataFrame.

    Schema
    ──────
    subject | condition | event | channel | frequency | psd_mean | psd_std

    This long format makes Step 6 comparisons straightforward:

      # relative power per event vs rest, per channel and frequency
      df.groupby(['subject','condition','channel','frequency']).apply(...)

      # pivot for event × condition matrix
      df.pivot_table(values='psd_mean',
                     index=['subject','channel','frequency'],
                     columns=['condition','event'])
    """
    rows = []
    for event_name, data in subject_avg.items():
        psd_mean = data["psd_mean"]   # (n_ch, n_freqs)
        psd_std  = data["psd_std"]
        for ch_idx, ch_name in enumerate(data["ch_names"]):
            for f_idx, freq in enumerate(data["freqs"]):
                rows.append({
                    "subject":   subject_id,
                    "condition": condition,
                    "event":     event_name,
                    "channel":   ch_name,
                    "frequency": round(float(freq), 4),
                    "psd_mean":  float(psd_mean[ch_idx, f_idx]),
                    "psd_std":   float(psd_std[ch_idx, f_idx]),
                })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point — example for one subject, one condition
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    SUBJECT_ID = "S02_ME" # Start at subject 2 because subject 1 is left-handed
    CONDITION  = "motor_execution"   # or "motor_imagery" for Step 5
    OUTPUT_DIR = "./results"

    if "s01" in SUBJECT_ID.lower():
        raise ValueError("Subject 1 is left-handed. Please, skip they for this analysis.")

    SUBJECT_NUM = str(int(SUBJECT_ID.split("_")[0][1:]))  # Extract the subject number from the SUBJECT_ID
    FILE_NAME  = f"{CONDITION.replace('_', '')}_subject{SUBJECT_NUM}"
    # 10 GDF recordings for this subject + condition
    gdf_files = [
            f"../data/{CONDITION}/{SUBJECT_ID}/{FILE_NAME}_run{i:d}.gdf"
            for i in range(2, 11)
        ]

    subject_avg = process_subject(
        gdf_files  = gdf_files,
        output_dir = OUTPUT_DIR,
        subject_id = SUBJECT_ID,
        condition  = CONDITION,
        # epoch window
        tmin       = -0.5,
        tmax       = 4.0,
        baseline   = None,       # no voltage-level baseline; rely on detrend=1
        # artifact rejection
        reject_uv  = 150.0,
        flat_uv    = 1.0,
        # preprocessing
        l_freq     = 0.5,
        h_freq     = 100.0,
        # PSD — Welch, 2 s segments at 256 Hz → Δf = 0.5 Hz, 50 % overlap
        psd_method = "welch",
        fmin       = 1.0,
        fmax       = 45.0,
        n_per_seg  = 512,
        n_overlap  = 256,
    )

    # Export to tidy CSV for Step 6
    df = subject_avg_to_dataframe(subject_avg, SUBJECT_ID, CONDITION)
    csv_path = (
        Path(OUTPUT_DIR) / SUBJECT_ID / CONDITION
        / f"{SUBJECT_ID}_{CONDITION}_average_psd.csv"
    )
    df.to_csv(csv_path, index=False)
    _log(f"\nDataFrame saved → {csv_path}")
    print(df.head(10).to_string(index=False))