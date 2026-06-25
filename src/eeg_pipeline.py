"""
eeg_pipeline.py
───────────────────────────────────────────────────────────────────────────────
EEG Analysis Pipeline

Step 1  load_raw()                  Load GDF, extract events (no pick yet)
Step 1b check_nan()                 Detect/handle NaN values before preprocessing
Step 2  preprocess_raw()            Filter, Artifact removal (ICA), Interpolate bad channels,
                                    Re-reference, pick channels
Step 3  create_epochs()             Segment, detrend
Step 4  compute_epoch_spectra()     Welch PSD (baseline + active) and ERD/ERS per epoch
Step 5  process_single_recording()  Full single-recording pipeline
Step 6  main_load_and_process()      Batch-process all recordings across subjects
───────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

import mne
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.spatial.distance import cdist
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis as sp_kurtosis


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

SPECTRAL_BANDS: Dict[str, Tuple[float, float]] = {
    'alpha': (8.0,  13.0),
    'beta':  (13.0, 30.0),
}

_CONDITION_MAP: Dict[str, str] = {
    'motorexecution': 'ME',
    'motorimagination':   'MI',
}

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
# Step 1 — Load
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

    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='warn')

    _log(
        f"[load]    {Path(gdf_file).name} | "
        f"{len(raw.ch_names)} ch | "
        f"{raw.n_times} samples ({raw.times[-1]:.1f} s) | "
        f"{len(events)} events"
    )
    return raw, events, event_id


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1b — NaN Handling
# ═══════════════════════════════════════════════════════════════════════════════

def check_nan(
    raw: mne.io.Raw,
    epoch_duration: float = 5.0,
    global_ch_fraction: float = 0.25,
    global_duration_fraction: float = 0.10,
    seq_nan_fraction: float = 0.05,
    total_nan_fraction: float = 0.10,
) -> bool:
    """
    Detect and handle NaN values in EEG channels before preprocessing.

    Three-stage strategy
    ────────────────────
    Stage 1 — Global dropout
        A time sample is "global dropout" if ≥ global_ch_fraction of EEG channels
        are simultaneously NaN. Contiguous global dropout samples are annotated as
        BAD_dropout in raw.annotations (MNE will exclude them from ICA and epoching).
        If BAD_dropout occupies ≥ global_duration_fraction of the total recording,
        return True so the caller can skip this file entirely.

    Stage 2 — Per-channel criteria (evaluated only outside BAD_dropout regions)
        Criterion 2 — contiguous NaN run ≥ seq_nan_fraction × epoch_duration
                       → channel added to raw.info['bads']
        Criterion 3 — total NaN samples / valid samples ≥ total_nan_fraction
                       → channel added to raw.info['bads']
        Bad channels will be interpolated later in preprocess_raw().

    Stage 3 — Linear interpolation
        Isolated NaN samples remaining in non-bad channels are filled with
        linear temporal interpolation in place, channel by channel.

    Parameters
    ──────────
    epoch_duration           : expected epoch length in seconds; used to compute the
                               contiguous NaN threshold for criterion 2.
    global_ch_fraction       : fraction of EEG channels simultaneously NaN for a
                               sample to be considered global dropout (default 0.25).
    global_duration_fraction : maximum tolerated fraction of recording time occupied
                               by global dropout before the file is discarded (0.10).
    seq_nan_fraction         : contiguous NaN run threshold as a fraction of
                               epoch_duration (0.05 → 0.25 s = 128 samples at 512 Hz).
    total_nan_fraction       : maximum tolerated fraction of non-dropout NaN samples
                               per channel before it is flagged bad (default 0.10).

    Returns
    ───────
    True  — recording should be discarded; do not continue the pipeline.
    False — recording is usable; raw.annotations and raw.info['bads'] updated
            in-place; short NaN gaps filled by linear interpolation in-place.
    """
    raw.load_data()

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        _log("[nan_check] no EEG channels found — skipping NaN check")
        return False

    data     = raw.get_data(picks=eeg_picks)   # (n_ch, n_times), copy
    ch_names = [raw.ch_names[p] for p in eeg_picks]
    sfreq    = raw.info['sfreq']
    _, n_times = data.shape

    seq_threshold = int(seq_nan_fraction * epoch_duration * sfreq)  # samples

    # ── Stage 1: Global dropout detection ────────────────────────────────────
    nan_mask            = np.isnan(data)                    # (n_ch, n_times)
    frac_nan_per_sample = np.mean(nan_mask, axis=0)         # (n_times,)
    global_dropout      = frac_nan_per_sample >= global_ch_fraction  # bool (n_times,)

    # Group contiguous global dropout samples into annotation segments
    changes    = np.diff(global_dropout.astype(np.int8), prepend=0, append=0)
    seg_starts = np.where(changes == 1)[0]
    seg_ends   = np.where(changes == -1)[0]

    for s, e in zip(seg_starts, seg_ends):
        raw.annotations.append(
            onset       = float(s) / sfreq,
            duration    = float(e - s) / sfreq,
            description = 'BAD_dropout',
        )

    total_dropout_samples = int(np.sum(global_dropout))
    dropout_fraction      = total_dropout_samples / n_times

    if dropout_fraction >= global_duration_fraction:
        _log(
            f"[nan_check] DISCARD — global dropout {dropout_fraction:.1%} "
            f"≥ {global_duration_fraction:.0%} of recording"
        )
        return True

    # ── Stage 2: Per-channel criteria (outside BAD_dropout regions) ───────────
    n_valid_total = n_times - total_dropout_samples   # samples outside BAD_dropout

    new_bads:  List[str] = []
    bad_seq:   List[str] = []
    bad_total: List[str] = []

    for i, ch in enumerate(ch_names):
        nan_outside = nan_mask[i] & ~global_dropout    # NaNs outside global events

        # Criterion 2: contiguous NaN run ≥ seq_threshold
        if np.any(nan_outside):
            ch_changes  = np.diff(nan_outside.astype(np.int8), prepend=0, append=0)
            run_starts  = np.where(ch_changes == 1)[0]
            run_ends    = np.where(ch_changes == -1)[0]
            if np.any((run_ends - run_starts) >= seq_threshold):
                bad_seq.append(ch)
                new_bads.append(ch)
                continue    # criterion 3 is redundant once criterion 2 fires

        # Criterion 3: total NaN fraction
        if n_valid_total > 0 and int(np.sum(nan_outside)) / n_valid_total >= total_nan_fraction:
            bad_total.append(ch)
            new_bads.append(ch)

    raw.info['bads'] = list(set(raw.info['bads'] + new_bads))

    # ── Stage 3: Linear interpolation of short NaN gaps in good channels ──────
    interp_count = 0
    for i, ch in enumerate(ch_names):
        if ch in new_bads:
            continue
        pick_idx = eeg_picks[i]
        nan_idx  = np.where(np.isnan(raw._data[pick_idx]))[0]
        if len(nan_idx) == 0:
            continue
        valid_idx = np.where(~np.isnan(raw._data[pick_idx]))[0]
        if len(valid_idx) < 2:
            continue
        raw._data[pick_idx, nan_idx] = np.interp(
            nan_idx, valid_idx, raw._data[pick_idx, valid_idx]
        )
        interp_count += 1

    _log(
        f"[nan_check] global_dropout={dropout_fraction:.1%} "
        f"({len(seg_starts)} segment(s) → BAD_dropout) | "
        f"bad_seq={bad_seq} | bad_total={bad_total} | "
        f"interpolated={interp_count} ch | "
        f"seq_threshold={seq_threshold} samples ({seq_threshold / sfreq:.3f} s)"
    )
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Preprocess Raw
# ═══════════════════════════════════════════════════════════════════════════════

def detect_bad_channels(
    raw: mne.io.Raw,
    z_threshold: float = 3.0,
    flat_uv: float = 0.5,
    corr_threshold: float = 0.4,
    n_neighbors: int = 4,
) -> List[str]:
    """
    Detect bad EEG channels on the continuous high-pass filtered data.

    Three independent criteria are applied:
    1. Flat / dead channels — std below flat_uv µV (disconnected or saturated electrode)
    2. Noisy channels       — robust Z-score of channel std exceeds z_threshold
                              (median + MAD scaling to avoid influence of bad channels)
    3. Spatial outliers     — Pearson correlation with the mean of the n_neighbors
                              nearest channels falls below corr_threshold

    Requires a montage to be set on raw (for criterion 3).
    Results are written to raw.info['bads'] and returned as a list.
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks=eeg_picks)          # (n_ch, n_times)
    ch_names = [raw.ch_names[p] for p in eeg_picks]

    bad_flat:  List[str] = []
    bad_noisy: List[str] = []
    bad_corr:  List[str] = []

    # ── Criterion 1: Flat / dead channels ────────────────────────────────────
    # Channels with std below 0.5 µV are disconnected, bridged, or saturated.
    ch_std = np.std(data, axis=1, ddof=1)
    bad_flat = [ch_names[i] for i in np.where(ch_std < flat_uv * 1e-6)[0]]

    # ── Criterion 2: Noisy channels (Z-score on std) ──────────────────
    # This uses median + MAD (median absolute deviation). MAD-based scaling (×1.4826)
    # makes the Z-score consistent with std for Gaussian data while remaining
    # robust to the outliers
    median_std = np.median(ch_std)
    mad = np.median(np.abs(ch_std - median_std))
    if mad > 0:
        robust_z = (ch_std - median_std) / (1.4826 * mad)
        bad_noisy = [ch_names[i] for i in np.where(robust_z > z_threshold)[0]]

    # ── Criterion 3: Spatial correlation with nearest neighbours ─────────────
    # A "good" EEG channel should correlates well with its nearest neighbours. A channel
    # that does not is either from a bad electrode or from a very different spatial source
    positions = np.array([raw.info['chs'][p]['loc'][:3] for p in eeg_picks])
    has_position = ~np.all(positions == 0, axis=1)   # exclude channels with no loc

    already_bad = set(bad_flat + bad_noisy)
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)                  # exclude self-distance

    for i, ch in enumerate(ch_names):
        if not has_position[i] or ch in already_bad:
            continue
        # Nearest neighbours: exclude already-flagged and position-less channels
        sorted_idx = np.argsort(dists[i])
        valid_nbrs = [
            j for j in sorted_idx
            if ch_names[j] not in already_bad and has_position[j]
        ][:n_neighbors]
        if len(valid_nbrs) < 2:
            continue
        neighbor_mean = np.mean(data[valid_nbrs], axis=0)
        corr = np.corrcoef(data[i], neighbor_mean)[0, 1]
        if corr < corr_threshold:
            bad_corr.append(ch)

    all_bads = list(set(bad_flat + bad_noisy + bad_corr))
    raw.info['bads'] = all_bads

    _log(
        f"[bad_ch]  flat={bad_flat} | noisy={bad_noisy} | low_corr={bad_corr} "
        f"| total={len(all_bads)} channel(s) marked bad"
    )
    return all_bads


def run_ica(
    raw: mne.io.Raw,
    ica_method: str = 'fastica',
    random_state: int = 42,
    max_iter: int = 1000,
    eog_threshold: float = 3.0,
    n_components: float = 20,
) -> mne.preprocessing.ICA:
    """
    Fit ICA on the high-pass filtered continuous data and automatically remove
    EOG artifact components. Apply the solution to raw in-place.

    Parameters
    ──────────
    ica_method        : 'fastica' or 'infomax'.
                    'infomax' runs with extended=True (extended Infomax), which
                    handles both sub- and super-Gaussian sources and is generally
                    preferred for EEG. 'fastica' is faster and equally common.
    random_state  : integer seed for reproducibility across runs.
    max_iter      : maximum number of ICA iterations.
    eog_threshold : z-score threshold for automatic EOG component detection.
                    Lower values flag more components (higher recall, lower precision).

    Returns the fitted ICA object for post-hoc inspection and manual EMG review.
    EMG artifact components cannot be detected automatically with the available
    toolset and must be identified by visual inspection:
        ica.plot_components()   — topographic maps of all components
        ica.plot_sources(raw)   — time courses and power spectra

    Note
    ────
    n_components is set to n_good_EEG_channels − 1. The −1 accounts for the
    rank reduction already present in the data due to the mastoid hardware
    reference used during recording.
    """
    if n_components is None:
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        n_components = len(eeg_picks) - 1

    # Infomax requires extended=True to handle both sub- and super-Gaussian sources
    fit_params = {'extended': True} if ica_method == 'infomax' else {}

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=ica_method,
        fit_params=fit_params,
        random_state=random_state,
        max_iter=max_iter,
        verbose=False,
    )
    ica.fit(raw, picks='eeg', reject_by_annotation=True, verbose=False)

    # EOG components — correlated with dedicated EOG channels
    eog_indices, _ = ica.find_bads_eog(
        raw,
        ch_name=EOG_CHANNELS,
        threshold=eog_threshold,
        verbose=False,
    )

    ica.exclude = eog_indices
    ica.apply(raw, verbose=False)

    _log(
        f"[ica]     method={ica_method} | n_components={n_components} | "
        f"eog={eog_indices} | excluded={ica.exclude}"
    )
    _log("[ica]     NOTE: inspect EMG components manually — "
         "ica.plot_components() / ica.plot_sources(raw)")
    return ica


def preprocess_raw(
    raw: mne.io.Raw,
    events: np.ndarray,
    l_freq: float = 1.0,
    h_freq: float = 45.0,
    ica_method: str = 'fastica',
    reference: str = "average",
) -> Tuple[mne.io.Raw, np.ndarray]:
    """
    Preprocess the continuous Raw signal before epoching.

    Operation order (order is critical)
    ─────────────────────────────────────
    1. Load into RAM
    2. Remove DC level by subtracting the mean between the first and last movement event
    3. High-pass FIR filter at l_freq to remove slow electrode drift
    4. Bad channel detection
    5. Independent Component Analysis (ICA) for artifact removal  (ica_method)
    6. Low-pass FIR filter at h_freq to remove high-frequency noise and line noise
    7. Interpolate bad channels
    8. Re-reference to Common Average Reference (CAR)
    9. Downsample to 256 Hz — event sample indices are rescaled to match

    Parameters
    ──────────
    ica_method : passed to run_ica(); 'fastica' (default) or 'infomax'

    Returns
    ───────
    raw    : preprocessed Raw object at 256 Hz
    events : event array with sample indices rescaled to 256 Hz
    """
    # Loading data into RAM is necessary for filtering and ICA
    raw.load_data()

    # Removing DC level by subtracting the mean between the first and last movement event
    # to avoid transients in the beginning and end of the recording
    int_time = events[0,0] - int(2 * raw.info['sfreq']) # first sample time index
    end_time = events[-1,0] + int(3 * raw.info['sfreq'])   # last sample time index
    dc_offset = np.mean(raw.get_data()[:, int_time:end_time], axis=1, keepdims=True)
    raw._data -= dc_offset

    # High-pass FIR filter (zero-phase) at l_freq to remove slow electrode drift
    raw.filter(l_freq=l_freq, h_freq=None, method="fir", fir_window="hann", phase="zero", verbose=False)

    # Bad Channel Detection
    detect_bad_channels(raw)

    # Independent Component Analysis (ICA)
    _ = run_ica(raw, method=ica_method)

    # Low-pass FIR filter (zero-phase) at h_freq to remove high-frequency noise and 50 (or 60) Hz line noise
    end_freq = 50 if h_freq <= 45.0 else 10
    raw.filter(l_freq=None, h_freq=h_freq, trans_bandwidth=h_freq-end_freq, method="fir", fir_window="hann", phase="zero", verbose=False)

    # Interpolate bad channels using spherical spline interpolation
    raw.interpolate_bads(reset_bads=True, mode='accurate', origin='auto', method={'eeg': 'spline'}, verbose=False)

    # Re-reference to Common Average Reference (CAR).
    # The right mastoid was the hardware reference and is absent from the recorded
    # channels. add_reference_channels adds it back as a zero-signal channel (voltage
    # relative to itself is always zero), so the CAR average spans all 62 channels
    # (61 EEG + mastoid). The name 'Mastoid_R' is a label only — no signal is recovered.
    raw = mne.add_reference_channels(raw, ref_channels=['Mastoid_R'])
    raw.set_eeg_reference(reference, projection=False, verbose=False)

    # Downsample to 256 Hz (anti-aliasing already provided by the 45 Hz low-pass above).
    # Passing events= causes MNE to rescale sample indices from the original sfreq to 256 Hz
    # and return them alongside the raw; without this the events array is stale.
    raw, events = raw.resample(256.0, events=events, verbose=False)

    _log(
        f"[preproc] ref={reference} | bp={l_freq}–{h_freq} Hz | "
        f"ica={ica_method} | {len(raw.ch_names)} ch remaining"
    )
    return raw, events


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Epoch segmentation
# ═══════════════════════════════════════════════════════════════════════════════

def reject_epochs(
    epochs: mne.Epochs,
    condition: str,
    amp_uv_me: float = 200.0,
    amp_uv_mi: float = 150.0,
    jp_threshold: float = 5.0,
    kurt_threshold: float = 5.0,
    norm_power_me: float = 0.8,
    norm_power_mi: float = 0.65,
) -> None:
    """
    Apply four complementary artifact rejection criteria to epochs in-place.

    Must be called BEFORE epochs.pick() so all EEG channels are available
    for joint probability, kurtosis, and normalized power computation.

    Criteria
    ────────
    1. Amplitude threshold — any sample in any channel exceeds ±amp_uv µV.
                             ME: 200 µV  |  MI: 150 µV
    2. Joint probability   — robust Z-score of per-epoch channel std exceeds
                             jp_threshold × SD. Flags epochs whose variance is
                             statistically improbable given the distribution
                             across all epochs (proxy for EEGLAB joint probability).
    3. Kurtosis            — robust Z-score of per-epoch channel excess kurtosis
                             exceeds kurt_threshold × SD. Flags epochs with
                             heavy-tailed, non-Gaussian amplitude distributions.
    4. Normalized power    — ratio of bandpower(20–40 Hz) / bandpower(4–40 Hz)
                             exceeds norm_threshold in any channel. Detects EMG
                             contamination. ME: 0.8  |  MI: 0.65

    Criteria 2 and 3 use median + MAD-based robust Z-scores (×1.4826) to prevent
    the flagged epochs themselves from inflating the reference statistics.
    All four Butterworth filters are 4th-order zero-phase (sosfiltfilt).

    Parameters
    ──────────
    condition : 'ME' or 'MI' — selects thresholds for criteria 1 and 4.

    Epochs are dropped in-place via epochs.drop(). No return value.
    """
    condition = condition.upper()
    if condition not in ('ME', 'MI'):
        raise ValueError(f"condition must be 'ME' or 'MI', got '{condition}'")

    amp_threshold  = (amp_uv_me  if condition == 'ME' else amp_uv_mi) # value in micro volts
    norm_threshold =  norm_power_me if condition == 'ME' else norm_power_mi

    data = epochs.get_data(picks='eeg')           # (n_epochs, n_ch, n_times)
    n_epochs, _, _ = data.shape
    sfreq = epochs.info['sfreq']
    nyq   = sfreq / 2.0

    # ── Criterion 1: Amplitude threshold ─────────────────────────────────────
    bad_amp = set(
        np.where(np.any(np.abs(data) > amp_threshold, axis=(1, 2)))[0].tolist()
    )

    # ── Criterion 2: Joint probability (robust Z-score of per-epoch std) ─────
    ep_std  = np.std(data, axis=2)                           # (n_epochs, n_ch)
    med_std = np.median(ep_std, axis=0)                      # (n_ch,)
    mad_std = np.median(np.abs(ep_std - med_std), axis=0)    # (n_ch,)
    valid   = mad_std > 0
    z_jp    = np.zeros_like(ep_std)
    z_jp[:, valid] = (ep_std[:, valid] - med_std[valid]) / (1.4826 * mad_std[valid])
    bad_jp  = set(np.where(np.max(np.abs(z_jp), axis=1) > jp_threshold)[0].tolist())

    # ── Criterion 3: Kurtosis (robust Z-score of per-epoch excess kurtosis) ──
    ep_kurt  = sp_kurtosis(data, axis=2, fisher=True)         # (n_epochs, n_ch)
    med_kurt = np.median(ep_kurt, axis=0)
    mad_kurt = np.median(np.abs(ep_kurt - med_kurt), axis=0)
    valid_k  = mad_kurt > 0
    z_kurt   = np.zeros_like(ep_kurt)
    z_kurt[:, valid_k] = (ep_kurt[:, valid_k] - med_kurt[valid_k]) / (1.4826 * mad_kurt[valid_k])
    bad_kurt = set(np.where(np.max(np.abs(z_kurt), axis=1) > kurt_threshold)[0].tolist())

    # ── Criterion 4: Normalized power (EMG index) ────────────────────────────
    sos_emg = butter(4, [20.0 / nyq, 40.0 / nyq], btype='bandpass', output='sos')
    sos_eeg = butter(4, [4.0  / nyq, 40.0 / nyq], btype='bandpass', output='sos')

    pow_emg = np.sum(sosfiltfilt(sos_emg, data, axis=2) ** 2, axis=2)  # (n_epochs, n_ch)
    pow_eeg = np.sum(sosfiltfilt(sos_eeg, data, axis=2) ** 2, axis=2)  # (n_epochs, n_ch)

    with np.errstate(divide='ignore', invalid='ignore'):
        norm_power = np.where(pow_eeg > 0, pow_emg / pow_eeg, 0.0)

    bad_norm = set(
        np.where(np.any(norm_power > norm_threshold, axis=1))[0].tolist()
    )

    # ── Drop all flagged epochs ───────────────────────────────────────────────
    all_bad = sorted(bad_amp | bad_jp | bad_kurt | bad_norm)
    if all_bad:
        epochs.drop(all_bad, reason='CUSTOM_REJECTION', verbose=False)

    _log(
        f"[reject]  cond={condition} | "
        f"amp={len(bad_amp)} | jp={len(bad_jp)} | "
        f"kurt={len(bad_kurt)} | norm_power={len(bad_norm)} | "
        f"dropped={len(all_bad)}/{n_epochs}"
    )

    _log(
        f"[reject]  cond={condition} | "
        f"amp={bad_amp} | jp={bad_jp} | "
        f"kurt={bad_kurt} | norm_power={bad_norm} | "
        f"dropped={all_bad}"
    )


def create_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    condition: str,
    tmin: float = -2.0,
    tmax: float = 3.0,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    flat_uv: float = 1.0,
    verbose: bool = False,
) -> mne.Epochs:
    """
    Segment the preprocessed Raw signal into epochs and apply artifact rejection.

    Parameters
    ──────────
    condition : 'ME' or 'MI' — passed to reject_epochs() to select the
                appropriate amplitude and normalized power thresholds.
    tmin      : seconds before cue onset (negative → pre-stimulus window).
                Default −2.0 captures the fixation cross period (paradigm t=0 s).
    tmax      : seconds after cue onset. Default 3.0 reaches cue offset (t=5 s).
    baseline  : (t_start, t_end) for MNE baseline subtraction on the raw voltage.
                None (default) defers all normalization to the PSD analysis stage.
    flat_uv   : epochs where any channel stays below this peak-to-peak threshold
                (µV) are dropped by MNE. Catches disconnected or saturated channels.
    detrend=1 : linear detrend applied per epoch/channel before spectral analysis.
                Prevents 1/f² leakage from intra-epoch linear drifts.

    Rejection order
    ───────────────
    1. MNE flat detection   (flat_uv, catches dead channels within an epoch)
    2. reject_epochs()      (amplitude, joint probability, kurtosis, EMG index)
    3. epochs.pick()        (reduce to channels of interest after all rejection)
    """
    epochs = mne.Epochs(
        raw,
        events,
        event_id=EVENT_TO_CODE,
        tmin=tmin,
        tmax=tmax,
        picks="eeg",
        baseline=baseline,
        flat={"eeg": flat_uv * 1e-6},
        detrend=1,
        preload=True,
        verbose=verbose,
    )
    epochs.drop_bad(verbose=verbose)

    # Custom rejection with all channels still present - Do not do this for now
    # Is removing more channels that is really needed
    # reject_epochs(epochs, condition)

    # Reduce to channels of interest only after rejection is complete
    epochs.pick(CHANNELS_OF_INTEREST)

    _log(
        f"[epochs]  {len(epochs)} kept | "
        f"shape: {epochs.get_data().shape}  (epochs × ch × samples)"
    )
    return epochs

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 — PSD computation per epoch and averaging per event type
# ═══════════════════════════════════════════════════════════════════════════════
def compute_psd_per_epoch(
    epochs: mne.Epochs,
    method: str = "welch",
    fmin: float = 1.0,
    fmax: float = 45.0,
    tmin: float = -2.0,
    tmax: float = 3.0,
    window: str = "hann",
    verbose: bool = False,
) -> mne.time_frequency.EpochsSpectrum:
    """
    Compute PSD for every epoch via MNE's compute_psd() using the Welch (default) and
    Hanning window (default).

    The result is an EpochsSpectrum with shape (n_epochs, n_channels, n_freqs)
    that preserves the event structure of the original Epochs object.

    Welch parameter guidance
    ──────────────────────────
    Frequency resolution:  Δf = sfreq / n_per_seg
    Number of segments K:  ≈ n_times / (n_per_seg − n_overlap)

    Example, for 256 Hz data, 5.0 s epochs (1280 samples):
      n_per_seg = 512   →  Δf = 0.5 Hz, K ≈ 3 non-overlapping segments
      n_overlap = 256   →  50 % overlap, K ≈ 5 effective segments
                            → variance ∝ 1/(K × n_epochs)

    The overlap here is *within* a single epoch (internal to Welch).
    It is entirely independent of inter-epoch overlap, which must remain
    zero to keep realizations independent for ensemble averaging.

    method='multitaper' is also accepted; in that case n_per_seg and
    n_overlap are ignored and MNE uses Slepian tapers internally.
    """
    sfreq = epochs.info['sfreq']
    n_overlap = int(0.5*sfreq)
    n_per_seg = int(sfreq)
    n_fft = int(2*sfreq)

    spectrum = epochs.compute_psd(
        method=method,
        fmin=fmin,
        fmax=fmax,
        tmin=tmin,
        tmax=tmax,
        n_fft=n_fft,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        window=window,
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


def compute_epoch_spectra(
    epochs: mne.Epochs,
    baseline_tmin: float = -2.0,
    baseline_tmax: float = 0.0,
    active_tmin: float = 0.0,
    active_tmax: float = 3.0,
    fmin: float = 1.0,
    fmax: float = 45.0,
    method: str = "welch",
    window: str = "hann",
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 1 — Per-epoch spectral estimation.

    Calls compute_psd_per_epoch() twice — once per time window — then computes
    the frequency-resolved ERD/ERS for every epoch:

        PSD_baseline(f) : Welch PSD over [baseline_tmin, baseline_tmax]
        PSD_active(f)   : Welch PSD over [active_tmin,   active_tmax]
        ERD/ERS(f)      = 10 × log10( PSD_active(f) / PSD_baseline(f) )

    ERD/ERS sign convention
    ────────────────────────
    Negative (dB) → ERD: power suppressed relative to baseline (desynchronization)
    Positive (dB) → ERS: power enhanced relative to baseline (synchronization)
    NaN           → baseline bin is zero (should not occur with Welch on real EEG)

    Returns
    ───────
    psd_baseline : np.ndarray (n_epochs, n_channels, n_freqs)  µV²/Hz
    psd_active   : np.ndarray (n_epochs, n_channels, n_freqs)  µV²/Hz
    erds_curve   : np.ndarray (n_epochs, n_channels, n_freqs)  dB
    freqs        : np.ndarray (n_freqs,)                       Hz
    """
    spec_baseline = compute_psd_per_epoch(
        epochs, method=method, fmin=fmin, fmax=fmax,
        tmin=baseline_tmin, tmax=baseline_tmax,
        window=window, verbose=verbose,
    )
    spec_active = compute_psd_per_epoch(
        epochs, method=method, fmin=fmin, fmax=fmax,
        tmin=active_tmin, tmax=active_tmax,
        window=window, verbose=verbose,
    )

    psd_baseline = spec_baseline.get_data()   # (n_epochs, n_ch, n_freqs)
    psd_active   = spec_active.get_data()     # (n_epochs, n_ch, n_freqs)
    freqs        = spec_baseline.freqs

    with np.errstate(divide='ignore', invalid='ignore'):
        erds_curve = np.where(
            psd_baseline > 0,
            10.0 * np.log10(psd_active / psd_baseline),
            np.nan,
        )

    _log(
        f"[spectra] baseline {baseline_tmin}–{baseline_tmax} s | "
        f"active {active_tmin}–{active_tmax} s | "
        f"shape {psd_baseline.shape}  (epochs × ch × freqs)"
    )
    return psd_baseline, psd_active, erds_curve, freqs


def _median_freq(
    psd_band: np.ndarray,
    freqs_band: np.ndarray,
    auc: np.ndarray,
) -> np.ndarray:
    """
    Vectorized median frequency: f where cumulative band power reaches 50% of AUC.

    Replaces the serial per-bin Simpson loop with a single cumulative_trapezoid
    pass followed by linear interpolation for sub-bin precision. Operates on all
    (epoch, channel) pairs simultaneously instead of one at a time.

    Parameters
    ──────────
    psd_band  : (n_epochs, n_channels, n_freqs_band)
    freqs_band: (n_freqs_band,)
    auc       : (n_epochs, n_channels)  pre-computed band AUC

    Returns
    ───────
    np.ndarray (n_epochs, n_channels) — median frequency in Hz
    """
    n_e, n_ch, n_f = psd_band.shape
    flat = psd_band.reshape(-1, n_f)        # (N, n_f)
    half = auc.reshape(-1) / 2.0            # (N,)

    # cum[k] = integral from freqs_band[0] to freqs_band[k+1]
    cum = cumulative_trapezoid(flat, freqs_band, axis=-1)    # (N, n_f-1)
    idx = np.argmax(cum >= half[:, np.newaxis], axis=-1)     # (N,) — crossing bin index

    rows     = np.arange(len(flat))
    cum_prev = np.where(idx > 0, cum[rows, np.clip(idx - 1, 0, n_f - 2)], 0.0)
    cum_at   = cum[rows, idx]
    f_lo     = freqs_band[idx]
    f_hi     = freqs_band[np.clip(idx + 1, 0, n_f - 1)]
    denom    = cum_at - cum_prev

    f_med = np.where(
        denom > 0,
        f_lo + (half - cum_prev) / denom * (f_hi - f_lo),
        f_lo,
    )
    return f_med.reshape(n_e, n_ch)


def extract_band_metrics(
    psd_baseline: np.ndarray,
    psd_active: np.ndarray,
    freqs: np.ndarray,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Dict]:
    """
    Stage 2 — Per-epoch band metric extraction.

    For every epoch and channel, computes three spectral metrics from both the
    baseline and active PSD windows, plus the band-level ERD/ERS scalar:

        AUC          : Simpson integral of PSD over the band  [µV²]
        Median freq  : f where cumulative band power = 50% of AUC  [Hz]
        Max power    : peak PSD value within the band  [µV²/Hz]
        ERD/ERS band : 10 × log10(AUC_active / AUC_baseline)  [dB]

    AUC uses Simpson's rule (same method as when the user had previously
    used simpson() for this purpose). Median frequency uses a vectorized
    cumulative trapezoid pass with linear interpolation for sub-bin precision.
    ERD/ERS scalar is derived from the band-integrated powers, not from the
    frequency-resolved ERD/ERS curve mean, because it directly measures the
    relative change in total band energy.

    Parameters
    ──────────
    psd_baseline : (n_epochs, n_channels, n_freqs)   output of compute_epoch_spectra
    psd_active   : (n_epochs, n_channels, n_freqs)   output of compute_epoch_spectra
    freqs        : (n_freqs,)                         frequency axis in Hz
    bands        : band definitions; defaults to SPECTRAL_BANDS (alpha, beta)

    Returns
    ───────
    Dict[band_name → {
        'freqs_band' : np.ndarray (n_freqs_band,)
        'baseline'   : {'auc', 'median_freq', 'max_power'}  each (n_epochs, n_channels)
        'active'     : {'auc', 'median_freq', 'max_power'}  each (n_epochs, n_channels)
        'erds_band'  : np.ndarray (n_epochs, n_channels) in dB; NaN when baseline AUC = 0
    }]
    """
    if bands is None:
        bands = SPECTRAL_BANDS

    result: Dict[str, Dict] = {}
    for band_name, (f_low, f_high) in bands.items():
        mask       = (freqs >= f_low) & (freqs <= f_high)
        freqs_b    = freqs[mask]
        psd_base_b = psd_baseline[..., mask]   # (n_epochs, n_ch, n_freqs_b)
        psd_act_b  = psd_active[..., mask]     # (n_epochs, n_ch, n_freqs_b)

        auc_base = simpson(psd_base_b, x=freqs_b, axis=-1)   # (n_epochs, n_ch)
        auc_act  = simpson(psd_act_b,  x=freqs_b, axis=-1)

        with np.errstate(divide='ignore', invalid='ignore'):
            erds_band = np.where(
                auc_base > 0,
                10.0 * np.log10(auc_act / auc_base),
                np.nan,
            )

        result[band_name] = {
            'freqs_band': freqs_b,
            'baseline': {
                'auc':         auc_base,
                'median_freq': _median_freq(psd_base_b, freqs_b, auc_base),
                'max_power':   np.max(psd_base_b, axis=-1),
            },
            'active': {
                'auc':         auc_act,
                'median_freq': _median_freq(psd_act_b, freqs_b, auc_act),
                'max_power':   np.max(psd_act_b, axis=-1),
            },
            'erds_band': erds_band,
        }

        _log(
            f"[metrics] {band_name} ({f_low:.0f}–{f_high:.0f} Hz) | "
            f"{mask.sum()} bins | "
            f"mean AUC  baseline={auc_base.mean():.3e}  active={auc_act.mean():.3e} µV²"
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5 — Process a single recording
# ═══════════════════════════════════════════════════════════════════════════════

def process_single_recording(
    gdf_file: str,
    condition: str,
    # epoch window
    tmin: float = -2.0,
    tmax: float = 3.0,
    baseline: Optional[Tuple] = None,
    # artifact rejection
    flat_uv: float = 1.0,
    # preprocessing
    l_freq: float = 1.0,
    h_freq: float = 45.0,
    # PSD
    fmin: float = 1.0,
    fmax: float = 45.0,
) -> Optional[Dict]:
    """
    Full pipeline for one GDF recording.

    load → check NaN → preprocess → epoch → Stage 1 spectral estimation → save

    Stage 1 produces three per-epoch arrays:
        psd_baseline : Welch PSD over the pre-cue window [tmin, 0 s]
        psd_active   : Welch PSD over the task window    [0 s, tmax]
        erds_curve   : 10 × log10(psd_active / psd_baseline)  [dB]

    Output pickle
    ─────────────
    {
      'psd_baseline' : np.ndarray (n_epochs, n_channels, n_freqs)
      'psd_active'   : np.ndarray (n_epochs, n_channels, n_freqs)
      'erds_curve'   : np.ndarray (n_epochs, n_channels, n_freqs)
      'freqs'        : np.ndarray (n_freqs,)
      'events'       : np.ndarray (n_epochs, 3)
      'ch_names'     : List[str]
      'condition'    : str  ('ME' or 'MI')
    }

    Returns None if check_nan() recommends discarding the recording.
    Large intermediate objects are deleted after saving to free RAM.
    """
    _log(f"\n{'═' * 60}")
    _log(f"Recording : {Path(gdf_file).name}")
    _log(f"{'═' * 60}")

    raw, events, _ = load_raw(gdf_file)

    if check_nan(raw):
        return None

    raw, events = preprocess_raw(raw, events, l_freq=l_freq, h_freq=h_freq)

    epochs = create_epochs(
        raw, events,
        condition=condition,
        tmin=tmin, tmax=tmax,
        baseline=baseline,
        flat_uv=flat_uv,
    )
    del raw

    psd_baseline, psd_active, erds_curve, freqs = compute_epoch_spectra(
        epochs, fmin=fmin, fmax=fmax,
    )

    metrics = extract_band_metrics(psd_baseline, psd_active, freqs)

    result = {
        # Stage 1: per-epoch spectral arrays
        'psd_baseline': psd_baseline,
        'psd_active':   psd_active,
        'erds_curve':   erds_curve,
        'freqs':        freqs,
        # Stage 2: per-epoch band metrics
        'metrics':      metrics,
        # metadata
        'events':       epochs.events.copy(),
        'ch_names':     list(epochs.ch_names),
        'condition':    condition,
    }

    del epochs, psd_baseline, psd_active, erds_curve
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6 — Batch processing entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main_load_and_process(
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    condition_filter: Optional[str] = None,
    overwrite: bool = False,
    skip: list[str] = None,
    **pipeline_kwargs,
) -> Dict:
    """
    Batch-process all GDF recordings found under data_dir.

    Walks data_dir/motor_execution/ and data_dir/motor_imagery/, calls
    process_single_recording() for each GDF file, and saves the result to a
    mirrored directory tree under output_dir.

    Path mapping (raw → processed)
    ───────────────────────────────
    data/motor_execution/S01_ME/motorexecution_subject1_run1.gdf
    → data_processed/motor_execution/S01_ME/processed_motorexecution_subject1_run1.pkl

    The condition ('ME' or 'MI') is inferred from the containing subdirectory:
        motor_execution/ → 'ME'
        motor_imagery/   → 'MI'

    Parameters
    ──────────
    data_dir         : root raw-data directory.
                       Defaults to <project_root>/data/
    output_dir       : root output directory.
                       Defaults to <project_root>/data_processed/
    condition_filter : 'ME' or 'MI' to restrict the batch to one condition.
                       None (default) processes both conditions.
    overwrite        : if False (default), skip any recording whose output
                       pickle already exists on disk — allows safe resumption.
    skip             : list of substrings identifying paths, folder names, and
                       file names to not process. For example, `skip = ['S01']`
                       excludes any path containing 'S01' anywhere (matches
                       S01_ME, S01_MI, etc.); `skip = ['S01_ME']` is more
                       specific, only excludes that folder;
                       `skip = ['motorexecution_subject1_run1.gdf']`
                       excludes the specific filenames.
    **pipeline_kwargs: forwarded verbatim to process_single_recording();
                       e.g. l_freq=0.5, h_freq=40.0, flat_uv=1.5

    Returns
    ───────
    Summary dict:
        {'processed': int, 'discarded': int, 'skipped': int, 'failed': int}
        processed  — successfully saved to disk
        discarded  — check_nan() flagged the recording for removal
        skipped    — output already existed (overwrite=False) or condition filtered
        failed     — an exception was raised during processing
    """
    project_root = Path(__file__).resolve().parent
    data_root = Path(data_dir) if data_dir else project_root.joinpath("data")
    out_root = Path(output_dir) if output_dir else project_root.joinpath("data_processed")

    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    if condition_filter is not None:
        condition_filter = condition_filter.upper()
        if condition_filter not in ('ME', 'MI'):
            raise ValueError(
                f"condition_filter must be 'ME', 'MI', or None — got '{condition_filter}'"
            )

    # Collect and optionally filter GDF files
    gdf_files = sorted(data_root.rglob("*.gdf"))
    if condition_filter:
        keep_folder = next(f for f, c in _CONDITION_MAP.items() if c == condition_filter)
        gdf_files = [p for p in gdf_files if keep_folder in p.parts]

    if skip:
        gdf_files = [p for p in gdf_files if not any(s.lower() in str(p).lower() for s in skip)]

    n_total = len(gdf_files)
    if n_total == 0:
        _log(f"[main] No GDF files found under {data_root}")
        return {'processed': 0, 'discarded': 0, 'skipped': 0, 'failed': 0}

    _log(f"[main] {n_total} GDF file(s) found | output root → {out_root}")

    summary: Dict[str, int] = {'processed': 0, 'discarded': 0, 'skipped': 0, 'failed': 0}

    for i, gdf_path in enumerate(gdf_files, start=1):

        # Infer condition from directory parts
        condition = next(
            (cond for folder, cond in _CONDITION_MAP.items() if folder in gdf_path.parts),
            None,
        )
        if condition is None:
            _log(f"[main] {i}/{n_total}  SKIP — cannot infer condition: {gdf_path}")
            summary['skipped'] += 1
            continue

        # Mirror the relative path under data_root into out_root
        rel_path = gdf_path.relative_to(data_root)
        out_path = out_root / rel_path.parent / f"processed_{rel_path.stem}.pkl"

        _log(f"[main] {i}/{n_total}  {rel_path}  [{condition}]")

        if not overwrite and out_path.exists():
            _log(f"[main]   already processed — skipping (pass overwrite=True to redo)")
            summary['skipped'] += 1
            continue

        try:
            result = process_single_recording(
                gdf_file=str(gdf_path),
                condition=condition,
                output_path=str(out_path),
                **pipeline_kwargs,
            )
            if result is None:
                _log(f"[main]   DISCARDED — check_nan() flagged this recording")
                summary['discarded'] += 1
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    pickle.dump(result, f)
                _log(f"[saved]   {out_path}")
                summary['processed'] += 1

        except Exception as exc:
            _log(f"[main]   FAILED — {type(exc).__name__}: {exc}")
            summary['failed'] += 1

    _log(
        f"\n[main] Batch complete — "
        f"processed={summary['processed']} | "
        f"discarded={summary['discarded']} | "
        f"skipped={summary['skipped']} | "
        f"failed={summary['failed']}"
    )
    return summary


if __name__ == '__main__':
    main_load_and_process()