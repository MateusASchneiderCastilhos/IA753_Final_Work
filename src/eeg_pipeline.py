"""
eeg_pipeline.py
───────────────────────────────────────────────────────────────────────────────
EEG preprocessing and per-epoch spectral analysis pipeline.

Processing stages
    load_raw()                  Load a GDF recording and extract event annotations.
    check_nan()                 Detect and handle NaN samples before preprocessing.
    preprocess_raw()            DC removal, high/low-pass filtering, bad-channel
                                detection, ICA artifact removal, interpolation,
                                common-average reference, and resampling.
    create_epochs()             Segment into cue-locked epochs and pick channels.
    compute_epoch_spectra()     Welch PSD (baseline + active) and ERD/ERS(f) per epoch.
    extract_band_metrics()      Per-band scalars: AUC, median frequency, band ERD/ERS.
    process_single_recording()  Full single-recording pipeline (load -> metrics).
    main_load_and_process()     Batch-process every recording under a data directory.
───────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

import mne
import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.spatial.distance import cdist


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

CODE_TO_EVENT = {v: k for k, v in EVENT_TO_CODE.items()}

CHANNELS_OF_INTEREST = ["C3", "C1", "Cz", "C2", "C4"]

EOG_CHANNELS = ["eog-l", "eog-m", "eog-r"]

EXCLUDE_CHANNELS = [
    "thumb_near", "thumb_far", "thumb_index", "index_near", "index_far",
    "index_middle", "middle_near", "middle_far", "middle_ring", "ring_near",
    "ring_far", "ring_little", "litte_near", "litte_far", "thumb_palm",
    "wrist_bend", "roll", "pitch", "gesture",
    "handPosX", "handPosY", "handPosZ",
    "elbowPosX", "elbowPosY", "elbowPosZ",
    "ShoulderAdductio", "ShoulderFlexionE", "ShoulderRotation",
    "Elbow", "ProSupination", "Wrist", "GripPressure",
]

SPECTRAL_BANDS = {
    'alpha': (8.0,  13.0),
    'beta':  (13.0, 30.0),
}

_CONDITION_MAP = {
    'motorexecution': 'ME',
    'motorimagination':   'MI',
}

# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _event_converter(event_str: str) -> Optional[int]:
    """
    Map a GDF annotation string to its integer event code.

    Used as the ``event_id`` callback of ``mne.events_from_annotations`` so that
    only the seven task annotations are kept and every other annotation is
    ignored.

    Args:
        event_str (str): raw annotation description read from the GDF file.

    Returns:
        Optional[int]: the integer event code if ``event_str`` is one of the
        seven valid task codes ('1536'–'1542'); otherwise None, which tells MNE
        to ignore that annotation.
    """
    valid = {"1536", "1537", "1538", "1539", "1540", "1541", "1542"}
    return int(event_str) if isinstance(event_str, str) and event_str in valid else None


def _log(msg: str) -> None:
    """
    Print a pipeline progress message to standard output.

    Args:
        msg (str): the message to print.

    Returns:
        None
    """
    print(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw(gdf_file: str) -> Tuple[mne.io.Raw, np.ndarray, Dict]:
    """
    Load a GDF recording and extract its embedded event annotations.

    Non-EEG kinematic channels are dropped on read, the three EOG channels are
    typed as EOG, and the standard 10-05 montage is applied. Channel picking is
    intentionally deferred to preprocess_raw() so that the common-average
    reference can later be computed over all EEG channels.

    Args:
        gdf_file (str): path to the GDF recording to load.

    Returns:
        Tuple[mne.io.Raw, np.ndarray, Dict]: a 3-tuple of
            - raw (mne.io.Raw): the loaded recording (not yet preloaded),
            - events (np.ndarray): event array of shape (n_events, 3),
            - event_id (Dict): mapping from annotation label to integer code.
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
    Detect and handle NaN samples in the EEG channels before preprocessing.

    The strategy has three stages. (1) Global dropout: time samples where at
    least ``global_ch_fraction`` of EEG channels are simultaneously NaN are
    annotated as ``BAD_dropout`` so MNE excludes them from ICA and epoching; if
    such dropout exceeds ``global_duration_fraction`` of the recording, the whole
    file is flagged for discarding. (2) Per-channel criteria (evaluated outside
    dropout regions): a channel is marked bad if it has a contiguous NaN run of at
    least ``seq_nan_fraction`` x ``epoch_duration`` seconds, or if its total NaN
    fraction reaches ``total_nan_fraction``. (3) Interpolation: isolated NaN
    samples left in good channels are filled by linear temporal interpolation.
    Annotations, ``raw.info['bads']``, and interpolated samples are modified in
    place.

    Args:
        raw (mne.io.Raw): the recording to check; loaded into memory in place.
        epoch_duration (float): expected epoch length in seconds, used to derive
            the contiguous-NaN threshold of stage 2. Default 5.0.
        global_ch_fraction (float): fraction of EEG channels that must be
            simultaneously NaN for a sample to count as global dropout.
            Default 0.25.
        global_duration_fraction (float): maximum tolerated fraction of the
            recording occupied by global dropout before the file is discarded.
            Default 0.10.
        seq_nan_fraction (float): contiguous-NaN run threshold as a fraction of
            ``epoch_duration``. Default 0.05.
        total_nan_fraction (float): maximum tolerated fraction of NaN samples per
            channel (outside dropout) before the channel is flagged bad.
            Default 0.10.

    Returns:
        bool: True if the recording should be discarded (global dropout too
        large); False if it is usable, in which case annotations, bad channels,
        and short-gap interpolation have been applied in place.
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
        f"({len(seg_starts)} segment(s) -> BAD_dropout) | "
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
    Detect bad EEG channels on the continuous, high-pass filtered data.

    Three independent criteria are applied: (1) flat/dead channels whose standard
    deviation is below ``flat_uv`` microvolts; (2) noisy channels whose robust
    (median + MAD) z-score of the channel standard deviation exceeds
    ``z_threshold``; and (3) spatial outliers whose Pearson correlation with the
    mean of their ``n_neighbors`` nearest channels falls below ``corr_threshold``.
    A montage must be set on ``raw`` for criterion 3. The detected channels are
    written to ``raw.info['bads']`` in place and also returned.

    Args:
        raw (mne.io.Raw): continuous recording with a montage set; typically
            already high-pass filtered.
        z_threshold (float): robust z-score threshold for the noisy-channel
            criterion. Default 3.0.
        flat_uv (float): standard-deviation threshold in microvolts below which a
            channel is considered flat/dead. Default 0.5.
        corr_threshold (float): minimum Pearson correlation with the neighbour
            mean below which a channel is a spatial outlier. Default 0.4.
        n_neighbors (int): number of nearest neighbours used for the spatial
            correlation criterion. Default 4.

    Returns:
        List[str]: names of the channels flagged as bad (also stored in
        ``raw.info['bads']``).
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
    n_components: float = 32,
) -> mne.preprocessing.ICA:
    """
    Fit ICA on the continuous data and automatically remove EOG components.

    The ICA is fitted on the (high-pass filtered) EEG channels, the components
    correlated with the dedicated EOG channels are identified with
    ``find_bads_eog`` and rejected, and the cleaned solution is applied to ``raw``
    in place. If ``n_components`` is None or exceeds the number of good EEG
    channels, it is capped at ``n_good_EEG_channels - 1`` to respect the rank
    reduction introduced by the mastoid hardware reference. EMG components are not
    removed automatically and must be inspected manually via
    ``ica.plot_components()`` / ``ica.plot_sources(raw)``.

    Args:
        raw (mne.io.Raw): continuous, preloaded, high-pass filtered recording;
            modified in place by ``ica.apply``.
        ica_method (str): ICA algorithm, 'fastica' (default) or 'infomax'
            (extended Infomax when selected).
        random_state (int): seed for reproducible decompositions. Default 42.
        max_iter (int): maximum number of ICA iterations. Default 1000.
        eog_threshold (float): z-score threshold for automatic EOG-component
            detection; lower values flag more components. Default 3.0.
        n_components (float): number of ICA components to fit; capped at
            ``n_good_EEG_channels - 1`` when None or too large. Default 32.

    Returns:
        mne.preprocessing.ICA: the fitted ICA object (with ``exclude`` set to the
        rejected EOG components) for post-hoc inspection and manual EMG review.
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    n_eeg_picks = len(eeg_picks)
    if n_components is None or n_components >= n_eeg_picks:
        n_components = n_eeg_picks - 1

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
    Preprocess the continuous recording before epoching.

    The operations are applied in this (order-critical) sequence: load into
    memory; remove the DC level by subtracting the mean over the task span;
    high-pass FIR filter at ``l_freq`` to remove slow drift; detect bad channels;
    run ICA to remove EOG artifacts; low-pass FIR filter at ``h_freq`` (its
    stop-band suppresses line noise); interpolate the bad channels by spherical
    spline; re-reference to a common average reference after re-adding the mastoid
    hardware reference as a zero channel; and downsample to 256 Hz (rescaling the
    event sample indices accordingly).

    Args:
        raw (mne.io.Raw): continuous recording from ``load_raw``; modified and
            returned.
        events (np.ndarray): event array of shape (n_events, 3) at the original
            sampling rate; used to bound the DC-removal window and rescaled to the
            new rate.
        l_freq (float): high-pass cut-off in Hz (also the transition bandwidth).
            Default 1.0.
        h_freq (float): low-pass cut-off in Hz. Default 45.0.
        ica_method (str): ICA algorithm forwarded to ``run_ica``, 'fastica'
            (default) or 'infomax'.
        reference (str): reference passed to ``set_eeg_reference``. Default
            "average" (common average reference).

    Returns:
        Tuple[mne.io.Raw, np.ndarray]: a 2-tuple of
            - raw (mne.io.Raw): the preprocessed recording resampled to 256 Hz,
            - events (np.ndarray): the event array with sample indices rescaled
              to 256 Hz.
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
    raw.filter(l_freq=l_freq, h_freq=None, l_trans_bandwidth=l_freq, method="fir", fir_window="hann", phase="zero", verbose=False)

    # Bad Channel Detection
    detect_bad_channels(raw)

    # Independent Component Analysis (ICA)
    _ = run_ica(raw,ica_method=ica_method)

    # Low-pass FIR filter (zero-phase) at h_freq to remove high-frequency noise and 50 (or 60) Hz line noise
    end_freq = 50 - h_freq if h_freq <= 45.0 else 10
    raw.filter(l_freq=None, h_freq=h_freq, h_trans_bandwidth=end_freq, method="fir", fir_window="hann", phase="zero", verbose=False)

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
    Segment the preprocessed continuous signal into cue-locked epochs, drop flat
    epochs, and reduce the data to the channels of interest.

    A per-epoch linear detrend (detrend=1) is applied to remove intra-epoch
    linear drifts before spectral analysis. MNE flat detection then drops epochs
    in which any channel is effectively dead, and the retained epochs are
    restricted to CHANNELS_OF_INTEREST.

    Args:
        raw (mne.io.Raw): preprocessed continuous recording (already filtered,
            re-referenced, and resampled).
        events (np.ndarray): event array of shape (n_events, 3) whose sample
            indices match the sampling rate of ``raw``.
        condition (str): recording condition label, 'ME' or 'MI'. Retained for
            interface consistency and potential per-condition epoch rejection;
            it is not used internally in the current pipeline.
        tmin (float): epoch start in seconds relative to the cue (negative =
            pre-cue). Default -2.0 captures the fixation/baseline window.
        tmax (float): epoch end in seconds relative to the cue. Default 3.0
            reaches the end of the task window.
        baseline (Optional[Tuple[Optional[float], Optional[float]]]): interval
            for MNE baseline correction of the raw voltage. None (default) defers
            all normalization to the spectral analysis stage.
        flat_uv (float): peak-to-peak threshold in microvolts; epochs in which
            any channel stays below it are dropped. Default 1.0.
        verbose (bool): forwarded to MNE for logging verbosity. Default False.

    Returns:
        mne.Epochs: the retained epochs, restricted to CHANNELS_OF_INTEREST.
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

    # Reduce to channels of interest only after epoch rejection is complete
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
    Estimate the PSD of every epoch over a chosen time window.

    Wraps ``Epochs.compute_psd`` with Welch settings derived from the sampling
    rate: a 1-second Hann segment (``n_per_seg = sfreq``), 50% overlap
    (``n_overlap = sfreq/2``), and an FFT length of ``2 x sfreq`` (zero-padding
    that halves the frequency grid spacing). The returned spectrum preserves the
    event structure of the input epochs. The Welch overlap here is internal to a
    single epoch and is independent of inter-epoch spacing.

    Args:
        epochs (mne.Epochs): the epochs to analyze.
        method (str): spectral estimator, 'welch' (default) or 'multitaper'
            (which ignores the Welch segment settings).
        fmin (float): lower frequency bound in Hz. Default 1.0.
        fmax (float): upper frequency bound in Hz. Default 45.0.
        tmin (float): start of the analysis window in seconds relative to the
            cue. Default -2.0.
        tmax (float): end of the analysis window in seconds relative to the cue.
            Default 3.0.
        window (str): taper applied to each Welch segment. Default "hann".
        verbose (bool): forwarded to MNE for logging verbosity. Default False.

    Returns:
        mne.time_frequency.EpochsSpectrum: per-epoch spectrum of shape
        (n_epochs, n_channels, n_freqs).
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
        f"Espectral Resolution = {freqs[1] - freqs[0]:.3f} Hz | "
        f"range: {freqs[0]:.1f}–{freqs[-1]:.1f} Hz"
    )
    return spectrum


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
    Compute the per-epoch baseline and active PSDs and the ERD/ERS spectrum.

    ``compute_psd_per_epoch`` is called twice, once over the baseline window and
    once over the active window, and the frequency-resolved ERD/ERS is obtained
    per epoch as ``10 * log10(PSD_active(f) / PSD_baseline(f))``. By convention a
    negative value is ERD (power suppressed relative to baseline) and a positive
    value is ERS; bins with a zero baseline yield NaN.

    Args:
        epochs (mne.Epochs): the epochs to analyze.
        baseline_tmin (float): start of the baseline window in seconds. Default -2.0.
        baseline_tmax (float): end of the baseline window in seconds. Default 0.0.
        active_tmin (float): start of the active window in seconds. Default 0.0.
        active_tmax (float): end of the active window in seconds. Default 3.0.
        fmin (float): lower frequency bound in Hz. Default 1.0.
        fmax (float): upper frequency bound in Hz. Default 45.0.
        method (str): spectral estimator forwarded to ``compute_psd_per_epoch``.
            Default "welch".
        window (str): taper forwarded to ``compute_psd_per_epoch``. Default "hann".
        verbose (bool): forwarded to MNE for logging verbosity. Default False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: a 4-tuple of
            - psd_baseline (np.ndarray): (n_epochs, n_channels, n_freqs) in µV²/Hz,
            - psd_active (np.ndarray): (n_epochs, n_channels, n_freqs) in µV²/Hz,
            - erds_curve (np.ndarray): (n_epochs, n_channels, n_freqs) in dB,
            - freqs (np.ndarray): (n_freqs,) frequency axis in Hz.
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
    Compute the median frequency of a band for all (epoch, channel) pairs.

    The median frequency is the frequency at which the cumulative band power
    reaches 50% of the band AUC. It is obtained with a single
    ``cumulative_trapezoid`` pass followed by linear interpolation for sub-bin
    precision, vectorized over all epochs and channels at once.

    Args:
        psd_band (np.ndarray): band-limited PSD of shape
            (n_epochs, n_channels, n_freqs_band).
        freqs_band (np.ndarray): frequency axis of the band, shape
            (n_freqs_band,).
        auc (np.ndarray): pre-computed band AUC of shape (n_epochs, n_channels).

    Returns:
        np.ndarray: median frequency in Hz, shape (n_epochs, n_channels).
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
    Extract the per-epoch, per-band scalar descriptors from the PSD windows.

    For every epoch and channel, and for each band, two descriptors are computed
    from both the baseline and active PSDs, together with the band-level ERD/ERS
    scalar:
        - AUC: Simpson integral of the PSD over the band [µV²];
        - median frequency: the frequency at 50% of the band AUC [Hz];
        - band ERD/ERS: 10 * log10(AUC_active / AUC_baseline) [dB].
    The band ERD/ERS scalar is derived from the band-integrated powers (not from
    the mean of the ERD/ERS spectrum over the band), so it measures the relative
    change in total band energy.

    Args:
        psd_baseline (np.ndarray): baseline PSD of shape
            (n_epochs, n_channels, n_freqs), from ``compute_epoch_spectra``.
        psd_active (np.ndarray): active PSD of shape
            (n_epochs, n_channels, n_freqs), from ``compute_epoch_spectra``.
        freqs (np.ndarray): frequency axis in Hz, shape (n_freqs,).
        bands (Optional[Dict[str, Tuple[float, float]]]): band definitions as
            name -> (f_low, f_high); defaults to SPECTRAL_BANDS (alpha, beta).

    Returns:
        Dict[str, Dict]: one entry per band name, each a dict with
            - 'freqs_band' (np.ndarray): band frequency axis, (n_freqs_band,);
            - 'baseline' (Dict): {'auc', 'median_freq'}, each (n_epochs, n_channels);
            - 'active' (Dict): {'auc', 'median_freq'}, each (n_epochs, n_channels);
            - 'erds_band' (np.ndarray): band ERD/ERS in dB, (n_epochs, n_channels),
              NaN where the baseline AUC is zero.
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
            },
            'active': {
                'auc':         auc_act,
                'median_freq': _median_freq(psd_act_b, freqs_b, auc_act),
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
    Run the full pipeline for a single GDF recording.

    The stages are: load the recording, handle NaNs, preprocess the continuous
    signal, epoch it, compute the per-epoch baseline/active PSDs and ERD/ERS
    spectrum, and extract the per-band scalar descriptors. The function returns a
    result dict (which the caller saves to disk), or None if ``check_nan``
    recommends discarding the recording. Large intermediate objects are released
    before returning.

    Args:
        gdf_file (str): path to the GDF recording.
        condition (str): recording condition label, 'ME' or 'MI'.
        tmin (float): epoch start in seconds relative to the cue. Default -2.0.
        tmax (float): epoch end in seconds relative to the cue. Default 3.0.
        baseline (Optional[Tuple]): MNE baseline-correction interval for epoching;
            None (default) defers normalization to the spectral stage.
        flat_uv (float): flat-epoch peak-to-peak threshold in microvolts.
            Default 1.0.
        l_freq (float): high-pass cut-off in Hz. Default 1.0.
        h_freq (float): low-pass cut-off in Hz. Default 45.0.
        fmin (float): lower PSD frequency bound in Hz. Default 1.0.
        fmax (float): upper PSD frequency bound in Hz. Default 45.0.

    Returns:
        Optional[Dict]: None if the recording is discarded; otherwise a dict with
            - 'psd_baseline', 'psd_active', 'erds_curve' (np.ndarray): each of
              shape (n_epochs, n_channels, n_freqs);
            - 'freqs' (np.ndarray): frequency axis, (n_freqs,);
            - 'metrics' (Dict): per-band scalar descriptors from
              ``extract_band_metrics``;
            - 'events' (np.ndarray): event array, (n_epochs, 3);
            - 'ch_names' (List[str]): analyzed channel names;
            - 'condition' (str): 'ME' or 'MI'.
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
    Batch-process every GDF recording found under a data directory.

    Recursively finds all ``*.gdf`` files under ``data_dir``, infers each
    recording's condition from its parent folder (``motor_execution`` -> 'ME',
    ``motor_imagery`` -> 'MI'), runs ``process_single_recording`` on each, and
    saves the result as a pickle in a mirrored tree under ``output_dir`` (e.g.
    ``.../S02_ME/motorexecution_subject2_run1.gdf`` becomes
    ``.../S02_ME/processed_motorexecution_subject2_run1.pkl``).

    Args:
        data_dir (Optional[str]): root directory of raw recordings; defaults to
            ``<project_root>/data``.
        output_dir (Optional[str]): root directory for outputs; defaults to
            ``<project_root>/data_processed``.
        condition_filter (Optional[str]): 'ME' or 'MI' to restrict the batch to
            one condition; None (default) processes both.
        overwrite (bool): if False (default), recordings whose output pickle
            already exists are skipped, allowing safe resumption.
        skip (list[str]): substrings; any GDF path containing one of them
            (case-insensitive) is skipped, matching folder or file names
            (e.g. ['S01'] or ['motorexecution_subject1_run1.gdf']). Default None.
        **pipeline_kwargs: forwarded verbatim to ``process_single_recording``
            (e.g. ``l_freq``, ``h_freq``, ``flat_uv``).

    Returns:
        Dict: a summary with integer counts
        {'processed', 'discarded', 'skipped', 'failed'}, where 'processed' were
        saved to disk, 'discarded' were flagged by ``check_nan``, 'skipped' were
        condition-filtered or already existed, and 'failed' raised an exception.
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
        gdf_files = [p for p in gdf_files if keep_folder in str(p)]

    if skip:
        gdf_files = [p for p in gdf_files if not any(s.lower() in str(p).lower() for s in skip)]

    n_total = len(gdf_files)
    if n_total == 0:
        _log(f"[main] No GDF files found under {data_root}")
        return {'processed': 0, 'discarded': 0, 'skipped': 0, 'failed': 0}

    _log(f"[main] {n_total} GDF file(s) found | output root -> {out_root}")

    summary: Dict[str, int] = {'processed': 0, 'discarded': 0, 'skipped': 0, 'failed': 0}

    for i, gdf_path in enumerate(gdf_files, start=1):

        # Infer condition from directory parts
        condition = next(
            (cond for folder, cond in _CONDITION_MAP.items() if folder in str(gdf_path)),
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
