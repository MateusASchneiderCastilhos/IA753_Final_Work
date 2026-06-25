## Recommended Preprocessing Pipeline

### Critical Issue with Your Proposed Order

Before detailing each step, your proposed order has a fundamental problem: re-referencing before filtering and ICA is wrong. If large DC offsets, slow drifts, or high-amplitude artifacts are present in any channel, computing the common average reference on contaminated data spreads those artifacts to every channel — including your channels of interest. The correct order is:

Filter → Artifact removal (ICA) → Interpolate bad channels → Re-reference → Epoch → Detrend

___

### Complete Recommended Pipeline

___

### Step 1 — Load Data and Prepare

In MNE, read_raw_gdf() reads GDF natively including event markers. At this stage:

Assign the standard 10-10 electrode montage (g.tec systems use this layout)
Explicitly mark channels 62, 63, 64 as EOG type — MNE uses channel types to guide ICA artifact detection
Note the original reference (right mastoid) for later

___

#### Step 2 — DC Offset Removal

Do this before any filtering. g.tec amplifiers with 0.01 Hz analog cutoff can leave non-trivial DC offsets in the digitized signal. Large DC offsets cause Gibbs ringing artifacts at epoch boundaries when you later filter and epoch the data, and they destabilize ICA training.

In MNE: subtract the channel mean across the entire continuous recording per channel. This is a simple, safe operation with no frequency-domain consequences — it removes only the zero-frequency component.

___

#### Step 3 — High-Pass Filter at 1 Hz (Continuous Data)

**Apply to continuous data, before epoching and before ICA.**

**Recommended cutoff: 1 Hz, zero-phase FIR (Hamming window, MNE default)**

- Why 1 Hz and not 0.5 Hz: 0.5 Hz leaves slow physiological drifts (galvanic skin response, slow movement artifacts) that will destabilize ICA. 1 Hz removes these reliably.
- Why 1 Hz and not 4 Hz: 4 Hz is unnecessarily aggressive. Your lowest frequency of interest is 8 Hz (alpha), which is 8× the 1 Hz cutoff — more than enough transition band margin. There is no benefit to cutting at 4 Hz.
- Why before ICA: This is a hard requirement. ICA decomposes the data based on a covariance matrix estimated from the signal. Slow drifts and DC offsets inflate the covariance structure and produce unstable, non-physiological independent components. The standard recommendation in the field (Winkler et al. 2015, MNE documentation) is a minimum 1 Hz high-pass before ICA.
- Why on continuous data: MNE's default FIR filter at 1 Hz has a filter length of approximately 3.3 × sfreq / cutoff = 3.3 × 512 / 1 ≈ 1700 samples (~3.3 seconds). This is longer than your 5-second epochs. Filtering on already-segmented epochs would produce severe edge artifacts. Always filter continuous data first, then epoch.

___

#### Step 4 — Bad Channel Detection

Identify bad channels on the filtered continuous data. Two complementary approaches:

**Automated detection (recommended as first pass):**

pyprep.NoisyChannels (works with MNE): detects channels based on deviation from median amplitude, correlation with neighboring channels, high-frequency noise ratio, and RANSAC (random sample consensus spatial interpolation). This is the most robust automated method available.
Alternatively: mne.preprocessing.find_bad_channels_maxwell() if you have a MaxFilter-compatible system (less relevant for g.tec).

**Visual inspection (mandatory second pass):**

Always review the PSD plot of all channels to spot channels with elevated broadband noise
Review the raw timeseries for channels with flat lines, excessive amplitude, or sustained artifact bursts

**Your interpolation strategy is correct but needs one refinement:**

If a channel of interest (C3, C1, Cz, C2, C4) is bad → do not interpolate yet — mark it as bad, complete ICA, then interpolate in Step 7
If any other channel is bad → mark as bad; it will be excluded from ICA and later CAR computation

**Important**: the number of bad channels directly affects ICA quality. If more than ~10% of channels are bad (>6 channels), ICA results become unreliable. In that case, session-level quality should be flagged.

___

#### Step 5 — Independent Component Analysis (ICA)

This is the most critical preprocessing step for your study. Let me be thorough.

Setup:

Method: fastica (default in MNE) or infomax — both are appropriate; infomax is slightly more theoretically motivated for biomedical signals but slower
n_components: set equal to the rank of the data, which is n_good_channels - 1 (because re-referencing to a single reference reduces rank by 1). For your case: approximately 55–58 components depending on bad channels found
Fit ICA on the continuous high-pass filtered (1 Hz) data
Artifact identification — four categories to address:

EOG (blinking and eye movements):

You have dedicated EOG channels (62, 63, 64) — use them
ica.find_bads_eog(raw, ch_name=['eog_ch_name']) in MNE correlates ICA components against EOG channels
Typically identifies 1–3 components; verify topographically (frontal distribution, bilateral)
This is the most reliable automated identification in MNE
ECG (cardiac artifact):

No dedicated ECG channel was mentioned — use ica.find_bads_ecg(raw) which searches for components with QRS-like temporal structure
Cardiac artifact in central channels is less common but can appear; verify with component time course (regular ~1 Hz modulation)
EMG and movement artifacts (ME condition — your critical challenge):

No fully automated method is as reliable as visual inspection here
EMG components have characteristic features: flat or rising power spectrum (broadband, not 1/f), peripheral topography (near the edges of the cap, around temporal/frontal areas where neck and jaw muscles project), and bursty, irregular time course
Movement artifacts (cable movement, electrode pops) appear as: large-amplitude isolated events in the time course, broadband topography
For your ME condition specifically: upper limb muscle EMG has a limited footprint on the scalp (the limbs are physically distant from electrodes), but high-amplitude broadband noise can still project to central areas. Inspect components with significant power in 20–45 Hz range and non-physiological topographies
Recommendation: be conservative — only remove components you are confident are artifactual. Removing genuine neural components is worse than leaving weak EMG residuals when your analysis is band-limited to 8–30 Hz
Residual line noise (50 Hz):

Your analog notch already attenuated 50 Hz. After the digital 45 Hz low-pass (Step 6 below), this is a non-issue. You can skip manual ICA-based line noise removal.
Step 6 — Low-Pass Filter at 45 Hz
Apply after ICA, on continuous data.

Recommended cutoff: 45 Hz, zero-phase FIR

Why 45 Hz and not 40 Hz: Your beta band ends at 30 Hz. A 40 Hz cutoff leaves only 10 Hz of transition margin above beta. A 45 Hz cutoff gives 15 Hz of margin, ensuring zero distortion at 30 Hz. Given that 50 Hz is already notch-filtered analogically, there is no reason to be conservative with the high cutoff.
Why after ICA: ICA was fitted on high-pass filtered data (1 Hz). The low-pass is applied to the ICA-corrected signal. Applying low-pass before ICA would be redundant since ICA doesn't benefit from the high-frequency attenuation.
Combined effect: your final data will be bandpass-filtered 1–45 Hz via two sequential FIR steps, which is mathematically equivalent to a single 1–45 Hz bandpass filter.
Note on the analog pre-filtering: the analog 0.01–200 Hz filter and 50 Hz notch are hardware-applied and cannot be changed. They do not conflict with your digital processing — they simply mean there is no aliased content above 200 Hz and no 50 Hz line artifact remaining. Your digital filtering operates on this already-conditioned signal.

Step 7 — Interpolate Bad Channels
After ICA, before re-referencing.

Using spherical spline interpolation (MNE default: raw.interpolate_bads()). This reconstructs the bad channel's signal from the spatial distribution of surrounding good channels.

Why after ICA: ICA operated without the bad channels (they were excluded). Interpolating now gives you a full-rank channel set before computing the common average reference — if you interpolate before ICA, the interpolated (synthetic) signal would contaminate ICA training.

Why before re-referencing: the common average reference averages across all channels. A missing channel means the average is biased. Interpolating first gives the CAR computation the full spatial complement.

For your channels of interest: if C3, C1, Cz, C2, or C4 were marked bad, they are reconstructed here from neighboring channels.

Step 8 — Re-reference to Common Average
Add the reference electrode back first. In MNE:


raw = mne.add_reference_channels(raw, ref_channels=['Mastoid_R'])
raw.set_eeg_reference(ref_channels='average', projection=False)
Adding the mastoid reference channel back before computing CAR ensures that the average is computed over the true full spatial complement (all 62 EEG channels, including the original reference). Omitting this step biases the CAR computation because the reference channel's signal is implicitly zero — which it is not.

Step 9 — Downsampling to 256 Hz
My firm recommendation: do it. Your apprehension is not warranted here.

After the 45 Hz low-pass filter, the signal contains no frequency content above 45 Hz. The Nyquist frequency for 256 Hz sampling is 128 Hz — far above 45 Hz. There is zero information loss for your analysis (8–30 Hz). The low-pass filter already served as the anti-aliasing filter.

Practical benefits: file sizes halve, all subsequent computations (ICA — already done, epoching, PSD) run on half the data volume. For 14 subjects with two conditions of 420 trials each, this is a meaningful reduction.

In MNE: raw.resample(256) — MNE automatically applies an anti-aliasing filter before resampling.

Step 10 — Epoch Segmentation
Extract epochs aligned to the fixation cross onset (t=0s):


epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=5.0,
                    baseline=None, preload=True, detrend=None)
Set baseline=None at this stage — apply normalization during PSD computation, not here
Keep all 7 event types (6 movements + rest) with their respective event IDs from the GDF markers
preload=True loads all epochs into memory for the subsequent steps
Step 11 — Bad Epoch Rejection
After epoching, inspect for residual artifacts that ICA did not remove (e.g., large-amplitude electrode pops, head movements):

Automated approach — Autoreject (autoreject Python library, Jas et al. 2017):

Learns a rejection threshold per channel per subject using cross-validation
More principled than a fixed global threshold (e.g., ±100 µV)
Produces a rejection log that you can report (percentage of trials rejected per subject)
Manual fallback: if Autoreject is not available, use amplitude-based rejection: peak-to-peak amplitude > 150 µV across EEG channels is a conservative but standard threshold.

Important: track how many trials are rejected per subject and per class. If rejection rates are high (>20% of trials), the subject's data quality should be flagged. Unequal rejection rates across classes could bias your between-class comparisons.

Step 12 — Linear Detrending per Epoch
Apply after epoching, before PSD computation:


epochs.apply_function(scipy.signal.detrend, type='linear')
Or equivalently, mne.Epochs(..., detrend=1) at the epoching step.

With proper 1 Hz high-pass filtering already applied, residual linear trends within a 5-second epoch are minimal. This step is a safety net — it ensures no epoch-level linear slope (due to slow non-stationarities that survived filtering) biases the PSD computation. Specifically important for the within-trial baseline normalization: you want the baseline window (t=0–2s) and active window (t=2–5s) to have comparable low-frequency behavior.

Complete Pipeline Summary
Order	Step	Key Parameters	Rationale
1	Load + prepare	Montage, channel types	Foundation
2	DC removal	Subtract per-channel mean	Prevents ringing in filtering
3	High-pass filter	1 Hz, zero-phase FIR, continuous	ICA stability, drift removal
4	Bad channel detection	PyPREP + visual inspection	Before spatial operations
5	ICA	~55–58 components	EOG, ECG, EMG removal
6	Low-pass filter	45 Hz, zero-phase FIR, continuous	Band limiting, anti-aliasing
7	Interpolate bad channels	Spherical spline	Restore full spatial complement
8	Re-reference	Common average (add mastoid back first)	Remove mastoid asymmetry
9	Downsample	256 Hz	Halves data, no information loss
10	Epoch segmentation	t=0–5s, baseline=None	Aligned to fixation cross
11	Bad epoch rejection	Autoreject	Remove residual artifacts
12	Linear detrend	Per epoch	Remove epoch-level slope
Three Points Specific to Your Study
1. Apply the exact same pipeline to ME and MI independently. Do not share ICA solutions between conditions — they were recorded in separate sessions on potentially different days. Each session gets its own bad channel detection, ICA, and rejection log.

2. Report rejection statistics. For each subject and condition, report the number of trials rejected per class. Reviewers will ask. Aim for <15% rejection; flag subjects above 25%.

3. ME beta-band interpretation after ICA. Even after ICA, some residual EMG energy in the 20–30 Hz range may remain in the ME condition. When comparing ME vs. MI in upper beta (25–30 Hz), note this limitation explicitly. If ME beta power is substantially higher than MI beta power (beyond what is physiologically expected), suspect residual EMG rather than genuine neural differences.