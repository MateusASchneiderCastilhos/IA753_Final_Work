# Methods

## Participants

This study used a publicly available EEG dataset (Ofner et al., 2017) accessible through the BNCI Horizon 2020 database. Fifteen healthy right-handed subjects (9 female, aged 22–40 years, mean age 27 $\pm$ 5 years) without known neurological disorders participated in the original data collection. But only the 14 right-handed subjects were considered. Written informed consent was obtained from all participants, and the study was conducted in accordance with the protocol approved by the Ethics Committee of the Medical University of Graz (approval number 28–108 ex 15/16).

## Experimental Protocol

Each subject participated in two recording sessions on separate days, with a maximum interval of one week between sessions. The first session comprised motor execution (ME) and the second comprised motor imagery (MI). Immediately before the MI session, subjects performed one additional ME run to reinforce kinesthetic motor imagery.

Both sessions involved the same six movement types performed with the right upper limb: elbow flexion, elbow extension, forearm supination, forearm pronation, hand open, and hand close. A rest class was additionally recorded, resulting in seven classes in total. During ME, subjects performed sustained physical movements. During MI, subjects performed kinesthetic imagery of the same movements executed in the ME session. Subjects were explicitly instructed to avoid any physical movement during MI.

A trial-based paradigm was employed with visual cues displayed on a computer screen. Each trial began at $t = 0$ s with the appearance of a fixation cross, which subjects were instructed to fixate. At $t = 2$ s, a cue indicating the required task appeared on screen. At $t = 5$ s, the cue disappeared and subjects returned to the starting position. Each session comprised 10 runs of 42 trials each, yielding 420 trials per condition and 60 trials per class per condition.

## Data Acquisition

EEG signals were recorded from 61 channels covering frontal, central, parietal, and temporal areas, using active electrodes and four 16-channel amplifiers (g.tec Medical Engineering GmbH, Graz, Austria). Signals were sampled at 512 Hz. The reference electrode was placed on the right mastoid and the ground electrode on AFz. Analog signal processing included an 8th-order Chebyshev bandpass filter (0.01–200 Hz) and a notch filter at 50 Hz to suppress power line interference. Three additional channels (channels 62–64) recorded electrooculographic (EOG) signals. Data were stored in GDF format.

## Preprocessing

All preprocessing was performed in Python (v3.13.14) using MNE-Python (v1.12.1), NumPy (v2.4.6), and SciPy (v1.17.1). The preprocessing pipeline is described below and was applied independently to the ME and MI sessions.

### High-Pass Filtering

A zero-phase finite impulse response (FIR) high-pass filter at 1 Hz (Hanning window, MNE default filter length) was applied to the continuous raw data. This step removed the DC offset and low-frequency physiological drifts, both of which compromise the stability of subsequent ICA decomposition.

### Bad Channel Detection

Bad channels were identified on the high-pass filtered continuous data through two complementary approaches: automated detection based on abnormal amplitude variance and low spatial correlation with neighbouring channels, and visual inspection of the raw signal timeseries and per-channel power spectra. Channels identified as bad were marked and excluded from subsequent processing steps.

### Artifact Removal by Independent Component Analysis

<mark>Carefull review and check this informations. Understand how all of theses proecess are performed</mark>

Independent Component Analysis (ICA) was applied to the high-pass filtered EEG signals to remove physiological artifacts. The number of components was set equal to the effective rank of the data, determined after excluding bad channels. EOG-related components (ocular blinks and eye movements) were identified by computing the Pearson correlation between each ICA component time course and the EOG reference channels (channels 62–64); components exceeding a correlation threshold of r = 0.9 were flagged for removal. ECG-related components were identified using a correlation-based approach against a synthetic cardiac signal derived from the data. EMG and movement artifact components were identified by visual inspection of component scalp topographies, time courses, and power spectra, targeting components with broadband, non-1/f spectral profiles and peripheral topographic distributions. All identified artifactual components were subsequently removed.

### Low-Pass Filtering

Following ICA, a 45 Hz zero-phase FIR low-pass filter (Hanning window) was applied to the continuous data. In combination with the preceding 1 Hz high-pass filter, this yielded an effective bandpass of 1–45 Hz, encompassing both the alpha (8–13 Hz) and beta (13–30 Hz) frequency bands of interest with adequate spectral margin above the upper analysis boundary.

### Bad Channel Interpolation

Channels previously identified as bad were reconstructed using spherical spline interpolation, restoring the full spatial complement of 61 EEG channels prior to re-referencing.

### Re-referencing

The original reference electrode (right mastoid) was reintroduced into the channel set before re-referencing. The data were then re-referenced to the common average reference (CAR), computed across all 62 EEG channels. This step removes the spatial asymmetry introduced by the unilateral mastoid reference, which is particularly relevant when comparing contralateral (C3, C1) and ipsilateral (C4) sensorimotor channels.

### Downsampling

The data were downsampled from 512 Hz to 256 Hz. The 45 Hz low-pass filter applied in the preceding step served as the anti-aliasing filter, ensuring no spectral content above the new Nyquist frequency (128 Hz) was present prior to resampling.

### Epoch Segmentation

Continuous data were segmented into epochs time-locked to the fixation cross onset ($t = 0$ s), with a duration of 5 s ($t = 0–5$ s), encompassing both the pre-cue baseline period ($t = 0–2$ s) and the active task period ($t = 2–5$ s). The movemnt/task onset was set to $t = 2$ s. No baseline correction was applied at this stage; normalization was performed during spectral analysis. All seven class event types (six movements and rest) were retained.

### Bad Epoch Rejection

<mark>Carefull review and check this informations. Understand how all of theses proecess are performed</mark>

Epochs were screened for residual artifacts using four complementary criteria applied in combination:

Amplitude threshold: epochs containing peak-to-peak amplitude exceeding 200 µV (ME) or 150 µV (MI) in any EEG channel were rejected. The distinction reflects the inherently higher signal amplitude expected during physical execution.

Joint probability: the joint probability of each epoch was computed as the log-likelihood of the epoch data given a distribution estimated from all epochs. Epochs exceeding 5 times the standard deviation of the joint probability distribution were rejected.

Kurtosis: the kurtosis of the amplitude distribution was computed per epoch. Epochs exceeding 5 times the standard deviation of the kurtosis distribution were rejected.

Normalized power (EMG index): to specifically detect EMG contamination, a normalized power metric was computed per channel per epoch as the ratio of power in the 20–40 Hz band to power in the 4–40 Hz band, using zero-phase 4th-order Butterworth filters applied to the epoched signal. A high ratio indicates disproportionate high-frequency content consistent with muscle artifact. Epochs in which this metric exceeded 0.8 (ME) or 0.65 (MI) in any channel were rejected.

Subjects for whom the total proportion of rejected epochs exceeded 10% were excluded from further analysis.

### Linear Detrending

A linear trend was removed from each accepted epoch individually, eliminating any residual intra-epoch linear drift following bandpass filtering.

### Spectral Analysis

Spectral analysis was conducted on five central channels: C3, C1, Cz, C2, and C4, which overlie the primary sensorimotor cortex and supplementary motor area. These channels were selected to capture the lateralized and midline sensorimotor dynamics associated with right upper limb movement and imagery.

### Power Spectral Density Estimation

For each epoch, the Power Spectral Density (PSD) was estimated separately for the pre-cue baseline window (t = 0–2 s) and the active task window (t = 2–5 s) using Welch's method, as implemented in SciPy (scipy.signal.welch). Welch's method was configured with 1-second segments (256 samples at 256 Hz), 50% overlap (128 samples), and a Hann window, resulting in a frequency resolution of 1 Hz.

PSD estimates were normalized relative to the pre-cue baseline using the event-related desynchronization/synchronization (ERD/ERS) framework expressed in decibels:

$$\text{ERD/ERS (dB)} = 10 \times \log_{10}\left(\frac{P_{\text{active}}}{P_{\text{baseline}}}\right)$$

where $P_{\text{active}}$ is the PSD of the active task window and $P_{\text{baseline}}$ is the PSD of the pre-cue baseline window of the same trial. This within-trial normalization accounts for session-level absolute power differences between the ME and MI conditions — which were recorded on separate days — and reduces inter-subject spectral variability. Negative ERD/ERS values indicate power decrease (desynchronization) relative to baseline; positive values indicate power increase (synchronization).

Normalized PSD estimates were averaged across trials within each class and condition at the individual subject level.

### Spectral Metrics

Two metrics were extracted from the normalized PSD for each frequency band (alpha: 8–13 Hz; beta: 13–30 Hz), each channel, each class, and each condition:

Band power: the area under the normalized PSD curve across the frequency band, computed by numerical integration using the trapezoidal rule. This metric quantifies the total ERD/ERS magnitude within the band.

Median frequency: the frequency within the band that divides its total power in half (equal integrated area above and below). This metric captures the spectral centroid within each band, providing information about the distribution of power across frequencies that band power alone does not reflect.

The rest class served as the reference condition in between-class comparisons. Group-level normalized PSD curves were obtained by averaging individual subject-level normalized PSDs for each class and condition, for visualization purposes.

### Statistical Analysis

Statistical inference was performed at the individual subject level using the 14 per-subject metric estimates (band power and median frequency), with group-level PSD figures used for visualization. Two families of comparisons were conducted:

ME vs MI: for each movement class, each channel, and each frequency band, band power and median frequency were compared between ME and MI conditions across subjects.

Movement vs rest: within each condition (ME and MI separately), band power and median frequency of each movement class were compared against the rest class for each channel and frequency band.

For each comparison, normality of the distribution across subjects was assessed using the Shapiro-Wilk test. Where normality was satisfied, a paired-samples t-test was applied; otherwise, the non-parametric Wilcoxon signed-rank test was used, as implemented in SciPy (scipy.stats). To control for the false discovery rate arising from multiple simultaneous comparisons across channels, classes, frequency bands, and metrics, p-values were adjusted using the Benjamini-Hochberg FDR procedure (mne.stats.fdr_correction). Statistical significance was defined at a corrected threshold of α = 0.05.