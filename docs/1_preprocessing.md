# Methodoligical Procedures for Data Processing in Ofner et al 2017

Just to clarify, the Ofner et al 2017 is the paper that generated the EEG data we are using on our research.

## EEG Data Acquisition and Analogic Pre-processing:

- The  EEG  was  measured  from  61  channels  covering  frontal,  central,  parietal  and  temporal  areas  using active electrodes;
- Reference electrodewas placed on the right mastoid and ground electrode on AFz.
- Sampled at 512 Hz;
- Pass-band filtered from 0.01 Hz to 200 Hz (8th order Chebyshev);
- Notch filtered at 50 Hz. Not specified rejection bandwidth, filter type and filter order;

## Digital pre-processing (Ofner et al 2017):

- Removed noisy channels based on the joint probability of each channel. There were no more details about how they were computed and related procedures;
- Downsampling to 256 Hz;
- Artefacts identification by band-pass filtering (0.3 Hz to 70 Hz, 4th order zero-phase Butterworth filter) the data and finding:
    1. values above or below thresholds of -200 μV and 200 μV, respectively;
    2. trials with abnormal joint probabilities;
    3. trials with abnormal kurtosis;
    - **Note**: The previous methods 2 and 3 used as threshold 5 times the standard deviation of their statistic to detect artefact contaminated trials. The artefact contaminated trials were only marked for removal but not yet removed in this phase;
- The 256 Hz EEG data (unfiltered) was then band-pass filtered (0.3 Hz to 3 Hz) with a zero-phase 4th order Butterworth filter and re-referenced the data to a common average reference (it is not clear if is the average among all EEG channels).
- The trials marked as artefact contaminated (steps 1, 2 and 3 previously) were then discarded.

# Methodoligical Procedures for Data Processing in Guzman et al 2026

Just to clarify, this is a very interisting new study. So, some methodological procedures might be useful for our research.

## Digital pre-processing (Guzman et al 2026):

- EEG data segmentation in events (classes of movement tasks);
- Voltage offsets removal by band-pass filtering (0.5 Hz to 60 Hz, 4th order zero-phase Butterworth filter);
- Artefacts identification by computing the following metrics for each EEG channel:
    1. peak-to-peak voltage ($V_w^{pp}$): $V^{pp} = \max\{x_{4-40}\} - \min\{x_{4-40}\}$
    2. standard deviation ($\sigma$): $\sigma = \sqrt(\frac{1}{n-1} \sum_{t=1}^{n}{(x_{4-40} - \mu)^2})$
    3. normalized power ($Pn$): $Pn = \frac{\sum_{t=1}^{n}{(x_{20-40})^2}}{\sum_{t=1}^{n}{(x_{4-40})^2}}$
    - **Note 1**: where $x_{4-40}$ and $x_{20-40}$ are EEG signals band-passed (4th order zero-phase Butterworth) with cutoff frequencies of 4-40 Hz and 20-40 Hz, respectively. And $\mu = \frac{1}{n} \sum_{t=1}^{n}{(x_{4-40} - \mu)^2}$
    - **Note 2**: Check https://doi.org/10.3389/fninf.2022.961089 to understand why these previous 3 metrics are used and what they mean.
- A trial was discarded if it did not meet any of the following threshold conditions for **motor execution**:
    1. a peak-to-peak value ($V^{pp}$) exceeding 400 $\mu$V;
    2. an amplitude standard deviation ($\sigma$) greater than 60 $\mu$V;
    3. a normalized power $Pn$ (signal-to-noise ratio) above 0.8;
- A trial was discarded if it did not meet any of the following threshold conditions for **motor imagery**:
    1. a peak-to-peak value ($V^{pp}$) exceeding 150 $\mu$V;
    2. an amplitude standard deviation ($\sigma$) greater than 40 $\mu$V;
    3. a normalized power $Pn$ (signal-to-noise ratio) above 0.865;
- The previous preprocessed data were bandpass filtered using fourth-order zero-phase Butterworth filters for each EEG channel to isolate the alpha (7 to 13 Hz) and beta (14 to 35 Hz) frequency bands.

# Other Papers with Interisting and Useful Pre-processing Steps:

- [Dynamics of sensorimotor-related brain oscillations: EEG insights from healthy individuals in varied upper limb movement conditions](https://link.springer.com/article/10.1007/s00221-025-07116-6). Research conducted by USP.
- [The reorganization mechanism of brain functional networks by long term actual exercise and exercise imagination training: spectrum and network topology analysis based on EEG](https://link.springer.com/article/10.1186/s12984-026-01935-6)

# Possible Pipeline to Pre-process the EEG Data

Here are some possible step-by-step of pre-processing. However, considering the previous preprocessing that were applied in published papers, might be a better way to define the preprocessing steps in our research.

  - Change reference. Maybe use as the new reference the average of the electrodes.
  - Band-pass filtering using a 4th order zero-phase Butterworth (bandwidth to be defined) and maybe a Notch filter at 50 Hz and its harmonics.
    - In order to use a Notch filter the PSD of the whole EGG time-series (10 runs times number of channels per subject) must be check to identify the influence in the power by the 50 Hz and its harmonics. Moreover, depending on the band-pass bandwidth, we do not need to apply the Notch filter. This second option, might be the more suitable.
  - Dowmnsampling if needed. I believe that it is not needed.
  - Epoch segmentation (realizations).
  - Reject epochs by amplitude threshold. Which value use? Why? Papers?
  - Detrending per epoch.
  - Baseline wander correction. It must be investigate if do this correction after epoch segmentation or before, i.e., for each individual full EGG time-series. Or even in both cases, but why? There are papers about this?

## Key Points of the Data to Consider

- Some channels has very abnormal outliers at the beggining of the recording. This might be taked into account.
- Altought any previous paper has centered the EEG channels to zero mean (by doing x[n] - mean(x[n])), is this a good practice to be done? In case of positive answer, why? If this information is important: consider that the power spectrum density will be computed furhter.