# Step-by-step Of Event Identification and PSD Estimation

0) Segmentating the data by epochs that indicate each event occurence.
1) I need for my analysis the voltage time-series of each channel between the begginning and end of each epoch.
2) For each voltage time-series corresponding to a single epoch I want to compute its PSD. I will define further if I will use the Welch or multitaper method .
3) The provided code and the two previous steps (1 and 2) are applied to a single record, but for each subject in my dataset I have 10 recordings. Then, I have thought in applying these steps to all recordings. Maybe in a sequential way to save memory, but could also be in a parallel/vectorized way as well. Also, the result for each recording could be saved as a csv, pickle or other kind of file to also save runtime memory.
4) There are a total of 14 subjects in my dataset. For each subject, the previous steps will give me 10 results. Then, I would like to combine these 10 results and compute an average PSD for each kind of event. An average of other metrics can also be computed in this step if more metrics are needed. Then, the average results will be saved in another csv, pickle or other kind of file. That is, we will have an average result per subject.
5) Then, I will apply all these previous steps (from 1 to 4) to the data of the same subjects but in another experimental condition. In total, we have two conditions: "motor execution" and "motor imagery".
6) After that I am intending to compare the PSD in alpha and beta bands between the two conditions. Since the average PSD is very particular for each subject, I was thinking in a way of doing a relative comparation, for example, comparing the event A with the control event (rest) in both conditions. Then, compare the eventA-rest in motor execution to the eventA-rest in motor imagery. I am not so sure which metrics I am going to use and how I will do this comparation. But, the key point in this step 6 is to define a structure to store the data in a way that analysis can be done in a easy and efficient way.

## Considerations

- Steps 1+2 → single function. MNE's `epochs.compute_psd()` inherently absorbs step 1 (time-series access) internally — there is no reason to extract the array explicitly before calling it. The result is an EpochsSpectrum object with shape (n_epochs, n_channels, n_freqs), which is exactly what we need before averaging.

- Step 3 → orchestration function. A `process_single_recording()` that chains the existing `load_and_prepare_data` → preprocessing on Raw → `segment_data_by_events` → step 1+2 function → saves result to disk. This is the process that gets called 10 times per subject.

- Step 4 → aggregation function. A `compute_subject_average_psd()` that loads the 10 saved results and uses `EpochsSpectrum.average()` (or a manual mean if combining across files) to produce one averaged PSD per event per subject.

## Attention Points

Before implementing, two issues in the current code worth fixing:

1. `pick_channels(CHANNELS_OF_INTEREST)` is called inside `load_and_prepare_data` before re-referencing. As noted, re-referencing needs all channels. The `pick` should move to ***after*** re-referencing, inside the preprocessing step on `Raw`.
2. `flat={'eeg': 400e-6}` looks inverted — `flat` rejects epochs whose peak-to-peak is below the threshold (disconnected electrode). At 400 µV that would never trigger. A sensible value is around 1e-6 (1 µV). The upper amplitude rejection belongs in the reject parameter instead.