# IA753 Final Work — Motor Execution vs. Motor Imagery from EEG

Course project for **IA753 (Biological Signal Analysis, UNICAMP)**. The code
quantifies and compares the sensorimotor rhythms elicited during **Motor
Execution (ME)** and **Motor Imagery (MI)** of upper-limb movements, using the
publicly available dataset of **Ofner et al. (2017)** — *"Upper limb movements
can be decoded from the time-domain of low-frequency EEG"* (PLOS ONE 12(8):
e0182578).

## What this repository does, and why

The scientific question is whether imagining a movement engages the same
sensorimotor machinery as actually performing it, and how strongly. To answer
it, the code turns raw EEG into two interpretable descriptors over the
sensorimotor channels **C3 / Cz / C4** in the **alpha (8–13 Hz)** and
**beta (13–30 Hz)** bands:

- **Power Spectral Density (PSD)** via Welch's method — the spectral fingerprint
  of a baseline (pre-cue) window versus an active (movement/imagery) window.
- **ERD/ERS** (Event-Related Desynchronization / Synchronization) —
  `10·log10(active / baseline)` in dB. Negative values mark the desynchronization
  (power drop) that accompanies motor activity; positive values mark
  synchronization.

The implementation mirrors the **Methods** of the accompanying paper and is
split into two layers, each a standalone module:

1. **Preprocessing & spectral estimation** (`src/eeg_pipeline.py`) — reads each
   raw GDF recording, filters and cleans it, cuts it into epochs around each
   movement cue, and computes per-epoch baseline/active PSDs and the ERD/ERS
   spectrum plus per-band scalar descriptors. Output: one pickle per run.
2. **Descriptive analysis** (`src/eeg_analysis.py`) — pools each subject's runs
   (~60 trials/class), builds a reusable data *backbone* (tidy scalar table +
   stacked curve arrays), and emits the curve figures, scalar box plots, and
   summary tables. Aggregation collapses epochs to one mean per subject first,
   then computes group statistics across the 14 subject-means (avoids
   pseudoreplication).

### Processing parameters (defaults)

| Stage | Parameter | Value |
|---|---|---|
| Filtering | FIR band-pass (Hann, zero-phase) | 1–45 Hz |
| Epoching | window around cue (movement onset) | −2.0 to +3.0 s |
| Baseline window | reference for ERD/ERS | −2.0 to 0.0 s |
| Active window | movement/imagery | 0.0 to +3.0 s |
| PSD | Welch, 1 s Hann segments, 50 % overlap, `n_fft = 2·sfreq` | 1–45 Hz |
| Channels | analyzed | C3, Cz, C4 |
| Bands | alpha / beta | 8–13 / 13–30 Hz |

## Project Structure

```
IA753_Final_Work/
├── README.md                    — this file
├── IA753_Final.ipynb            — Google Colab notebook: clone, install, run both pipelines
├── requirements.txt             — pip dependencies (pinned)
├── environment.yml              — conda environment specification
├── pyproject.toml               — packaging metadata + optional extras (notebook, dev)
├── setup.py                     — setuptools install (mirrors pyproject dependencies)
│
├── data/                        — raw dataset (not tracked; download separately)
│   ├── README                   — dataset provenance, license, and download link
│   ├── dataset_description.pdf   — official dataset description
│   ├── motor_execution/         — ME recordings
│   │   └── Sxx_ME/*.gdf          — one folder per subject, ~10 GDF runs each
│   └── motor_imagery/           — MI recordings
│       └── Sxx_MI/*.gdf          — one folder per subject, ~10 GDF runs each
│
├── data_processed/              — Stage 1 output: per-run pickles, mirrors the data/ tree
│   └── {motor_execution,motor_imagery}/Sxx_XX/processed_*.pkl
│
├── results/                     — Stage 2 output
│   ├── backbone/
│   │   ├── scalars.pkl           — tidy per-epoch band-scalar DataFrame
│   │   └── curves.npz            — subject-mean PSD/ERD-ERS curves + CI bounds
│   ├── figures/
│   │   ├── psd/{pooled,per_class}/    — PSD baseline-vs-active SVG figures
│   │   ├── erds/{pooled,per_class}/   — ERD/ERS curve SVG figures
│   │   └── scalars/{,per_subject}/    — band-scalar box plots (SVG)
│   └── tables/
│       └── summary_by_group.csv       — mean / SD / 95 % CI per group
│
└── src/
    ├── main.py                  — local entry point: runs Stage 1 then Stage 2
    ├── eeg_pipeline.py          — Stage 1: GDF $\to$ preprocess $\to$ epoch $\to$ spectra
    └── eeg_analysis.py          — Stage 2: pool runs $\to$ backbone $\to$ figures/tables
```

> In the `data/` there is no recordings due to its high size. To reproduce the
> full analysis, download the complete dataset from the BNCI Horizon 2020
> database (see [data/README.md](data/README.md)) and place the recordings under
> `data/motor_execution/Sxx_ME/` and `data/motor_imagery/Sxx_MI/`.

## Usage

### Setup and requirements

- **Python 3.13+**
- Key packages (pinned in `requirements.txt`): `mne`, `numpy`, `scipy`,
  `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.

Using conda (recommended):
```bash
conda env create -f environment.yml
conda activate ia753
```

Or with pip:
```bash
pip install -r requirements.txt
```

### Main function parameters

Both entry points call two functions. Their parameters:

**`main_load_and_process(...)`** — Stage 1 (`eeg_pipeline.py`)

| Parameter | Meaning |
|---|---|
| `data_dir` | Root of the raw GDF recordings (`.../data`). Condition is inferred from the folder name (`motor_execution` $\to$ ME, `motor_imagery` $\to$ MI). |
| `output_dir` | Where per-run pickles are written, in a tree mirroring `data_dir`. |
| `condition_filter` | `"ME"`, `"MI"`, or `None` (both). Restricts the batch to one condition. |
| `skip` | List of path substrings to skip, e.g. `["S01"]` to exclude subject S01. `None` processes all. |
| `overwrite` | `False` skips recordings already processed (safe to resume); `True` reprocesses everything. |

**`main_analysis(...)`** — Stage 2 (`eeg_analysis.py`)

| Parameter | Meaning |
|---|---|
| `processed_root` | Root of the Stage 1 pickles (set equal to `output_dir`). |
| `results_root` | Where the backbone, figures, and tables are written. |
| `rebuild` | `False` reuses an existing backbone if present (fast); `True` re-reads every pickle and rebuilds it. |
| `per_subject_figs` | `True` also emits per-subject figures; group-level figures are always produced. |

### Run on Google Colab

Open **[IA753_Final.ipynb](IA753_Final.ipynb)** in Colab (use the *Open in Colab*
badge at the top of the notebook) and run the cells top to bottom:

1. Clone the repo and install dependencies (`!pip install -r requirements.txt`).
   Colab will ask to **restart the kernel** — do so, then continue from the next
   cell (do not re-run the install cell).
2. The path cell adds `/content/IA753_Final_Work/src` to `sys.path` and imports
   both pipelines.
3. The parameter cell sets `BASE_DIR = "/content/IA753_Final_Work"` and the same
   parameters described above.
4. The last two cells run `main_load_and_process(...)` then `main_analysis(...)`.

### Run locally

Everything is wired in **[src/main.py](src/main.py)**. It derives the repo root
from its own location (`BASE_DIR = Path(__file__).resolve().parent.parent`), so
no path editing is needed as long as the layout above is respected. Adjust the
parameter block near the top if you want to filter conditions, change output
locations, or force a rebuild, then run:

```bash
python src/main.py
```

This runs Stage 1 (raw GDF $\to$ processed pickles) followed by Stage 2 (backbone $\to$
figures + tables). With `overwrite=False` and `rebuild=False` (the defaults),
re-running is cheap: already-processed recordings and an existing backbone are
reused.

## Dataset & License

Data: Ofner P, Schwarz A, Pereira J, Müller-Putz GR (2017), *PLOS ONE* 12(8):
e0182578 — CC BY 4.0, Institute of Neural Engineering, TU Graz. See
[data/README](data/README) for the download link and full attribution.

Code: MIT.
