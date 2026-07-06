"""
main.py — Run the full EEG workflow (preprocessing + descriptive analysis) at once.

Stage 1  main_load_and_process(): reads raw GDF recordings under DATA_DIR and
         writes per-run processed pickles under OUTPUT_DIR.
Stage 2  main_analysis(): reads those pickles and writes the backbone, figures,
         and summary tables under RESULTS_ROOT.
"""

from pathlib import Path

from eeg_pipeline import main_load_and_process
from eeg_analysis import main_analysis

# Repo root, derived from this file's location /src/main.py
BASE_DIR = Path(__file__).resolve().parent.parent

# ── EEG data processing parameters ────────────────────────────────────────────
data_dir = BASE_DIR / "data"                 # raw GDF recordings
output_dir = BASE_DIR / "data_processed"     # per-run processed pickles
# Processing only "ME" (motor execution) or "MI" (motor imagery) data, or both
condition_filter = None      # "ME", or "MI", or None (both)
# Skip specific subjects/runs
skip = None                  # e.g. ["S01"] to skip subject S01
overwrite = False            # False -> skip recordings already processed ( do not overwrite)

# ── EEG data analysis parameters ──────────────────────────────────────────────
processed_root = output_dir  # analysis reads exactly where processing wrote
results_root = BASE_DIR / "results"          # results: backbone, figures, tables
rebuild = False              # False -> reuse an existing backbone if present
per_subject_figs = True      # also emit per-subject figures

# ── Run processing then analysis ──────────────────────────────────────────────
if __name__ == "__main__":
    main_load_and_process(
        data_dir=data_dir,
        output_dir=output_dir,
        condition_filter=condition_filter,
        skip=skip,
        overwrite=overwrite,
    )

    main_analysis(
        processed_root=processed_root,
        results_root=results_root,
        rebuild=rebuild,
        per_subject_figs=per_subject_figs,
    )
