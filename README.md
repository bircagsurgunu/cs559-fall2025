# CS559 Fall 2025 – Facial Attractiveness Regression (TensorFlow)

Authors: `<Insert Names and Bilkent IDs>`

## Overview
- Predict continuous attractiveness scores on the range `[0, 10]` from SCUT_FBP5500_downsampled face images.
- CNN-based regression on `(80, 80, 3)` inputs.
- Evaluation includes standard MAE and rounded-clipped MAE (RC-MAE).

## Repository Structure
- `model.py` – Modular CNN regressor with configurable depth, filters, dropout, L2, batch norm, and weight initialisation (Xavier or Gaussian). Includes RC-MAE metric, early stopping callbacks, and helper functions.
- `utils.py` – Dataset loading via `tf.data`, plotting utilities, JSON/CSV helpers, and qualitative success/failure image export.
- `train.py` – Runs the full suite of experiments (architecture, loss, init, BN, L2, dropout, Adam tuning). Each experiment logs metrics, saves models, and updates aggregate summaries.
- `report.tex` – IEEEtran-formatted report summarising methodology, experiments, and discussion. Compile with `pdflatex`.
- `results/` – Output directory created by `train.py` (see below for contents).

## Dataset Expectations
- Place the dataset at `--data-root` (e.g., `path/to/SCUT_FBP5500_downsampled/`).
- Default layout (used by the provided archive):
  - `train/`, `val/`, `test/` folders containing `.jpg` images.
  - Each filename encodes its label in the prefix: `<label>_<acquisition_id>.jpg`, where labels lie in `[0, 8]`.
  - The code parses the prefix directly and constrains predictions within `[0, 10]`.
- Optional legacy layout: if a `labels.csv` exists, it is used to build random splits when directory splits are absent.
- Images are resized to `80×80` if necessary.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install tensorflow pillow matplotlib
```

## Running Experiments
```bash
python train.py --data-root /path/to/SCUT_FBP5500_downsampled --output results
```

### Optional arguments
- `--labels-csv`: Custom path to labels file (only needed for CSV-driven datasets).
- `--val-split` / `--test-split`: Adjust data splits when random splitting is used (e.g., datasets without explicit `val/` or `test/`).
- `--batch-size`: Override baseline batch size.
- `--experiments`: Restrict experiments, e.g.\
  `python train.py --data-root ... --experiments architecture dropout`

## Outputs (`results/`)
- `runs/<run_name>/` – Per-run history, metrics, saved model.
- `summary.csv` – Aggregated metrics across all runs.
- `<experiment>.csv` – Metrics per experiment group.
- `validation_mae_curves.png` – Curves for the best validation RC-MAE run.
- `success_examples/`, `failure_examples/` – Top qualitative examples on the test set.
- `best_model_summary.json` – Metadata for the best-performing configuration.

- Seeds set through `utils.set_global_seed`.
- Early stopping monitors validation RC-MAE (`rounded_clipped_mae`), clipping and rounding predictions in `[0, 10]`.
- Minor numerical variations may occur across hardware (CPU vs GPU).

## Report
- Compile IEEE report:
  ```bash
  pdflatex report.tex
  ```
- Ensure `results/` exists so tables and figures populate when compiling.

## Citation
- TensorFlow: M. Abadi *et al.*, “TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems,” 2016.
- Dataset: X. Liang *et al.*, “SCUT-FBP5500: A Diverse Benchmark Dataset for Multi-Paradigm Facial Beauty Prediction,” 2018.
