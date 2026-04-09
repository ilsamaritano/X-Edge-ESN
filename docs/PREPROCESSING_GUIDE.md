# Preprocessing Guide (Professional Reference)

This guide documents the preprocessing strategy implemented in `cicids2017_preprocessing.ipynb`.

## 1. Design Objective

Prepare CICIDS2017 flow CSVs for sequence-based models by generating stable, reproducible tensors in the shape:

- `X`: `(Batch, Timesteps, Features)`
- `y`: `(Batch,)`

This format is directly compatible with RNN, LSTM, GRU, Transformer encoders, and ESN-style sequence pipelines.

## 2. Dataset Observations

The notebook is tailored to the observed dataset behavior:

- Uniform schema across all daily files (same feature set).
- Presence of duplicate feature names in the header (e.g., `Fwd Header Length`).
- Presence of invalid numeric tokens (`Infinity` and `NaN`) across files.
- Strong class imbalance, including minority attack classes.

Because of these properties, the pipeline includes deterministic column disambiguation, robust numeric cleaning, and class-weight export.

## 3. Processing Steps

1. File discovery and profiling
- Enumerates all CSV files in the selected folder.
- Collects row count and per-file label distribution.

2. Schema normalization
- Strips spaces and normalizes feature names.
- Resolves duplicate column names with suffixes (`_dup1`, `_dup2`, ...).

3. Numeric sanitation
- Converts all feature columns to numeric with coercion.
- Replaces `+/-Infinity` with `NaN`.

4. Missing-data policy
- Drops columns with excessive missingness.
- Fills remaining missing values with median statistics.

5. Feature stability
- Removes constant (zero-variance) features.

6. Leakage-aware splitting
- Splits by source files, preserving chronological day order.
- Prevents random row-level leakage across train/validation/test.

7. Label encoding and scaling
- Fits `LabelEncoder` on train labels only.
- Fits `RobustScaler` on train features only.

8. Sequence construction
- Builds sliding windows with configurable `TIMESTEPS` and `STRIDE`.
- Generates labels from the last timestep (causal labeling policy).
- Avoids sequence windows crossing file boundaries.

9. Export
- Saves NumPy arrays for train/val/test sequences.
- Saves `metadata.json` with classes, feature names, shapes, split configuration, and class weights.

## 4. Configuration Knobs

In the notebook configuration cell:

- `DATA_DIR`: input CSV root.
- `TIMESTEPS`: sequence length.
- `STRIDE`: window step size.
- `MIN_NON_NULL_RATIO`: minimum valid-data threshold for keeping a feature.

## 5. Operational Guidance

- Increase `TIMESTEPS` to capture longer temporal context.
- Increase `STRIDE` to reduce total sequence count and memory footprint.
- Keep scaling and label encoding train-fitted only for valid evaluation.
- Preserve `metadata.json` with each experiment for reproducibility.

## 6. Integration Contract

Downstream training code can rely on:

- Sequence tensors stored in `artifacts/preprocessed_sequences`.
- Class map in `metadata.json -> classes`.
- Feature ordering in `metadata.json -> feature_names`.

This contract ensures consistent preprocessing across training, validation, and inference workflows.
