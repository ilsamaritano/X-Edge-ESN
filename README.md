# CICIDS2017 Sequence-Ready Preprocessing

This workspace contains the CICIDS2017 CSV files and a tailored preprocessing notebook that exports data in the shape required by sequence models:

- Input tensor: `(Batch, Timesteps, Features)`
- Target tensor: `(Batch,)`

## Dataset Sources

The repository includes two CSV sources:

- `MachineLearningCSV/MachineLearningCVE`
- `GeneratedLabelledFlows/TrafficLabelling`

The notebook defaults to `MachineLearningCSV/MachineLearningCVE` because it already matches the machine-learning feature format expected for model training.

## Notebook

Main notebook:

- `cicids2017_preprocessing.ipynb`

What the notebook does:

1. Profiles all CSV files (rows, columns, per-file label distribution).
2. Normalizes feature names and resolves duplicate columns deterministically.
3. Cleans invalid numeric values (`Infinity`, `-Infinity`, `NaN`).
4. Performs robust imputation and drops non-informative features.
5. Applies leakage-aware split by source files.
6. Encodes labels and scales features with train-only fitting.
7. Builds sequence tensors with configurable `TIMESTEPS` and `STRIDE`.
8. Exports training artifacts for downstream models.

## Output Artifacts

After execution, outputs are saved to:

- `artifacts/preprocessed_sequences`

Expected files:

- `X_train_seq.part1_of_3.npy`
- `X_train_seq.part2_of_3.npy`
- `X_train_seq.part3_of_3.npy`
- `y_train_seq.npy`
- `X_val_seq.npy`
- `y_val_seq.npy`
- `X_test_seq.npy`
- `y_test_seq.npy`
- `metadata.json`

Note: train sequences are sharded to comply with GitHub LFS 2GB per-object limit.

`metadata.json` includes:

- sequence settings (`timesteps`, `stride`)
- final feature list
- class names and class weights
- split file lists
- tensor shapes

## Recommended Usage

1. Open and run `cicids2017_preprocessing.ipynb` top-to-bottom.
2. Validate tensor shapes in the final printout.
3. Train your model using `X_*_seq` and `y_*_seq`.
4. Use `metadata.json` to keep preprocessing and training configuration aligned.

## Notes

- The CICIDS2017 class distribution is highly imbalanced. Class weights are exported to help stabilize training.
- If memory usage is high, increase `STRIDE` or reduce `TIMESTEPS`.
- You can switch data source by changing `DATA_DIR` in the notebook configuration cell.
