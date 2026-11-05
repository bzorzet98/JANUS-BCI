# dummy-project

Safety net for JANUS-BCI (PRD R21). A synthetic dataset small enough to run every stage on a CPU in seconds. Every step of `docs/IMPLEMENTATION_ORDER.md` must leave this project running.

## Synthetic dataset

`SyntheticBCI` (`src/eeg_datasets/synthetic.py`):

- 3 units (`1`, `2`, `3`) and 2 sessions (`session_1`, `session_2`).
- One `motor_imagery` run per unit and session.
- 4 EEG channels (`C3`, `Cz`, `C4`, `Pz`) plus a `stim` channel, at 128 Hz.
- 20 trials of 2 s, 10 per class (`left_hand` = 1, `right_hand` = 2). Each trial adds a 10 Hz sinusoid on `C3` (left) or `C4` (right) over Gaussian noise.
- Generated in memory from a fixed seed per (unit, session). Nothing is downloaded or read from disk.

## Preprocessing

```bash
python scripts/run_preprocessing.py \
  --config projects/dummy-project/configs/preprocessing/SyntheticBCI_s1-s2_dummy-preproc.json
```

Pipeline: band-pass 1 to 40 Hz, epochs of 0 to 2 s on the EEG channels, then arrays saved as `.npz`.

Output: `<DATA_ROOT>/preprocessed/dummy-preproc/SyntheticBCI/`, with one `<session>/subject_XX/subject_XX_motor_imagery_run_1.npz` per unit and session.

## Training

Run the preprocessing first.

```bash
python scripts/run_training.py \
  --config projects/dummy-project/configs/training/WithinUnit/dummy-project_WithinUnit_TinyNet_SyntheticBCI_dummy-train.json \
  --device cpu
```

Reads `dummy-preproc` (`session_1`, units 1 to 3), splits each unit into 12/4/4 trials (train/val/test) with `WithinUnitHoldoutSchema`, and trains `TinyNet` (`src/models/tiny_net.py`) for 10 epochs with seeds 11 and 22. `CustomCheckpoint` saves the weights.

Output: `<RESULTS_ROOT>/dummy-project/WithinUnit/TinyNet/SyntheticBCI_session_1/dummy-train/`, with `metrics_results.csv` and, per unit (`unit_1_fold_0` to `unit_3_fold_0`), `fold_metadata.json`, `outputs/seed_N_outputs.csv`, `outputs/seed_N_history.csv` and `weights/seed_N_best.pth`.

## Evaluation

Run the training first.

```bash
python scripts/run_testing.py \
  --config projects/dummy-project/configs/testing/WithinUnit-Test/dummy-project_WithinUnit-Test_TinyNet_SyntheticBCI_dummy-eval.json \
  --device cpu
```

Loads the `seed_N_best.pth` weights of `dummy-train` and evaluates each one on its whole unit (the 20 trials of `session_1`, training trials included).

Output: `<RESULTS_ROOT>/dummy-project/WithinUnit-Test/TinyNet/SyntheticBCI_session_1/dummy-eval/`, with `metrics_results.csv` (accuracy per unit and origin seed) and, per unit, `.mlflow_run_id_N`.

## HP search

Run the preprocessing first.

```bash
python scripts/run_hp_search.py \
  --hp_search_config projects/dummy-project/configs/hp_search/WithinUnit/dummy-project_WithinUnit_TinyNet_SyntheticBCI_optimizer-lr.json \
  --base_config projects/dummy-project/configs/training/WithinUnit/dummy-project_WithinUnit_TinyNet_SyntheticBCI_dummy-train.json \
  --device cpu
```

Uses the training config as the base and tries two Adam learning rates (0.01 and 0.001) on units 1 and 2, with seed 11 only. Each combination is a training run of its own.

Output: `<RESULTS_ROOT>/dummy-project/WithinUnit_HPSearch/TinyNet/SyntheticBCI_session_1/`, with `hp_combinations.csv` and one training folder per combination, labelled `dummy-train_hp<hash>`.
