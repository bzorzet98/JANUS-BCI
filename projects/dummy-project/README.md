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
