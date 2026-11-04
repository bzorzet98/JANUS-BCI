# Noticed, not touched

Findings that a step noticed but did not change, because they were outside its scope (`CLAUDE.md`, section 4). One section per step, in the order of `docs/IMPLEMENTATION_ORDER.md`. When a planned step covers an item, the item names it (for example → C2). When a step fixes an item, it deletes the line in the same Pull Request.

## A1

- `README.md`, `PROTOCOL.md` and `projects/NID/README.md` still name `scripts/run_production.py`. The commands today are the per-stage scripts (`run_preprocessing.py`, `run_training.py`, ...).
- `Subject._to_mne_epochs`: the error message talks about the keys `X` and `y`, but the check is on `data` and `labels`.
- `EEGDatabasePreprocessor.run`: when loading a subject fails, the partition `subject_XX/unknown/unknown` has no session either. Same family as C1 (→ C1).
- `Lee2019.__init__` uses a mutable list as the default of `sessions`.

## A2

- `CustomCheckpoint.on_epoch_end` overwrites `seed_N_epoch_0.pth` after epoch 0, and epoch `max_epochs` is never saved, because the loop runs from 0 to `max_epochs - 1`. Without `EarlyStopping(load_best=True)`, `seed_N_best.pth` holds the last-epoch weights.
- `create_dataset` and `create_dataloader` (`src/training/utils.py`): when a split key is missing (for example `train_eval`), they fall back to the whole block and fail with a `KeyError` instead of a clear message.
- `WithinUnitHoldoutSchema` takes `ptrain` but does not use it. Only `pval` and `ptest` decide the split.
- `RichProgressReporter` is written for preprocessing. During training it shows "Barrido de preprocesamiento", "Dataset: ?", a "Preprocessing ... subjects" bar and "Success: 0 · Failed: 0", because the training orchestrator passes other keys in `meta` and `on_finish` receives only `{"total": n}`.
- `MLflowLogger.log_dataset_summary` makes MLflow emit three `UserWarning`s per replica, because the dataset source is a plain string. Noise only.
- If `MLFLOW_TRACKING_URI` is not set, MLflow uses its default store in the current folder, which `.gitignore` does not cover (it covers only `mlflow/mlruns/`). Not checked on the author's machine.
- `get_git_commit_hash` catches only `CalledProcessError`. A missing git binary raises `FileNotFoundError` (→ C2).
- `.docker_image` is written only when `--docker-image` is given (→ C2, which adds the environment variable). Consider recording the image digest, not only the tag.
- `README.md` and `PROTOCOL.md` also name `src/networks` and `src/torch_utils`, which are not packages today. Models are in `src/models`, torch datasets in `src/eeg_datasets` and callbacks in `src/training/core/callbacks`.
- `projects/NID/configs/training/WS-Standard/2026-NID_WS-Standard_CTNet_Cho2017_CAR-Bilateral-Full.json` uses the old format (`general_script_config`, `model_to_train`, `src.networks`, `src.torch_utils`). It has no `recipe_name`, `data_provider`, `schema` or `orchestrator`, so the current training orchestrator cannot run it. C4 will reject it with a clear message (→ C4); migrating it is not in any step. Only this NID config was read.
