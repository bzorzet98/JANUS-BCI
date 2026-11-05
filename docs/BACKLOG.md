# Noticed, not touched

Findings that a step noticed but did not change, because they were outside its scope (`CLAUDE.md`, section 4). One section per step, in the order of `docs/IMPLEMENTATION_ORDER.md`. When a planned step covers an item, the item names it (for example → C2). When a step fixes an item, it deletes the line in the same Pull Request.

## A1

- `README.md`, `PROTOCOL.md` and `projects/NID/README.md` still name `scripts/run_production.py`. The commands today are the per-stage scripts (`run_preprocessing.py`, `run_training.py`, ...).
- `Subject._to_mne_epochs`: the error message talks about the keys `X` and `y`, but the check is on `data` and `labels`.
- `Lee2019.__init__` uses a mutable list as the default of `sessions`.

## A2

- `CustomCheckpoint.on_epoch_end` overwrites `seed_N_epoch_0.pth` after epoch 0, and epoch `max_epochs` is never saved, because the loop runs from 0 to `max_epochs - 1`. Without `EarlyStopping(load_best=True)`, `seed_N_best.pth` holds the last-epoch weights.
- `create_dataset` and `create_dataloader` (`src/training/utils.py`): when a split key is missing (for example `train_eval`), they fall back to the whole block and fail with a `KeyError` instead of a clear message.
- `WithinUnitHoldoutSchema` takes `ptrain` but does not use it. Only `pval` and `ptest` decide the split.
- `RichProgressReporter` is written for preprocessing. During training it shows "Barrido de preprocesamiento", "Dataset: ?", a "Preprocessing ... subjects" bar and "Success: 0 · Failed: 0", because the training orchestrator passes other keys in `meta` and `on_finish` receives only `{"total": n}`.
- `MLflowLogger.log_dataset_summary` makes MLflow emit three `UserWarning`s per replica, because the dataset source is a plain string. Noise only.
- If `MLFLOW_TRACKING_URI` is not set, MLflow uses its default store `mlruns/` in the current folder. Seen on the author's machine: `mlruns/` appeared in the repo root. `.gitignore` now ignores `mlruns/` at any level, but the store still lands in the repo root instead of `mlflow/`.
- `get_git_commit_hash` catches only `CalledProcessError`. A missing git binary raises `FileNotFoundError` (→ C2).
- `.docker_image` is written only when `--docker-image` is given (→ C2, which adds the environment variable). Consider recording the image digest, not only the tag.
- `README.md` and `PROTOCOL.md` also name `src/networks` and `src/torch_utils`, which are not packages today. Models are in `src/models`, torch datasets in `src/eeg_datasets` and callbacks in `src/training/core/callbacks`.
- `projects/NID/configs/training/WS-Standard/2026-NID_WS-Standard_CTNet_Cho2017_CAR-Bilateral-Full.json` uses the old format (`general_script_config`, `model_to_train`, `src.networks`, `src.torch_utils`). It has no `recipe_name`, `data_provider`, `schema` or `orchestrator`, so the current training orchestrator cannot run it. C4 will reject it with a clear message (→ C4); migrating it is not in any step. Only this NID config was read.

## A3

- Evaluation in the dummy-project uses the same dataset and session as training, and evaluates the whole unit, training trials included. This is a known limit (`STAGE_CONTRACTS.md`, section 4), so the evaluation accuracy means nothing there.
- Relaunching the HP search appends the rows of `hp_combinations.csv` again (→ G2). The combinations themselves report that there is nothing to do.
- `run_hp_search.py` uses underscores in `--hp_search_config` and `--base_config`, and has no `--force` or `--expand` to forward (→ G1).
- `TestingOrchestrator` builds the origin path with `project_config.project_name`, while `TrainedModelResolver` reads `model_source.project_name` (→ C3). The dummy evaluation config sets both to `dummy-project`.
- `TestingOrchestrator.run` writes `config.json` and `.git_commit` before resolving the weights. If `model_source` points to a folder that does not exist (for example a typo in the label), it logs a warning about the origin schema and an INFO line ("nada para hacer"), then ends without an error and leaves the record behind. Related to C3.
- `TestingOrchestrator.run` creates the output folder before the identity check, so a rejected run leaves an empty folder (→ F1).
- `TestingOrchestrator._run_replicate` writes `config.json` and `.git_commit` inside each partition folder, once per replica, so every unit folder (`unit_1_fold_0`, ...) holds a copy of the config. `STAGE_CONTRACTS.md` section 4 asks for one record per execution. Author's decision: a future step removes the per-partition copies and keeps a single `config.json` in the label folder. No step in the order names it yet.
- `TestingOrchestrator` decides what to resume from the `.mlflow_run_id_<seed>` markers (→ D3).
- The MLflow experiment of an evaluation is `project/strategy/recipe/dataset/label`, while training uses `project/strategy`.
- The HP search config has no `script_type`, although `PROTOCOL.md` section 5 asks for it in every config (and section 2 lists `hypersearch` as future). The orchestrator does not read it (→ G1, which validates the search config).

## B1

- The `<database>_<session>` rule is still repeated outside `experiment_path` in MLflow names: `TrainingOrchestrator.dataset_dir_name` (passed to `MLflowLogger.log_dataset_summary`) and the MLflow experiment name in `TestingOrchestrator.run`. They are names, not paths, so B1 left them.

## B2

- `database_preprocessor.py` keeps its own copy of `_now()`, identical to `_now` in `run_lifecycle.py`, which B2 kept private. Candidate for E1, which adds time fields to the run log.
- The `run_lifecycle.py` module docstring names `src/training/orchestrator.py`, which is now `orchestrator_training.py`.

## B3

- `expected` in the run log (→ E-block). Evaluation: all origin partitions × origin seeds, counting only partitions whose unit the evaluation dataset provides, since B3.2 skips the others. A partition without weights counts as failed, so the folder ends incomplete (proposed, to confirm). Preprocessing: `expected` is predicted from the dataset structure before reading any data, so it is approximate when runs are irregular.
- Evaluation on the same dataset as the origin (not scheduled). Today it evaluates the whole unit, so each fold also includes its training trials. The author's strategy is to split the results by train, validation and test ids, only when the evaluation dataset and the schema match the origin. It would reuse `train_idx`, `val_idx` and `test_idx` from `fold_metadata.json`. Precondition: `trial_idx` is the position in the unit's reading order (R17), so the indices mean the same thing only if the evaluation dataset keeps that order.
- `LeaveOneUnitOutSchema` casts unit ids to `int` (`leave_one_unit_out.py:68`) (→ I2). Text ids such as `"A"` are unsupported until then. "A works" in B3 covers holdout and k-fold only. Its `fold_metadata.json` stores the unit as `test_unit`, not `unit_id`, so evaluation cannot mirror a leave-one-unit-out origin either (B3.2 raises a clear error).
- Docs to update (a `docs/` PR). Evaluation supports k-fold origins since B3.2. `STAGE_CONTRACTS.md` section 4: remove the "origin schema must be a unit-level holdout" check and the known limit about k-fold origins; leave-one-unit-out stays unsupported (→ I2). `ARCHITECTURE.md` section 4: the same known limit. `IMPLEMENTATION_ORDER.md`, step C3: "clear errors for unsupported origins" is no longer a goal.
- Ordering of text unit ids (→ Phase 2). A folder listing sorts as `unit_1`, `unit_10`, `unit_2`, and a SQL `ORDER BY unit` does the same. Use the provider's order or a numeric cast. The code never relies on folder order.
- Preprocessing folders keep `subject_XX` with two digits. Preprocessing cannot import the partition-naming function (dependency rules), so the two naming rules differ. Known inconsistency, not scheduled.
- Module docstrings still name old matchers: `orchestrator_testing.py` (line 19, `WithinSubjectMatcher` -> `WithinUnitMatcher`) and `resolver.py` (line 6, `WithinSubjectMatcher`). Since B3.2 the matcher is `OriginPartitionMatcher`.

## C1

- `EEGDatabasePreprocessor._save_artifacts` writes to `artifacts/<stage>/subject_XX/<prefix>`, and neither the folder nor the prefix carries the session. With two sessions, the artifacts of the second session probably overwrite those of the first (read in the code, not executed). Fixing it changes the artifact layout in `STAGE_CONTRACTS.md` section 1. Not scheduled.
