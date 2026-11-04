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

## A3

- Evaluation in the dummy-project uses the same dataset and session as training, and evaluates the whole unit, training trials included. This is a known limit (`STAGE_CONTRACTS.md`, section 4), so the evaluation accuracy means nothing there.
- Relaunching the HP search appends the rows of `hp_combinations.csv` again (→ G2). The combinations themselves report that there is nothing to do.
- `run_hp_search.py` uses underscores in `--hp_search_config` and `--base_config`, and has no `--force` or `--expand` to forward (→ G1).
- `TestingOrchestrator` builds the origin path with `project_config.project_name`, while `TrainedModelResolver` reads `model_source.project_name` (→ C3). The dummy evaluation config sets both to `dummy-project`.
- `TestingOrchestrator.run` writes `config.json` and `.git_commit` before resolving the weights. If `model_source` points to a folder that does not exist (for example a typo in the label), it logs a warning about the origin schema and an INFO line ("nada para hacer"), then ends without an error and leaves the record behind. Related to C3.
- `TestingOrchestrator.run` creates the output folder before the identity check, so a rejected run leaves an empty folder (→ F1).
- `TestingOrchestrator._run_replicate` writes `config.json` and `.git_commit` inside each partition folder, once per replica, so every unit folder (`unit_01`, ...) holds a copy of the config. `STAGE_CONTRACTS.md` section 4 asks for one record per execution. Author's decision: a future step removes the per-partition copies and keeps a single `config.json` in the label folder. No step in the order names it yet.
- `TestingOrchestrator.run` builds partition names with its own `f"unit_{u:02d}"` (→ B3).
- `TestingOrchestrator` decides what to resume from the `.mlflow_run_id_<seed>` markers (→ D3).
- The MLflow experiment of an evaluation is `project/strategy/recipe/dataset/label`, while training uses `project/strategy`.
- The HP search config has no `script_type`, although `PROTOCOL.md` section 5 asks for it in every config (and section 2 lists `hypersearch` as future). The orchestrator does not read it (→ G1, which validates the search config).
