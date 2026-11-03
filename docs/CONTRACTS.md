JANUS-BCI: Stage Contracts (draft v1)

Status. Written from design decisions and a read-only audit of main. Runtime behavior was inferred by reading code, not executed. Requirement ids (R1 to R21) refer to the PRD.

0. Rules shared by all stages
0.1 Commands and flags
Command	Flags
run_preprocessing.py	--config, --docker-image, --progress {rich,none}, --force, --expand
run_training.py	the same, plus --device
run_testing.py	the same, plus --device
run_hp_search.py	--hp-search-config, --base-config, --device, --docker-image, --progress, --force, --expand
--overwrite-existing is removed. --force replaces it (R9).
--force and --expand together are an error (proposed).
Flag names in run_hp_search.py use hyphens like the others (today --hp_search_config and --base_config use underscores) (proposed).
0.2 What every execution does
 ① validate config ───────────── invalid ─► stop, write nothing        (R14)
 ② compare with the stored config.json
 ③ apply the flag rules ──────── rejected ─► stop, write nothing        (R6 to R9)
 ④ write config.json
 ⑤ read the source folders (read-only) and build the source references (R4)
 ⑥ write START
 ⑦ work; each failure is recorded and the run continues
 ⑧ write END, always (also on error or Ctrl+C), around the run call    (R2)
Comparison. The whole config is compared, ignoring description and orchestrator.device. This now holds for preprocessing too.
Writes. A stage writes only inside its own folder. Other folders are read-only.
No derived table is read to decide what to do.
0.3 The run log

START and END events, as in the architecture document, with one addition.

Source references (proposed extension of R4). START holds a list of references, one per folder the stage reads: kind, folder relative to a named root (data or results), status, last end time, config name. The preprocessing snapshot is the reference of kind preprocessing. The origin training of an evaluation is a reference of kind training. This list is what fills the “comes from” column of the watch.
Reproducibility. .git_commit and .docker_image stay as files. The dirty flag and the config file name go into START. The Docker image comes from --docker-image or an environment variable. A missing git binary is tolerated (R5).
Status is derived from the last START and its END (R3).
0.4 Validation before running (R14)

All stages check that the script type is the expected one, that required fields exist, and that old-format keys give a clear error. Training and evaluation also check a non-empty label and that database_name and database_session match the dataset block.

1. Preprocess

Purpose. Turn a raw dataset into a reusable preprocessed folder.

	
Command	run_preprocessing.py --config <file>
Config	database (class, module, kwargs: sessions, subjects, data_to_load), preprocessing_name, preprocessing_pipeline (stages), script_type, description
Folder	<DATA_ROOT>/preprocessed/<preprocessing_name>/<dataset code>/
Reads other folders	None. The raw data comes from the loader
Source references	None
Work unit	One run of one subject: (session, subject, context, run)
Expected	Predicted from the dataset structure before reading any data. It is approximate when runs are irregular

What it writes

File	Content	Change
config.json, .git_commit, .docker_image	Reproducibility record	—
Run log	START and END	(new)
dataset_description.json	Manifest: subjects, sessions, sampling rate, channel names, event ids, and later n_times, tmin, tmax	—
run_registry.csv	Per run: partition, subject, session, context, run id, status, times, output path	Partition key includes the session (new)
stage_metrics.csv and detail tables	Per run and stage	—
artifacts/<stage>/subject_XX/…	Heavy files such as ICA	—
<session>/subject_XX/<prefix>.npz	Data and labels (or _epo.fif, _raw.fif)	—

Re-execution

Config change. Today only the pipeline is compared and a change to the database block overwrites config.json silently. Now the whole config is compared, so adding subjects or sessions needs --expand.
--force deletes everything generated (data, artifacts, CSVs, manifest) and keeps the log. Today the overwrite flag deletes only the CSVs and leaves old data in place, which is a silent inconsistency.
--expand with a pipeline change is allowed but leaves a folder processed with two pipelines. The diff is recorded, and the documentation recommends --force.
Failure. A failed run is recorded as failed and the sweep continues, so the folder ends incomplete (proposed).

Not used here. MLflow.

Demonstration step. Preprocess a synthetic dataset of three units. Show that adding a fourth unit is rejected without --expand, and accepted with it.

2. Train

Purpose. Train one model per partition and seed, and save outputs.

	
Command	run_training.py --config <file>
Config	project_config (project, strategy, recipe, database name, session, label), databases, data_provider, schema, orchestrator (model_init_seed, device), model, optimizer, criterion, callbacks, torch_dataset, torch_dataloader, trainer, metrics_strategy
Folder	Experiment folder <project>/<strategy>/<recipe>/<database_session>/<label>/
Reads other folders	The preprocessing folder’s log, to take its snapshot
Source references	One of kind preprocessing
Work unit	One (partition, seed) pair
Expected	Partitions × seeds, written at START

Seeds. model_init_seed is a list, or {n_seeds, seed}, which expands the same way for the first seeds when n_seeds grows. Each replica has its own batch generator derived from the fold, so all seeds of a fold see the same batch order and the seed changes only the initial weights (R12).

Locating the preprocessing folder. The training engine knows nothing about EEG. The dataset object exposes a small method that returns its source folder, and the orchestrator calls it only if it exists. If it does not, the snapshot is empty (proposed; the method name is fixed at implementation).

What it writes

File	Content	Change
Reproducibility record, config.json, run log	As above	Log (new)
script_progress.csv	Status per partition and seed	Seed column (new, R11)
metrics_results.csv	split, metric_name, value, partition, model_init_seed. Test metrics plus training duration	—
<partition>/fold_metadata.json	Unit, train_idx, val_idx, test_idx, batch seed, bin_to_class	—
<partition>/outputs/seed_N_outputs.csv	trial_idx, split, y_true, y_pred, proba_*	Carries partition and seed (new, R17)
<partition>/outputs/seed_N_history.csv	Loss per epoch (DL only)	—
<partition>/weights/seed_N_best.pth	State dict, only if the config has the checkpoint callback	—

Re-execution

Same config. Pending (partition, seed) pairs run, and a pair interrupted midway is redone from the start. “Already done” is decided by progress, not by the existence of the folder.
--expand. Added subjects, seeds appended to the list, and added metrics run only what is new. Removed or reordered seeds are rejected (proposed).
--force. Deletes results and progress, keeps the log.
Failure. A failed seed does not stop the next seeds or partitions, and the folder ends incomplete (R13, proposed).

MLflow. One run per (partition, seed) in the experiment <project>/<strategy>, with the folder fields as tags. How this relates to synchronization is Phase 3.

Known limits

proba_* is the raw model output, so it may be logits.
AUC with more than two classes over logits probably fails.
y_true holds the encoded labels.
Leave-one-unit-out does not save its split indices, so it is fixed if the author uses it and declared unsupported otherwise.

Demonstration step. Train three units with two seeds, where one unit fails on purpose. Show the folder as incomplete, then relaunch and show it as finished, then add a seed with --expand.

3. Hyperparameter search

Purpose. Run one training per combination of config blocks.

	
Command	run_hp_search.py --hp-search-config <file> --base-config <file>
Config	A base training config, and a search config with the axes (whole blocks such as optimizer), fixed overrides and development units
Folder	<project>/<strategy>_HPSearch/<recipe>/<database_session>/ with hp_combinations.csv. Each combination is a training folder labelled <label>_hp<hash>
Validation	The base config is validated as a training config, and the search config is validated (new, R14)

What it adds to training

File	Content	Change
hp_combinations.csv	combo_hash, axis_name, axis_value_json	Rows are written once and replaced on relaunch, not duplicated (new)
Search run log	START and END with expected = number of combinations, and done and failed counts	(proposed)

Re-execution

--force is forwarded to every combination.
--expand applies at the search level: new axis values add combinations, and existing combinations keep their config and are not rerun.
A combination whose config changed is rejected like any training.
The hash in the label covers only the axes. Two searches that differ only in fixed overrides or development units produce the same label and are rejected with a diff, so the author uses --force or another label.

Demonstration step. A search with two combinations where one fails. Show the search as incomplete and the two training folders.

4. Evaluate

Purpose. Evaluate the weights of a training on a dataset.

	
Command	run_testing.py --config <file>
Config	project_config (the author writes “test” in the strategy name), model_source (project, strategy, recipe, database, session, label, script type), databases, data_provider, dataloaders, metrics_strategy
Folder	Experiment folder of its own
Reads other folders	The origin training folder (config, progress, weights, fold metadata) and the preprocessing folder’s log
Source references	One of kind training and one of kind preprocessing
Work unit	One (partition, origin seed) pair
Expected	Matched partitions × origin seeds that have weights

Validation before running (R14, plus three checks)

The origin project name comes from one place: model_source.project_name, which defaults to the project of this config. It is used both for the origin path and for the weights resolver. Today they are read from different places.
The output folder must differ from the origin folder, because --force would delete the origin’s weights (proposed).
The origin schema must be a unit-level holdout. A k-fold or leave-one-unit-out origin fails with a clear message instead of aborting midway.

What it writes

File	Content	Change
Reproducibility record, config.json, run log	As above	Record written once per execution, not once per replica
script_progress.csv	Status per (partition, origin seed)	Seed column (new)
metrics_results.csv	split, metric_name, value, partition, origin_seed	—
.mlflow_run_id_<seed>	MLflow run id	Bookkeeping only

Resume is decided by progress, not by the MLflow marker files (proposed), so a stage never depends on MLflow to decide what to do.

Re-execution. There is no unit selector, so evaluation covers every unit the dataset provides. Restricting units goes through the databases block, and changing it now needs --expand. A failed pair is recorded and the run continues, so the folder ends incomplete.

Known limits

It evaluates the whole unit, not the origin’s test indices. If the dataset is the same as the origin’s, training trials are included.
It saves no per-trial predictions, which Phase 2 may add.

Demonstration step. Evaluate the training from step 2 on the same synthetic dataset. Show that pointing the output folder at the training folder is rejected.

5. Changes per stage
Stage	Main changes
Preprocess	Run log, whole-config comparison, --expand and --force, session in the partition key, --force that really cleans
Train	Run log, progress per (partition, seed), batch generator per replica, validation, flags, source references
HP search	Flags, validation, combinations table without duplicates, search-level log
Evaluate	Run log, one origin project name, folder checks, resume by progress, clear errors for unsupported origins
All	One function builds the experiment path (copied in five places today), lifecycle helpers made public, END around the run call
6. Open points
Does the search have its own run log (section 3)?
Is the preprocessing snapshot extended to source references of several kinds (section 0.3)?
Does evaluation resume from progress instead of MLflow markers (section 4)?
Name and shape of the method that returns a dataset’s source folder.
--expand with a pipeline change in preprocessing: allowed with a warning, or rejected?
Hyphenated flag names in the HP search command.
Is the “output folder differs from the origin folder” check added to the PRD list?