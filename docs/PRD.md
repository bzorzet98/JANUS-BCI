# JANUS-BCI: Product Requirements Document (draft v1)

## 1. Context and motivation

JANUS-BCI began during a PhD in brain-computer interfaces, out of a tension between two needs: prototyping quickly, and producing rigorous, reproducible results for publication. The goal is to standardize the author's own way of working so that less time is lost on tracking and file structure, work can be reused across related projects, and the effort carries over to other fields. The repository is also meant to be shown as a portfolio piece.

## 2. Users

| User | Needs |
|---|---|
| Author (primary) | A consistent workflow for prototyping and for scientific production |
| External readers (interviewers, future public readers) | Understand the repository without the author's help |
| Claude Code (implementer) | Clear boundaries and contracts. It generates code only: the author runs commands and tests |

## 3. Problems to solve

- Prototyping and production corrupt each other.
- Losing track of which preprocessing, model or seed produced a result.
- Long sweeps fail and the failure is noticed only by browsing folders. Run status used to be tracked by hand in a spreadsheet.
- Reusing code across fields means rewriting infrastructure.
- Reusing data and results across related projects.
- Trusting what is delegated to a code assistant.

## 4. Scope

**In scope.** The four existing stages (preprocessing, training, hyperparameter search, evaluation) hardened with run tracking and re-execution rules; a run-status watch; a signal-metrics engine and a per-project results database (Phase 2); documentation; a demonstration project.

**Out of scope for now:**
- Automated tests, CI, and backtests against old projects.
- Statistics and figures as a repository stage (they belong to each project).
- A global database across projects, and any GUI.
- Per-project subject tags or cohorts.
- A second domain, and a domain-adaptation guide.
- Transformations applied at training time.
- Bit-exact reproducibility across different hardware.

## 5. Design principles

High cohesion and low coupling; DRY; KISS and YAGNI; SOLID; abstraction and encapsulation. Documentation states where each principle holds and where it does not.

**Rule for what to fix now.** A finding is fixed in Phase 1 only if (a) the new work touches it, (b) it can produce wrong results silently, or (c) it defines a contract that Phase 2 will read. Everything else is documented as known debt. Without tests, large refactors are deferred until the demonstration project exists as a safety net.

## 6. System overview

- **Independent stages.** Preprocess, train (including hyperparameter search), evaluate, and, in Phase 2, compute signal metrics. A project uses the subset it needs.
- **The disk is the source of truth.** Tables and databases are derived and can be rebuilt.
- **A folder is an experiment; an execution is each launch on it.** The folder is the path `project/strategy/recipe/database_session/label`.
- **The sandbox is the entry point for new ideas.** An idea is validated there, then integrated into `src` in its own branch. Everything in `src` is modifiable, and the user is responsible for the change.
- **A stage writes only inside its own folder.** It may read other folders, never write to them. No stage reads the derived tables to decide what to do.
- **Where things live:** code, configs and run tables in the repository; preprocessed data, results, weights and databases outside it, per machine.

## 7. Phase 1 requirements

### 7.1 Run tracking

| ID | Requirement |
|---|---|
| R1 | Each experiment folder keeps a **run log** with one START and one END event per execution. START records machine, mode (normal, expand or force), config name, commit, dirty-tree flag, Docker image, expected work units and the preprocessing snapshot. END records the exit (ok, error or interrupted) and the done and failed counts. |
| R2 | END is written in a single place, around the run call of each entry point, so it is written after errors and Ctrl+C. A killed process leaves an open START. |
| R3 | **Status is derived, never stored.** From the last START: no END means *in progress*; exit error means *failed*; done equals expected with no failures means *finished*; anything else means *incomplete*. An open run can be closed by hand by appending an END with exit error. |
| R4 | A **preprocessing snapshot** is taken at START by reading the preprocessing folder's log (name, dataset, config name, status, last end time). It is never updated afterwards. |
| R5 | Reproducibility record: the dirty-tree flag is saved, a missing git binary is tolerated, the Docker image can come from an environment variable, and the config file name is recorded. |

### 7.2 Re-execution

| ID | Requirement |
|---|---|
| R6 | The new config is compared with the stored one, ignoring `description` and `orchestrator.device`. This now applies to preprocessing too. |
| R7 | If the config differs and no flag is given, the run **stops** and prints a field-by-field diff. It writes nothing, so a rejected run never leaves an open START. |
| R8 | `--expand` keeps what exists and runs what is missing. It records the diff in START. Added subjects, added seeds at the end of the list, and added metrics are accepted. Removed or reordered seeds are rejected. *(proposed)* |
| R9 | `--force` redoes everything with the new config. It deletes results and progress but keeps the log. It replaces `--overwrite-existing`. Using `--force` and `--expand` together is an error. *(proposed)* |
| R10 | Write order: compare config, write `config.json`, take the snapshot, write START, work, write END. |
| R11 | Progress is tracked per (partition, seed), and the expected count is written at START. |
| R12 | Each replica has its own batch generator, derived from the fold, so every seed of a fold sees the same batch order. Adding a seed later does not change the others. |
| R13 | When a seed fails, the remaining seeds of that partition continue and the folder ends as incomplete. *(proposed)* |

### 7.3 Validation and contracts

| ID | Requirement |
|---|---|
| R14 | **Config validation before running:** script type, required fields, non-empty label, and consistency of database name and session. Errors are clear and nothing is written. |
| R15 | `unit` is a text key (`1`, `2`, `A`, `B`). Order comes from the data provider. |
| R16 | Partition folder names come from a single function. Numeric ids keep zero-padding so existing data stays readable. *(proposed)* |
| R17 | `trial_idx` is the position in the unit's reading order. Every output file carries its partition and seed. *(proposed)* |

### 7.4 Watch

| ID | Requirement |
|---|---|
| R18 | One command, run by the user, reads only the run logs and folder sizes. It regenerates one CSV per project (one row per experiment folder) and a general preprocessing table in the repository (folder name to be defined). |
| R19 | It is incremental: a folder is re-read only when its event count changed. A rebuild option reprocesses everything. It prints runs left open, with id and start time. |
| R20 | The project CSV also holds the preprocessing snapshot, a "preprocessing changed since" flag, and a "comes from" column with references to other runs. |

### 7.5 Fixes in scope

- One function builds the experiment path (currently copied in five places).
- Lifecycle helpers that three modules use are made public.
- The preprocessing partition key includes the session (a probable silent skip of session 2 in multi-session datasets).
- Hyperparameter search gets the same flags and config validation as the other stages, with consistent flag naming, and stops duplicating its combinations CSV.
- Testing reads the origin project name from one source.
- Testing from a k-fold or leave-one-unit-out origin fails with a clear message.
- Leave-one-unit-out does not save its split indices, so every partition would fail. It is fixed if the author uses it, and declared unsupported otherwise.

### 7.6 Demonstration

| ID | Requirement |
|---|---|
| R21 | A `dummy-project` with synthetic data runs preprocess, train and evaluate end to end. It shows a failed run, a killed run and a config-change rejection recorded correctly, and shows the watch twice yielding the same result. |

Each requirement is accepted when its step in the dummy-project works.

## 8. Phases 2 and 3 (open)

- **Phase 2, solved together.** A signal-metrics engine, independent of the training engine, with three levels (unit, trial, channel), the level as a parameter of each metric, and artifacts for outputs that are not metrics. A per-project SQLite database, rebuilt from the output files, with tables for model metrics, predictions, folds, signal metrics per level and artifacts.
- **Phase 3.** MLflow on the server and on the PC, and synchronization of folders between machines.

## 9. Definition of done

The repository is done when it has:
1. Clear, concise documentation of the boundaries and the connection between stages.
2. The run tables.
3. The dummy-project proving that table linkage works and that broken runs are recorded.

"Reproducible" means regenerable: code, configs and run tables travel with the repository. Data, results and weights do not, and are linked separately if needed.

## 10. Risks

- Phase 1 changes the lifecycle of all four stages. Without tests, the dummy-project is the only safety net.
- Scope has grown, so each requirement closes on its own step before the next begins.
- Findings about runtime behavior come from reading code and were not executed.

## 11. Open questions

1. Does a failed seed let the others continue (R13)?
2. Are `--expand` and `--force` mutually exclusive, and are removed or reordered seeds rejected (R8, R9)?
3. Folder name and location of the general preprocessing table.
4. Name of the new script type for signal metrics.
5. How MLflow logging relates to synchronization (Phase 3).
6. Does leave-one-unit-out get fixed or declared unsupported?
