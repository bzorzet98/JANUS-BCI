# JANUS-BCI: Architecture (draft v1)

> **Status.** Written from design decisions and a read-only audit of `main`. Runtime behavior was inferred by reading code, not executed. `src/models` was not audited. Proposals not yet confirmed are marked *(proposed)*. Work of Phase 1 is marked *(new)* and Phases 2 and 3 are marked *(planned)*.

## 1. Overview

JANUS-BCI is a set of **independent stages** that communicate **only through the disk**. A stage reads what other stages left in folders, writes only inside its own folder, and records how it ran.

```
 raw dataset ──► PREPROCESS ──► preprocessed folder ──► TRAIN ─────────► experiment folder
                                        │                  ▲                  │
                                        │                  │ runs N trainings │
                                        │              HP SEARCH              ▼
                                        │                              EVALUATE ──► experiment folder
                                        ▼
                         SIGNAL METRICS (planned, Phase 2)
```

- A project uses the stages it needs. The order above is the automated flow, not an obligation.
- Everything else (tables, databases, MLflow) is **derived** from folders and can be rebuilt.

## 2. Code structure and dependency rules

```
   scripts/ (one command per stage)  ·  JSON configs
                         │
                         ▼
   ┌───────────────────────────────────────────────────────────┐
   │ EEG DOMAIN                                                │
   │  raw loaders · preprocessed reader · preprocessing        │
   │  models · concrete signal calculations (planned)          │
   └──────────┬──────────────────────────────────┬─────────────┘
    implements│ the unit protocol               │ loaded by name
              ▼                                  ▼ from the config
   ┌─────────────────────────────┐   ┌──────────────────────────┐
   │ TRAINING ENGINE             │   │ SIGNAL-METRICS ENGINE    │
   │ units · partitions ·        │   │ (planned) levels ·       │
   │ trainer · model metrics     │   │ progress                 │
   └──────────────┬──────────────┘   └─────────────┬────────────┘
                  └──────────────┬─────────────────┘
                                 ▼
   ┌───────────────────────────────────────────────────────────┐
   │ UTILITIES                                                 │
   │  per-machine paths · import by name · run lifecycle ·     │
   │  run log (new) · config diff (new) · reproducibility      │
   └───────────────────────────────────────────────────────────┘
```

| Package | Responsibility | Knows EEG |
|---|---|---|
| `src/utils` | Paths, import by name, run lifecycle, reproducibility, progress reporting | No |
| `src/training` | Unit protocol, split schemas, trainer and tester, model metrics, MLflow logging, orchestrators for train, evaluate and HP search | No |
| `src/eeg_datasets` | Raw loaders, `Subject`, `PreprocessedDataset`, adapter to the engine | Yes |
| `src/preprocessing` | Stage engine and preprocessing orchestrator | Yes |
| `src/models` | About 35 architectures, loaded by name | Yes |
| `scripts/` | One thin command per stage, plus the watch (new) | No |
| `projects/<name>/` | Configs, commands and project-specific code | Depends |
| `sandbox/` | Throwaway exploration, no tracking | — |

**Dependency rules.**
1. Imports go down only. Upward references exist only **by name inside a config** (`class_name`, `module_name`, `params`).
2. `src/utils` and `src/training` never import `src/eeg_datasets`, `src/preprocessing` or `src/models`.
3. `src/preprocessing` does not import `src/training`.
4. Nothing in `src/` imports `scripts/` or `projects/`.
5. The two engines are independent of each other. Each defines its own small protocol for "units", and the EEG adapter implements it structurally.

**Enforcement.** Rules 2 to 4 held in the files the audit read, and rule 1 was not checked inside `src/models`. They are conventions: the only check that existed (`verify_10`) is broken.

## 3. The experiment folder and the run log

**Folder versus execution.** The folder is an experiment. Each launch of a command on it is an execution.

```
<RESULTS_ROOT>/<project>/<strategy>/<recipe>/<database_session>/<label>/
```

- `label` is free text written by hand.
- Evaluation keeps the author's convention: "test" goes in the strategy name by hand.
- A single function builds this path *(new)*. Today it is copied in five places.

**What an execution leaves in its own folder**

| File | Content |
|---|---|
| `config.json` | The config of the last accepted execution |
| Reproducibility record | Commit, dirty-tree flag *(new)*, Docker image, config file name *(new)* |
| **Run log** *(new)* | One line per event: START and END |
| Progress | Status per (partition, seed) *(new granularity)* |
| Outputs | Metrics, predictions, fold metadata, history, weights, artifacts |

**Run log events.** The file name and format are fixed at implementation: one line per event, editable by hand.

| Event | Fields |
|---|---|
| START | execution id, time, machine, mode (normal, expand or force), diff (expand or force only), config name, commit, dirty flag, Docker image, expected work units, preprocessing snapshot |
| END | execution id, time, exit (ok, error or interrupted), units done, units failed |

**Status is derived, never stored:**

```
 last START has no END ────────────────────────► IN PROGRESS
 END exit = error ─────────────────────────────► FAILED
 done = expected and failed = 0 ───────────────► FINISHED
 anything else (including Ctrl+C) ─────────────► INCOMPLETE
```

A killed process cannot write END, so it stays *in progress*. The author closes it by appending an END with exit error, or by relaunching.

**Execution order (new).**

```
 ① validate config ─ invalid ─► stop, write nothing
 ② compare with stored config
 ③ apply flag rules (section 5) ─ rejected ─► stop, write nothing
 ④ write config.json
 ⑤ take the preprocessing snapshot (read-only)
 ⑥ write START
 ⑦ work ── failures per (partition, seed) are recorded, the run continues
 ⑧ write END, always (also on error or Ctrl+C)
```

A rejected execution writes nothing, so it can never appear as *in progress*. END is written in one place, around the run call of each entry point.

## 4. Stages

| Stage | Consumes | Leaves | Reads other folders |
|---|---|---|---|
| **Preprocess** | Raw dataset, pipeline in the config | `<DATA_ROOT>/preprocessed/<name>/<dataset>/`: manifest, per-subject data, `run_registry.csv`, stage metrics, detail tables, artifacts | No |
| **Train** | A preprocessed folder through the unit protocol | Experiment folder: `metrics_results.csv`, and per partition `fold_metadata.json`, `outputs/seed_N_outputs.csv`, `seed_N_history.csv`, `weights/seed_N_best.pth` | The preprocessing log (snapshot) |
| **HP search** | A base config and a search config | `<strategy>_HPSearch/` with `hp_combinations.csv`; each combination is a training folder labelled `<label>_hp<hash>` | As training |
| **Evaluate** | Weights of a training (`model_source`) and its own dataset | Experiment folder with `metrics_results.csv` | The origin training folder (config, progress, weights, fold metadata) and the preprocessing log |

The contract of each stage is in `STAGE_CONTRACTS.md`.

**Contracts fixed in Phase 1 because Phase 2 will read them (R15 to R17)**
- `unit` is a text key. Order comes from the data provider.
- The partition folder name comes from one function. Numeric ids keep zero-padding.
- `trial_idx` is the position in the unit's reading order. Every output file carries its partition and seed.
- Each replica has its own batch generator derived from the fold.

**Known limits, documented and not fixed in Phase 1.** Evaluation evaluates the whole unit, not the origin's test indices. It saves no per-trial predictions. It does not support k-fold or leave-one-unit-out origins.

## 5. Re-execution behavior

```
 Launch on an existing folder
          │
 stored config.json? ── no ──► new folder ─────────────────────► mode: normal
          │ yes
 config equal? (ignoring description and device)
          ├─ yes ─► run what is pending ─────────────────────────► mode: normal
          └─ no
              │
          which flag?
           ├─ none ────► STOP, print field-by-field diff, write nothing
           ├─ --expand ► keep what exists, run what is missing ──► mode: expand + diff
           ├─ --force ─► delete results and progress (keep the log),
           │             run everything ──────────────────────────► mode: force + diff
           └─ both ────► error
```

| Case | Behavior |
|---|---|
| Relaunch after a cut | Runs what is pending. A partition that failed or stopped midway is redone |
| `--expand` with added subjects, seeds at the end, or a new calculation | Runs only what is new |
| `--expand` with another change (e.g. learning rate) | Allowed. The diff is recorded and results of two configs may mix |
| Seeds removed or reordered | Rejected, even with `--expand` *(proposed)* |
| New label | New folder, unrelated to the old one |

## 6. The watch (new)

A command the user runs. It reads **only run logs and folder sizes**, never outputs, and never writes to experiment folders.

```
 list folders with a run log
        │
 folder changed? (its event count differs from the stored one, or it has no row)
   ├─ no ─► leave its row
   └─ yes ─► read the log ─► derive status ─► gather row data ─► replace its row
        │
 print: updated folders · open runs (id, start time) · date of this update
```

- **Output:** one CSV per project (one row per experiment folder, in the repository under `projects/<name>/`) and a general preprocessing table in the repository outside `projects/`. Rows are sorted by folder so git diffs stay small.
- **Row:** folder, script type, config name, status, done/expected, last machine, first start, last end, number of executions, size, snapshot, "preprocessing changed since" flag, "comes from" references, stored event count.
- **Change detection by event count, not by date.** `rsync -a` keeps old dates and would hide copied folders.
- **Properties:** running it twice gives the same result, and deleting the CSV and rebuilding gives the same result.
- **"Preprocessing changed since"** compares the snapshot's end time with the current row of the general table. The author decides whether the change matters.

## 7. Where things live

| What | Where | Travels with the repository |
|---|---|---|
| Code, scripts, preprocessing configs | repository | yes |
| Project configs and project CSV | `projects/<name>/` | yes |
| General preprocessing table | repository root area (name to define) | yes |
| Preprocessed data | `<DATA_ROOT>/preprocessed/` | no |
| Results, weights and outputs | `<RESULTS_ROOT>/...` | no |
| Per-project database (planned) | next to the project's results | no |

"Reproducible" means regenerable from code, configs, run tables and raw data. Weights are shared separately when needed. The tables hold no absolute paths, only references relative to a named root.

## 8. How new work enters

```
 sandbox idea ──► validated? ──► choose destination in src ──► own branch ──► Pull Request
   (no tracking)       │                                            │
                       └─ no ─► archive or delete                   └─ checked with the dummy-project
```

Everything in `src` can be modified under the user's responsibility. Branches follow `feat/`, `fix/`, `docs/`, `refactor/` and `chore/`, and are merged through a Pull Request with **"Create a merge commit"**. Squash and rebase merging are not used, because each run stores the commit hash it ran on and both create new hashes. The conventions for commits and Pull Requests are in `CONTRIBUTING.md`, and the order of work is in `IMPLEMENTATION_ORDER.md`.

## 9. Planned

**Phase 2, designed together.**
- **Signal-metrics engine**: independent of the training engine; metrics declared like any block (class, module, params) with the **level** (unit, trial, channel) as a parameter; one folder per data source and metric set; metrics as rows, other outputs as artifacts.
- **Per-project database**: derived from output files by its own command, separate from the watch. Proposed tables: model metrics, predictions, folds, signal metrics per level, and an artifact list. Updates replace one folder at a time.

**Phase 3.** MLflow on the server and on the PC, and folder synchronization. The existing `scripts/sync/` expects another layout and is replaced.

## 10. Principles: where they hold and where they do not

| Principle | Holds | Does not hold (known debt) |
|---|---|---|
| Cohesion | The lifecycle manager has one job. MLflow logging has its own class | The training orchestrator (about 500 lines) and the preprocessor mix several responsibilities |
| Coupling | Training does not know EEG | Paths copied in five places *(fixed in Phase 1)*. Evaluation depends on training file names |
| DRY | The lifecycle is shared | Duplicated setup between training and evaluation. Argument parsing repeated in four commands |
| KISS and YAGNI | HP search reuses the training orchestrator | Unused code: base callback, loss strategy, caching flag, empty regression metrics |
| S and O | Registries for stages, artifact savers and metrics | Type-based branches in the orchestrators |
| L and I | Three split schemas with one signature. A three-method unit protocol | The leave-one-unit-out fold lacks indices the training code reads |
| D | Schemas depend on the protocol | Orchestrators depend on concrete metrics, MLflow and resolver classes. `PATHS` creates folders on import |

Large refactors wait for the dummy-project to exist.

## 11. Glossary

| Term | Meaning |
|---|---|
| Experiment folder | `project/strategy/recipe/database_session/label` |
| Execution | One launch of a command on an experiment folder |
| Unit | A subject, or in another domain a patient, image or session. A text key |
| Partition | A unit or fold directory (`unit_01`, `unit_01_fold_k`, `loo_test-unit_01`) |
| Replica | One run with a model-initialization seed |
| Run log | Per-folder file of START and END events |
| Snapshot | Copy of a preprocessing folder's state, taken when an execution starts and never updated |

## 12. To verify

- Runtime behavior of the lifecycle and of the END placement around Ctrl+C.
- Imports inside `src/models`.
- Whether the sandbox naming rules agreed earlier still fit.