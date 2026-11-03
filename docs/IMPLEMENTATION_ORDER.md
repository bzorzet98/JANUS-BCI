# JANUS-BCI: Phase 1 Implementation Order (draft v1)

> **Status.** Derived from the PRD (requirements R1 to R21), the architecture and the stage contracts. Phases 2 and 3 get their own order later. Runtime behavior was inferred by reading code, not executed. How Claude Code works on each step is defined in `CLAUDE.md`.

## 1. How a step runs

Each step is **one branch, one purpose, one cycle**:

1. The author creates the branch from `main`, using the name in the tables below.
2. The author tells Claude Code: *"Step E2 from docs/IMPLEMENTATION_ORDER.md. Follow CLAUDE.md."*
3. Claude Code posts a **detailed plan** and writes nothing until the author replies "go". For every class and function it creates or modifies, the plan says why, what it does today, what it will do after, its signature, and who is affected (`CLAUDE.md`, section 3.2).
4. Claude Code writes code only, and delivers a handoff block that compares the written code with the approved plan.
5. The author runs the check, commits, pushes, merges with a merge commit, runs the check again on `main`, and deletes the branch.

**Large steps are split into stages.** If a step creates or modifies more than two classes, changes a public signature, restructures an existing method, or is large in lines or files (thresholds in `CLAUDE.md`, section 3.3), Claude Code first proposes the stages and waits for approval. Each stage (`E3.1`, `E3.2`, ...) gets its own plan, "go", code, handoff and check. The stages share one branch, with one commit per stage, and the author merges once after the last stage. Every stage leaves the dummy-project running. Inside a step, new code is added first, then callers are switched to it, then the old path is removed.

*Steps likely to need stages* (an expectation, decided at plan time): B1, D2, D3, E3, E4, F3, F4, F5, H1.

**Every step leaves the dummy-project running.**

**Undoing a step.** Each step is one merge commit, so `git revert -m 1 <merge-commit>` undoes the whole step, including its stages.

**Before the first step**

- The remote points to the new account, and the old token is revoked.
- `main` is clean. Every step builds on it, so the first push of `main` publishes the two local commits (`fa4b330` and `43c6867`). They are the base we agreed to keep.

## 2. The order

*Needs* lists the steps that must be merged first. The R-ids refer to the PRD.

### A. Safety net: the dummy-project (R21, partial)

| Step | Branch | What changes | Needs | The author checks |
|---|---|---|---|---|
| A1 | `feat/dummy-raw-and-preprocess` | Synthetic raw dataset (3 units, 2 sessions, tiny) loadable by name, a preprocessing config, a README with the command. Uses the current code | — | Run preprocessing. Note what the current code does with the second session |
| A2 | `feat/dummy-train` | Tiny CPU model loadable by name, a training config with the checkpoint callback and 2 seeds | A1 | Run training: outputs and weights per unit and seed |
| A3 | `feat/dummy-eval-hp` | An evaluation config and an HP search config | A2 | Both run without errors |

*Note for A1.* To let Claude Code read the old synthetic dataset without switching branches, the author runs: `git show archive/lineage-wip:tests/fixtures/fake_bci.py > /tmp/fake_bci_archive.py`.

### B. Foundations: no behavior change

| Step | Branch | What changes | Needs | The author checks |
|---|---|---|---|---|
| B1 | `refactor/experiment-path` | One function builds the experiment path (copied in five places today) | A3 | Same folder tree as in block A |
| B2 | `refactor/lifecycle-public-helpers` | The lifecycle helpers used by three modules become public | B1 | Same behavior |
| B3 | `refactor/partition-naming` | One function builds partition names. `unit` is a text key. Numeric ids keep zero-padding (R15, R16) | B2 | Same partition folders. A non-numeric unit id such as `A` works |

### C. Silent errors

| Step | Branch | What changes | Needs | The author checks |
|---|---|---|---|---|
| C1 | `fix/preprocessing-session-key` | The preprocessing partition key includes the session | B3 | Both sessions are preprocessed |
| C2 | `fix/git-and-docker-record` | A missing git binary is tolerated. The Docker image can come from an environment variable (part of R5) | B2 | A run with git off the PATH completes |
| C3 | `fix/evaluation-guards` | One origin project name. Output folder must differ from the origin folder. Clear errors for unsupported origins | B1 | An output folder equal to the origin folder is rejected |
| C4 | `feat/config-validation` | A shared validator before running, for preprocessing, training and evaluation (R14) | A3 | A config with a missing field or an old-format key stops with a clear message and writes nothing |

### D. Seeds and progress

| Step | Branch | What changes | Needs | The author checks |
|---|---|---|---|---|
| D1 | `fix/generator-per-replica` | Each replica has its own batch generator, derived from the fold (R12) | A2 | Seed 22 gives identical outputs trained alone or after seed 11 |
| D2 | `feat/training-progress-per-seed` | Progress per (partition, seed). A failed seed does not stop the others (R11, R13) | D1, B3 | Force a failing seed, relaunch, and only the pending pairs run |
| D3 | `feat/evaluation-progress-per-seed` | The same for evaluation. Resume decided by progress, not by MLflow markers | D2 | A failed pair is recorded and the run continues |
| D4 | `feat/output-columns` | Output and history files carry partition and seed (R17) | D2 | The columns are present |

### E. Run log

| Step | Branch | What changes | Needs | The author checks |
|---|---|---|---|---|
| E1 | `feat/run-log-core` | The log module and the status derivation (R1, R3) | B2 | A small check script derives the four statuses from sample logs |
| E2 | `feat/run-log-preprocessing` | START and END in preprocessing, END written around the run call. START carries the dirty flag and the config name (R1, R2, R5) | E1, C2 | A normal run, a forced error, Ctrl+C and a kill leave the expected log |
| E3 | `feat/run-log-training` | The same for training. Expected counts are (partition, seed) pairs | E1, D2 | The same four cases |
| E4 | `feat/run-log-evaluation` | The same for evaluation | E1, D3 | The same four cases |
| E5 | `feat/source-references-training` | A dataset method returns its source folder. START holds the preprocessing snapshot (R4) | E2, E3 | Rebuilding the preprocessing later leaves the snapshot unchanged |
| E6 | `feat/source-references-evaluation` | START holds the training and preprocessing references | E4, E5 | Both references are present |

### F. Re-execution rules

| Step | Branch | What changes | Needs | The author checks |
|---|---|---|---|---|
| F1 | `fix/config-compare-rule` | Training and evaluation: ignore `description` and `device`, stop with a diff, no prompt, write nothing on rejection (R6, R7, R10) | C4 | Changing the learning rate stops without writing. Changing the description continues |
| F2 | `feat/force-flag` | `--force` replaces `--overwrite-existing` and keeps the log (R9) | E3, E4, F1 | `--force` redoes everything and the log keeps its history |
| F3 | `feat/expand-training` | `--expand`: added units and seeds at the end run only what is new. Removed or reordered seeds are rejected. The diff goes in START (R8) | D2, E3, F2 | Adding a seed at the end runs only that seed |
| F4 | `feat/expand-evaluation` | The same for evaluation | D3, E4, F3 | The same idea |
| F5 | `feat/preprocessing-expand` | Preprocessing: whole-config comparison and `--expand` (R6, R8) | E2, F3 | Adding a unit needs `--expand` and runs only that unit |
| F6 | `fix/preprocessing-force-clean` | Preprocessing `--force` deletes everything generated and keeps the log (R9) | F5 | No old data is left behind |

### G. HP search

| Step | Branch | What changes | Needs | The author checks |
|---|---|---|---|---|
| G1 | `fix/hp-search-flags-and-validation` | The same flags with hyphenated names. `--force` and `--expand` forwarded. Validates the base and search configs | F2, F3, C4 | An invalid base config is rejected |
| G2 | `fix/hp-combinations-no-duplicates` | The combinations file is replaced on relaunch, not duplicated | G1 | Relaunching does not add rows |
| G3 | `feat/run-log-hp-search` | A search-level log: expected = combinations. Optional | E1, G1 | One failed combination leaves the search *incomplete* |

### H. Watch

| Step | Branch | What changes | Needs | The author checks |
|---|---|---|---|---|
| H1 | `feat/run-watch` | The command, the CSV per project, change detection by event count, status derivation, open runs (R18, R19) | E4 | Four statuses appear. Running it twice, or deleting the CSV and rebuilding, gives the same result |
| H2 | `feat/preprocessing-table-and-references` | The general preprocessing table, the "changed since" flag, the "comes from" column (R20) | E5, H1 | The flag flips after rebuilding the preprocessing |

### I. Close

| Step | Branch | What changes | Needs | The author checks |
|---|---|---|---|---|
| I1 | `feat/dummy-demo` | The demonstration script: a failed run, a killed run, a config rejection, and the watch twice (R21) | all above | The whole story runs |
| I2 | `fix/leave-one-unit-out` or `chore/declare-unsupported` | Fix or declare unsupported (decision) | — | — |
| I3 | `chore/remove-unused-code` | Remove only the unused code the audit listed. Optional | I1 | The dummy-project still runs |

Requirement R10 (the write order) is satisfied once F1 and E5 are merged. The author may merge two small steps (for example B1 and B2) into one cycle. The default is one per cycle.

## 3. Decisions needed before a step

| Before | Decision | Proposal |
|---|---|---|
| C1 | Existing preprocessed folders have keys without the session | Reprocess with `--force`, or migrate. Decide at the step |
| E5 | Name and shape of the dataset method that returns its source folder | A small optional method that the orchestrator calls only if present |
| F2 | `--force` together with `--expand` | An error |
| F3 | Removed or reordered seeds | Rejected, even with `--expand` |
| F5 | `--expand` with a pipeline change in preprocessing | Allowed with a warning |
| G3 | Does the search have its own log? | Yes, optional |
| H2 | Name and location of the general preprocessing table | `registry/` in the repository root (tentative) |
| I2 | Leave-one-unit-out | Fix if the author uses it, otherwise declare it unsupported |
