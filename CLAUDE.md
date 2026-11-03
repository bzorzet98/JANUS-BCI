# CLAUDE.md

Working rules for Claude Code in JANUS-BCI. This file says **how** to work. Facts about the system live in `docs/` (PRD, ARCHITECTURE, STAGE_CONTRACTS, IMPLEMENTATION_ORDER). Conventions for branches, commits and Pull Requests live in `CONTRIBUTING.md`. If a doc and this file disagree about a fact, the doc wins. About how to work, this file wins.

## 1. Roles

- You write code. You do not run anything.
- The author reads the plan, approves it, runs the checks, commits, pushes, opens and merges the Pull Request, and re-tests. Every one of those steps is theirs.
- Talk to the author in Spanish. Code, comments, docstrings, docs and commit messages are in English. New user-facing messages are in English. Do not translate existing messages unless the step touches them.

## 2. Hard rules

1. **No terminal commands of any kind:** no git, python, pip, tests, scripts, ls, cat or grep. Use only the file read, search and edit tools. If a step cannot be done without running something, stop and say so.
2. **No commits, branches, stashes, resets, pushes, merges or Pull Requests.** Never suggest working on `main`.
3. **Never say that code works or that tests pass.** You cannot run it. Write "written, not executed".
4. **Write nothing before the author says "go"** on a plan (section 3.2). Reading files is allowed at any time.
5. Do not read or write under DATA_ROOT, RESULTS_ROOT, `.env` or `src/models/` (unless the step is about models). Do not use the contents of `archive/*` branches unless the author copies a file out for you.
6. No new dependencies. Python 3.11.
7. Do not create documentation files unless the step asks for it.

## 3. The loop: one step, one purpose

Each step comes from `docs/IMPLEMENTATION_ORDER.md` (for example "Step E2"). One step is one branch and one purpose.

### 3.1 Start

Read the step and only the doc sections it cites. Read only the files you need. If you can read `.git/HEAD`, check the branch: if it is `main`, stop and tell the author. If you cannot read it, ask the author to confirm the branch.

### 3.2 Plan (before any edit)

Post the plan in the template of section 7 and **wait for "go"**. The plan is precise on purpose: the author audits it before any code exists.

For every class and every function the step creates or modifies, the plan says:

- **Create or modify**, the file, and **why** it is needed.
- **If it creates:** its single responsibility in one sentence, its public API with signatures, what it depends on, and who will call it. Also why a function is not enough (KISS), if it is a class.
- **If it modifies:** which methods change, what each does **today**, what it will do **after**, whether the signature changes, which callers are affected, and which methods it does **not** touch.
- If the step creates or modifies no class or function (config, scripts, docs), say so and name exactly what changes.

Open decisions listed by the step are asked here. Do not guess them.

### 3.3 Size check and stages

Count what the plan touches. A step is **LARGE** if any of these holds (the author may change these thresholds here):

- more than 2 classes are created or modified;
- a public signature used by another module changes;
- the flow of an existing method is restructured, not just extended;
- more than about 150 lines change in total;
- more than 5 files are edited.

When in doubt, split. For a LARGE step:

1. **First reply: the split only.** List the stages (`E3.1`, `E3.2`, ...), and for each one the classes it touches, in one line. Wait for approval of the split before writing any detailed plan.
2. **Each stage is independently auditable** and leaves the dummy-project running. Prefer this order inside a step: add new code that nothing calls yet, then switch the callers to it, then remove the old path.
3. **Each stage has its own cycle:** detailed plan, "go", code, handoff, and the author's check.
4. The stages share the step's branch. The author makes one commit per stage and opens one Pull Request after the last stage.

### 3.4 Code

Make the smallest change that satisfies the plan. If you find that the plan needs to change, stop and post the change as a new plan. Do not improvise.

### 3.5 Handoff

Deliver the block in section 8 and stop. Do not start the next step or stage.

### 3.6 Feedback

If the author reports a failure, fix it inside the same step or stage. Do not widen the scope.

### 3.7 After the merge

If the author asks for a post-merge check, reply with the verification list only.

**Stop and ask** if: a doc contradicts the code; a prerequisite step does not look merged; the step turns out to need something the plan did not foresee.

## 4. Scope and cost discipline

- Change only what the plan names. No drive-by fixes, renames, formatting or reorganizing.
- A `refactor/` step does not change behavior. A `feat/` or `fix/` step does not refactor unrelated code.
- If you notice something else, do not touch it. List it under "Noticed, not touched" in the handoff.
- Every step and every stage must leave the dummy-project running.
- Do not scan the repository. Do not restate the design. The plan is detailed about classes and functions and short about everything else.

## 5. Design principles

Apply them. Say in the plan and the handoff where you could not.

- **High cohesion.** A function or module does one thing. If you need "and" to describe it, split it.
- **Low coupling.** Imports go down only:
  - `src/utils` is imported by `src/training`, `src/eeg_datasets` and `src/preprocessing`.
  - `src/training` and `src/utils` never import `eeg_datasets`, `preprocessing` or `models`.
  - `preprocessing` never imports `training`.
  - Nothing in `src/` imports `scripts/` or `projects/`.
  - Cross-package references go by name in a config (`class_name`, `module_name`, `params`).
- **DRY.** Before adding a helper, check that none exists (experiment path, partition name, lifecycle helpers, flags). Never copy a path or naming rule: call the single function.
- **KISS and YAGNI.** The smallest thing that works. Do not create a class where a function is enough. No new abstraction until a second use exists. No speculative options, hooks, hashes or generations.
- **SOLID, in practice.** One responsibility per class. Extend through registries and config names rather than editing the engine. Engines depend on a small protocol, not on concrete classes. Keep interfaces small.
- **Abstraction and encapsulation.** Expose a small public surface. Do not import underscore-prefixed names from another module. If a helper is shared, make it public.
- **Refactoring.** Restructure without changing behavior, in its own step.
- **Tie-breaker.** KISS and YAGNI beat generality. If a step forces a tradeoff against a principle, say it in one line.

## 6. System invariants

- A stage writes only inside its own experiment folder. It may read other folders, read-only. It never reads derived tables (CSV or database) to decide what to do.
- Status is derived from the run log, never stored.
- Order of an execution: validate config, compare with the stored config, apply flag rules, write `config.json`, take source references, write START, work, write END. A rejected execution writes nothing.
- The experiment path and the partition names each come from one function.
- Error messages say what is wrong and how to fix it, and are raised before anything is written.
- Flags: `--force` and `--expand`, never both. `--overwrite-existing` is removed.

## 7. Plan template (before "go")

```
PLAN — Step <id> [stage n/N] — branch <name>
Nothing is written until the author replies "go".

Purpose
  <one or two lines>

Size check
  Classes created: <n> · modified: <n> · functions created/modified: <n>
  Files edited: <n> · public signatures changed: <n>
  Verdict: SMALL | LARGE (→ send only the split, section 3.3)

Classes and functions
  [CREATE] <ClassName> — <file>
    Why: ...
    Responsibility: <one sentence>
    Public API: <method signatures>
    Depends on: <imports>
    Used by: <callers>
    Why not a function: ...

  [MODIFY] <ClassName>.<method> — <file>
    Why: ...
    Today: <what it does now>
    After: <what it will do>
    Signature: unchanged | <old> → <new>
    Does not touch: <other methods of the class>
    Callers affected: ...

  [CREATE|MODIFY] <function> — <file>
    (same fields)

  [NONE]  — only for a step that creates or modifies no class or function:
    name exactly what it changes (config, script, doc).

Files read: ...        Files edited: ...

Behavior
  Changes: ...
  Stays the same: ...

Principles
  Holds: ...   Tradeoff (if any): ...

Risks
  What could break and where.

Open decisions
  <questions for the author, one per line>

Preview of the check (the author runs it after the code)
  1. <command> → expected outcome
```

## 8. Handoff block (end of every step or stage)

```
STEP <id> [stage n/N] — branch <name>
Written, not executed.

As planned vs as written
  Same as the approved plan: yes | no
  Differences: <each one, with the reason>

Changed
  - <file>: <one-line reason>
  - Classes and functions: <name> [created|modified] — <methods touched>

Behavior
  Changes: ...
  Stays the same: ...

Principles
  Holds: ...   Tradeoff (if any): ...

Check (the author runs)
  1. <command> → expected outcome
  2. ...

Noticed, not touched
  - ...

Suggested commit
  <type>: <summary in English>
  (format and rules: CONTRIBUTING.md)

Git (the author runs)
  git add <files>
  git commit -m "<type>: <summary>"
  git push -u origin <branch>
  # large step: repeat the cycle for the next stage on the same branch
  # after the last stage and its check, open a Pull Request:
  #   base main, compare <branch>, merge with "Create a merge commit"
  # then, on the author's machine:
  git switch main
  git pull origin main
  # run the check again on main
  git branch -d <branch>
  git fetch --prune

Pull Request draft (the author pastes it on GitHub)
  Title: <the suggested commit, or the step summary>
  What changed: <bullets, one per class or file>
  Why: <the step id and the PRD requirements it satisfies>
  How I checked it: left for the author, from the "Check" list above
  Notes: <behavior that changes, behavior that stays the same, known limits>
```

## 9. Git conventions

The conventions for branches, commits and Pull Requests are in `CONTRIBUTING.md`. They are not repeated here. In short: one branch per step, `type: summary` commits in English, and a Pull Request merged with a merge commit, never a squash or a rebase.