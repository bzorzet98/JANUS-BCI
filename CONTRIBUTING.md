# Contributing to JANUS-BCI

How work is done in this repository: branches, commits and Pull Requests. This file is the single owner of these conventions. `CLAUDE.md` (the rules for Claude Code) and `docs/IMPLEMENTATION_ORDER.md` refer to it.

## 1. The cycle in short

1. Create a branch from an up-to-date `main`: one branch per step or purpose.
2. Make small commits, one purpose each.
3. Push the branch and open a Pull Request against `main`.
4. Merge it with **"Create a merge commit"**.
5. Update `main`, check again on it, and delete the branch.

Nothing is committed directly to `main`, not even documentation.

## 2. Branches

Format: `type/short-name`, lowercase, words separated by hyphens.

| Type | For |
|---|---|
| `feat` | New functionality |
| `fix` | Correcting an error |
| `docs` | Documentation only |
| `refactor` | Restructuring without changing behavior |
| `chore` | Small maintenance tasks (configuration, cleanup) |

`test` and `ci` exist in the standard and are not used for now, because automated tests and CI are out of scope.

Examples: `feat/run-log-training`, `fix/preprocessing-session-key`, `docs/add-prd`.

Branches of the roadmap steps use the names in `docs/IMPLEMENTATION_ORDER.md`. A branch named `archive/*` is a backup and is never merged.

## 3. Commits

### Format

```
<type>: <summary>

<body, optional>

Step: <id, optional>
```

| Rule | Detail |
|---|---|
| Type | One of `feat`, `fix`, `docs`, `refactor`, `chore` |
| Summary | English, imperative mood ("add", not "added"), lowercase after the colon, no final period, up to 72 characters |
| One purpose | If the summary needs "and", split it into two commits |
| Body | Optional. Explains **why**, since the diff already shows what. Lines of about 72 characters, separated from the summary by a blank line |
| Step | For roadmap steps, add `Step: E3` or `Step: E3.2` as the last line |
| Refactor | A `refactor:` commit does not change behavior. If it does, it is `feat:` or `fix:` |

### Examples

| Good | Bad |
|---|---|
| `feat: write START and END events to the training run log` | `update training` |
| `fix: include the session in the preprocessing partition key` | `fixed bug` |
| `refactor: build the experiment path in a single function` | `refactor and also fix paths and rename things` |
| `docs: add PRD, architecture and stage contracts` | `docs` |

A commit with a body:

```
fix: include the session in the preprocessing partition key

The key had no session, so in multi-session datasets the second session
would be skipped when the first one had already succeeded.

Step: C1
```

A quick test: reading only `git log --oneline`, the history of the project should be understandable.

## 4. Pull Requests

- **One Pull Request per step.** A large step has several stages: one commit per stage, one Pull Request after the last stage.
- **Title:** the same format as a commit summary (`type: summary`).
- **Description:** use the template, which GitHub loads automatically from `.github/pull_request_template.md`.
- **Merge with "Create a merge commit".** Never squash and never rebase: each run stores the commit hash it ran on, and both of those methods create new commits with different hashes, so the hashes stored by earlier runs would stop existing on `main`.
- **Never include** secrets, personal paths or machine names.
- If a Pull Request publishes commits that belong to no step (for example, older local commits), name them under "What changed".
- If a Pull Request is hard to describe in a few lines, it is probably too large and should be split.

### Template

```
## What changed
- <one bullet per change, with the class or file>

## Why
<the problem or requirement; reference the step id and the PRD requirements,
 e.g. "Step E3, R1, R2">

## How I checked it
1. <command> → <what I saw>

## Notes
- Behavior that changes:
- Behavior that stays the same:
- Known limits or follow-ups:
- Merge with a merge commit (each run stores the commit hash it ran on).
```

Fill "How I checked it" only after running the checks.

## 5. After the merge

```bash
git switch main
git pull origin main
git branch -d <branch>
git fetch --prune
```

Run the step's check again on `main`.

To undo a whole step, including all its stages, revert its merge commit: `git revert -m 1 <merge-commit>`.

## 6. Repository settings

In *Settings → General → Pull Requests*:

- Allow merge commits.
- Turn off squash merging and rebase merging.
- Turn on automatic deletion of head branches.

## 7. Runs and reproducibility

Production runs come from a commit on `main` with a clean working tree. Each run records the commit it ran on, and a dirty working tree means that commit does not contain the code that ran.

## 8. Working with Claude Code

Claude Code writes code only, one step at a time. The author runs the checks, commits, pushes and merges. The rules are in `CLAUDE.md`.