# Training run 20260903T005036Z_gemini_incumbent_robustness_v1

Status: `aborted`

The pre-run guard stopped execution before data loading or model fitting because
`training_run_history.py create` recorded `git_dirty=true`. The worktree had been
independently verified clean immediately before creation; the helper created the
untracked run directory before measuring Git state, so its recorded value was wrong.

No diagnostic computation ran and no operational artifact changed. A corrected
helper and a new immutable run are required.
