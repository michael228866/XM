# XM GOLD Research Policy

## 1. Primary strategy objective

The primary objective of this repository is:

MAXIMIZE ROBUST EXECUTABLE TRADING FREQUENCY
WHILE PRESERVING REQUIRED STRATEGY QUALITY.

The current operational problem is not insufficient peak win rate.

The current problem is that the mature `gemini.py` dedicated-long strategy trades too infrequently.

Therefore:

QUALITY IS A CONSTRAINT.

FREQUENCY IS THE PRIMARY OPTIMIZATION OBJECTIVE AFTER THE QUALITY CONSTRAINT IS SATISFIED.

Do not optimize win rate indefinitely at the expense of large reductions in executable trade frequency.

A candidate with:

* 62% realized WR
* PF 1.18
* positive expectancy
* materially higher trades/day

may be preferable to a candidate with:

* 70% realized WR
* PF 1.35
* very low trades/day

provided both are comparably robust and both pass all mandatory economic and chronological requirements.

---

## 2. Optimization hierarchy

Use the following lexicographic optimization hierarchy.

### Stage 1 — Robust economic viability

A candidate must first preserve:

* positive robust expectancy;
* positive Mean-R;
* positive PnL;
* PF > 1;
* positive break-even-adjusted win-rate edge;
* acceptable drawdown;
* execution realism;
* chronological validity;
* cost robustness.

Candidates failing robust economic viability cannot be rescued by higher frequency.

### Stage 2 — Quality floor

The strategic realized OOS win-rate target remains approximately:

>= 60%.

For current Gemini quality-preserving frequency research, the preferred quality floor is:

* pooled realized WR >= 60%;
* PF > 1.05;
* Mean-R > 0;
* PnL > 0;
* break-even-adjusted WR edge > 0;
* cost-stress PF > 1.00;
* no catastrophic chronological fold.

Preferred stronger evidence:

* PF >= 1.15;
* cost-stress PF >= 1.05.

The quality floor is a constraint, not an objective that should be maximized without limit.

Do not prefer 70% WR over 62% WR merely because 70% is numerically higher when the 62% candidate is economically robust and provides materially more executable trades.

### Stage 3 — Frequency maximization

Among candidates passing the required quality floor, maximize:

ROBUST EXECUTABLE TRADES PER DAY.

Use:

* unique executable trades;
* chronological non-overlapping execution;
* actual position constraints;
* actual signal episodes rather than repeated signal rows.

Do not maximize:

* raw qualifying rows;
* repeated persistent M1 signals;
* classification accuracy;
* in-sample signal count.

### Stage 4 — Tie-breakers

When candidates have similar robust executable frequency, prefer:

1. higher realized WR;
2. higher PF;
3. higher Mean-R;
4. larger positive break-even-adjusted edge;
5. lower Max DD;
6. better cost robustness;
7. better chronological fold consistency.

---

## 3. Pareto-frontier rule

Always construct and report the:

QUALITY × EXECUTABLE FREQUENCY PARETO FRONTIER.

At minimum compare:

* executable trades;
* trades/day;
* frequency uplift vs incumbent;
* realized WR;
* TP-first WR;
* PF;
* Mean-R;
* PnL;
* Max DD;
* average winner R;
* average loser R;
* payoff ratio;
* break-even WR;
* break-even-adjusted WR edge;
* cost-stress result.

Do not collapse all metrics into one arbitrary weighted score when performing frequency optimization.

In particular, do not use a scoring rule where additional WR points can automatically overpower a large trade-frequency improvement after the quality floor has already been satisfied.

For current frequency research, candidate selection should be lexicographic:

1. reject candidates failing mandatory quality/economic constraints;
2. among survivors, rank primarily by robust executable trades/day;
3. use quality metrics as robustness checks and tie-breakers.

---

## 4. Frequency uplift

Always report frequency uplift relative to the relevant incumbent.

Define:

frequency uplift (%) =

(candidate executable trades/day / incumbent executable trades/day - 1) × 100

Explicitly report whether the candidate achieves approximately:

* +25%;
* +50%;
* +100%

frequency improvement.

A very small frequency improvement should not be described as meaningful merely because another metric improved slightly.

---

## 5. Incumbent treatment

The current `gemini.py` dedicated-long strategy is the operational baseline.

Treat the existing core as mature unless evidence specifically justifies replacing it.

Do not modify production merely to create research progress.

Always compare a candidate against:

1. its direct parent;
2. the current Gemini operational baseline where relevant.

Current production replacement and research are separate operations.

No experiment automatically receives permission to modify:

* `gemini.py`;
* `gold_long_recent_candidate_xgb.json`;
* any operational model/configuration.

---

## 6. Research discovery gate

A research candidate does not need every individual development fold to exceed 60% WR.

For discovery, sampling uncertainty and regime consistency must be considered.

A candidate may remain worthy of further research when it generally satisfies:

* pooled chronological realized WR >= 58%;
* worst meaningful fold >= 50%;
* pooled PF > 1;
* pooled Mean-R > 0;
* pooled PnL > 0;
* positive pooled break-even-adjusted edge;
* no catastrophic fold;
* meaningful unique executable-frequency or quality improvement.

The 58% threshold is only a research/discovery threshold.

It is not the strategic production-quality target.

Candidates failing this discovery gate must remain `research_only`.

Do not promote them.

---

## 7. Production promotion gate

Production promotion requires stronger evidence than development research.

A candidate must not be promoted merely because it:

* has higher recent WR;
* has more signals;
* has more trades on one interval;
* has pooled WR >= 60%;
* passes one favorable fold;
* passes a large parameter sweep.

Production promotion requires:

* robust chronological historical behavior;
* realized WR consistent with the strategic quality target;
* PF > 1;
* positive Mean-R;
* positive PnL;
* positive break-even-adjusted edge;
* acceptable drawdown;
* cost robustness;
* execution/backtest alignment;
* independent validator review;
* genuinely untouched forward evidence when available.

The existing historical 2014–2026 research periods have already been inspected repeatedly and cannot be relabeled as untouched final evidence.

The previous forward cutoff was:

2026-09-01T02:00:00Z

Data at or after that cutoff has been inspected and is therefore:

`contaminated_for_future_gate_selection`

It is development evidence and must not be presented as untouched forward evidence.

Establish a new untouched-forward cutoff only after a candidate is completely frozen. The new cutoff must be the candidate freeze timestamp or later and must be recorded before any post-cutoff strategy outcome is inspected.

---

## 8. Chronological integrity

All model and strategy research must be past-only.

This includes:

* model training;
* feature fitting;
* normalization;
* calibration;
* threshold selection;
* RSI/filter selection;
* session selection;
* ranking selection;
* family selection;
* regime selection;
* meta-model training;
* exit selection;
* champion selection.

Never use future information.

Never use evaluated-period outcomes to design a rule and then report that same interval as OOS evidence.

Never use random train/test splits for final trading validation.

Any prediction presented as OOF/OOS must be generated by a model that did not train on that event or future events.

Reverse-applying a 2026-trained model to 2018–2024 is diagnostic/counterfactual research only.

It must never be described as genuine OOS validation.

---

## 9. Contaminated historical data

All previously inspected historical research intervals are development data.

Previously called:

* holdout;
* recent test;
* test;
* validation;

intervals lose untouched status once researchers inspect their outcomes and use them to make later decisions.

Do not create false independence by assigning a new Generation number.

A new Generation does not create a new untouched test.

Track all repeated testing because it increases multiple-testing risk.

Strong final evidence should eventually come from genuinely new forward data collected after a candidate has been frozen.

---

## 10. Execution requirements

Always evaluate actual executable trades.

Do not evaluate strategy quality from raw classification signals alone.

Execution must include, when applicable:

* chronological order;
* one-position or configured position occupancy;
* signal de-duplication;
* independent signal episodes;
* spread;
* costs;
* slippage assumptions;
* TP;
* SL;
* timeout;
* same-bar TP/SL convention;
* position limits;
* session rules;
* risk guards.

First-touch TP/SL evaluation must follow the repository's documented intrabar HIGH/LOW convention.

If TP-first labels and realized trade profitability have different semantics, report them separately.

Never equate TP-first accuracy with realized profitability.

---

## 11. Signal episodes

Repeated qualifying M1 rows must not automatically be counted as separate trading opportunities.

Where appropriate, report separately:

* qualifying rows;
* unique signal episodes;
* executable trades.

A persistent signal lasting several bars is generally one opportunity unless an explicitly predefined entry-time-only rule defines a new independent episode.

Do not inflate frequency by counting repeated bars from the same setup.

---

## 12. Required economic metrics

Every serious candidate must report:

* executable trades;
* trades/day;
* frequency uplift;
* TP-first WR;
* realized positive-trade WR;
* average winning R;
* average losing R;
* payoff ratio;
* break-even WR;
* break-even-adjusted WR edge;
* PF;
* Mean-R / expectancy;
* PnL;
* Max DD;
* TP exits;
* SL exits;
* timeout exits.

Define:

break-even-adjusted WR edge =
realized WR - realized break-even WR.

A candidate is economically unacceptable when any of these are true:

* PF <= 1;
* Mean-R <= 0;
* PnL <= 0;
* break-even-adjusted WR edge <= 0.

Do not manufacture high WR using an economically unfavorable payoff structure such as tiny TP and very large SL.

TP/SL/timeout changes must be treated as a separate research dimension from signal-quality changes.

---

## 13. Cost robustness

All strategy comparisons must state the cost model.

Whenever possible report:

* nominal performance;
* observed-spread performance;
* predefined cost-stress performance.

A frequency increase that disappears after realistic costs is not an improvement.

Do not silently change cost assumptions between the incumbent and candidate.

---

## 14. Fold reporting

Always report chronological folds separately.

Do not hide unstable behavior behind pooled averages.

For every candidate intended for serious consideration report:

* each fold's trade count;
* trades/day;
* WR;
* PF;
* Mean-R;
* PnL;
* Max DD.

Also report pooled results.

A candidate whose pooled result is driven almost entirely by one regime must be treated cautiously.

---

## 15. Multiple testing

Track all tested:

* model families;
* feature families;
* thresholds;
* RSI ranges;
* session rules;
* ranking rules;
* filters;
* regimes;
* exit profiles;
* interaction features;
* candidate architectures.

Post-hoc discoveries must be labeled as such.

Do not repeatedly search new variants after viewing results and then treat the final choice as if it were preregistered.

Prefer a small predefined hypothesis set over broad parameter mining.

If an experiment fails, preserve the failure and stop unless a genuinely new hypothesis or information source justifies another experiment.

---

# Training Run Preservation

## Mandatory Git commit/push discipline

Every formal research or training run must be anchored to Git before computation begins.

Before creating the run:

1. inspect `git status`;
2. commit all intended research code, configuration, and policy changes;
3. push that commit to the configured remote;
4. verify that the pushed commit is the commit about to be executed;
5. preferably verify `git_dirty = false` before calling `training_run_history.py create`.

The exact executed Git commit must be recorded in `manifest.json`.

A run may use `git_dirty = true` only when necessary. In that case:

* the immutable executed-script snapshot remains mandatory;
* the dirty state must be explicitly recorded;
* the manifest or report must explain why a clean committed state was not possible;
* the dirty diff or another durable reconstruction record should be preserved when material to the result.

The preferred standard is a clean, committed, and pushed research state before computation.

After experiment completion, report generation, independent validation, `finalize --register`, and a
`training_run_history.py validate` provenance PASS:

1. inspect `git status` again;
2. commit all Git-trackable research evidence;
3. push the archival commit;
4. verify that the remote contains the archival commit.

Git-trackable evidence includes, where applicable:

* research and experiment scripts;
* `TRAINING_RUNS.md`;
* manifest and provenance records;
* reports and metrics;
* candidate tables;
* validator reports;
* `FINALIZED.json` metadata;
* relevant documentation changes.

PASS, FAIL, `research_only`, and meaningful `aborted` runs must all be committed and pushed. Do not
publish only successful experiments. Here, provenance validation PASS means the archive is internally
well formed; it does not change a strategy FAIL into a successful experiment.

Large datasets, runtime logs, and model binaries may remain outside normal Git when repository policy
excludes them. Their exact identity, SHA-256, durable storage path or locator, and retention status must
remain in the run metadata. A temporary local path is not durable retention and must not be described as
preserved.

The post-run commit and push are archival provenance only. They do not authorize modifying `gemini.py`,
replacing the operational model, or promoting a candidate. Research archival and production promotion
remain separate operations.

The mandatory lifecycle is:

research code ready -> commit -> push -> verify clean Git state -> create training run -> execute
experiment -> validate -> finalize/register -> validate preserved provenance -> commit research results
-> push -> verify remote provenance.

---

## 16. Every research/training execution must be retained

Every actual:

* model training;
* retraining;
* calibration;
* threshold-selection run;
* RSI/filter-selection run;
* parameter-selection run;
* information-study model fit;
* model comparison;
* parameter sweep;
* frozen OOF gate evaluation that influences future strategy decisions;

must create a permanent run record.

This requirement applies even if:

* no new model was trained;
* the experiment failed;
* no candidate passed;
* the run was aborted;
* production remained unchanged.

Every execution receives a unique `run_id`.

---

## 17. Run directory

Use:

training_runs/<YYYYMMDDTHHMMSSZ>_<experiment_name>/

Never reuse an existing run directory.

Corrections require a new run.

Do not edit a finalized historical run and pretend it is the original result.

---

## 18. Required run artifacts

Every completed non-aborted run should preserve, where applicable:

* `manifest.json`
* immutable executed script snapshot
* `report.md`
* `metrics.json`
* `environment.txt`
* `stdout.log`
* `candidates.csv` when search/selection occurred
* model artifact when training occurred
* `model.sha256` when training occurred
* validator output/result
* `FINALIZED.json`

`FINALIZED.json` must contain SHA-256 hashes of retained run files so later modifications can be detected.

When a large artifact is intentionally excluded from Git, the manifest must still record its exact
identity, SHA-256, durable storage location, and retention status. A local temporary path alone is not a
retention method.

---

## 19. Run identity

The manifest must record:

* run_id;
* experiment_name;
* status;
* UTC start;
* UTC finish;
* Git commit;
* Git dirty state;
* Git dirty reason when dirty;
* Git branch and remote/ref used for the pre-run push where available;
* executed script path;
* immutable script snapshot;
* script SHA-256;
* exact command;
* arguments;
* random seeds or why none apply.

---

## 20. Data provenance

The manifest must record:

* symbols;
* data sources;
* source files;
* source SHA-256 where feasible;
* timezone;
* total data start/end;
* train start/end/rows;
* validation start/end/rows;
* test start/end/rows;
* purge;
* embargo.

For dynamic MT5 retrieval also record:

* terminal path;
* terminal information;
* broker/server information when available;
* fetch start/end;
* retrieval timestamp;
* returned row counts.

Explicitly record whether raw data snapshots were retained.

Never claim full reproducibility if the actual input snapshot was not retained.

---

## 21. Model provenance

When a model is trained record:

* model type;
* complete parameters;
* estimators/boosted rounds;
* complete feature list;
* feature count;
* label definition;
* horizon;
* training TP/SL semantics;
* execution TP/SL semantics;
* calibration method;
* artifact path;
* artifact SHA-256;
* retention location/status.

Training-label semantics and runtime-execution semantics must be documented separately when they differ.

---

## 22. Search provenance

If candidate selection or a parameter sweep occurs, preserve the entire predefined search space.

Do not save only the winner.

`candidates.csv` must retain every evaluated candidate, including failures.

At minimum record:

* candidate ID;
* parameters;
* fold;
* executable trades;
* trades/day;
* TP-first WR;
* realized WR;
* PF;
* Mean-R;
* PnL;
* Max DD;
* break-even WR;
* break-even-adjusted edge;
* cost stress;
* qualification verdict.

This is required so failed experiments are not unknowingly repeated later.

---

## 23. Environment

Each run must record:

* Python version;
* numpy version;
* pandas version;
* scikit-learn version;
* xgboost version;
* MetaTrader5 package version.

Where other important packages materially affect the result, include them too.

---

## 24. Training registry

Every finalized run must be appended to:

TRAINING_RUNS.md

The registry is append-only.

Never remove failed or aborted runs.

At minimum include:

* run_id;
* date;
* experiment;
* parent/incumbent;
* model SHA-256 if applicable;
* data interval;
* selected configuration;
* trades/day;
* WR;
* PF;
* Mean-R;
* PnL;
* Max DD;
* validator result;
* final status;
* artifact location.

---

## 25. Operational artifact protection

Training is not production replacement.

Newly trained artifacts must preferably be written directly into their unique training run directory.

Do not let a training script silently overwrite the only copy of:

* `gold_long_recent_candidate_xgb.json`;
* any model currently loaded by `gemini.py`;
* the current operational report/config.

If a legacy training script must write to an operational path:

1. archive the incumbent first;
2. record SHA-256;
3. preserve associated report/config;
4. restore/protect production unless replacement is independently authorized.

A newly trained model remains:

`research_only`

until the full promotion gate passes and production replacement is separately authorized.

---

## 26. Finalization

Use the repository's:

`training_run_history.py`

workflow:

1. inspect Git status
2. commit and push intended research code/config/policy
3. verify the executed commit and preferably a clean worktree
4. `create`
5. run the experiment
6. populate manifest/report/metrics/candidates
7. perform independent validation
8. `finalize --register`
9. `validate`
10. commit and push all Git-trackable run evidence
11. verify the remote contains the archival commit

Do not manually fabricate a finalized run.

The validator must detect missing provenance and post-finalization modification.

---

## 27. Current legacy model

Do not fabricate missing evidence for the existing 2026-08-25 dedicated-long operational model.

Its known limitation remains:

* original raw MT5 snapshot was not fully retained;
* complete package/environment lock was not retained.

Document known information accurately.

The stronger Training Run Preservation rule applies prospectively to future runs.

---

# Research behavior

## 28. Failure is a valid result

Do not create a new candidate merely to show progress.

If an experiment does not produce a candidate that improves the parent under the required constraints:

* report FAIL clearly;
* retain the run;
* retain candidate results;
* retain useful diagnostics;
* do not promote anything;
* do not immediately launch a broader sweep merely because the result was negative.

Prefer identifying why the hypothesis failed.

---

## 29. Independent validation

Use the repository-local `walk-forward-validator` independently from optimization.

The optimizer must not silently tune against validator failures.

If validation fails because an evaluated interval has been contaminated, multiple testing is excessive, chronology is invalid, or OOF predictions are not genuine:

* report the failure;
* downgrade the evidence claim;
* do not hide the issue through additional tuning.

---

## 30. Current Gemini research priority

For the current `gemini.py` program, the immediate research priority is:

INCREASE EXECUTABLE TRADES/DAY WITHOUT DEGRADING THE REQUIRED QUALITY FLOOR.

Current evidence indicates that recent low frequency is primarily caused by signal scarcity / entry gating rather than:

* position occupancy;
* spread;
* risk guard;
* order failure;
* long holding duration.

Therefore current research should prioritize narrowly defined quality-preserving entry-gate studies before researching:

* second positions;
* leverage increases;
* broad new signal families;
* large feature sweeps.

Do not assume this conclusion remains permanently true; update it when sufficient new production evidence shows otherwise.
