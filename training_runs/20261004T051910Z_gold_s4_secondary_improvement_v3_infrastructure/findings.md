# GOLD S4 Improvement v3 infrastructure

Status: PASS

INFRASTRUCTURE_ONLY. Compiler, synthetic fixtures and static BAT checks only. No candidate model training, real historical validation, strategy replay/search or historical input data use. The test process rejects historical arrays/models and all locked holdout access. Frozen reference constants and provenance metadata are the only reference evidence inspected.

23 fixed configurations centered on A_NO_LONG_HTF; four HTF aggregates and two timing features; all original folds. A_NO_LONG_HTF control and threshold probes will reuse existing models without retraining, only when USER later double-clicks the BAT. Independent full real validation is implemented but unexecuted in this certification. All real execution requires the USER session; validation additionally requires a one-use run-bound permit. No production promotion or production change. Historical development only.

Passing fixture certification is not proof that a real future user run will pass or improve. Approval binds the certified source after this archive is pushed.
