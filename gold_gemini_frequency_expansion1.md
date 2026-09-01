# GEMINI FREQUENCY EXPANSION 1

Status: `research_only`; `gemini.py` and the production model were not modified.

> This is a development-period counterfactual replay, not a new OOS claim. The current model was trained/selected on 2025-2026 data and is applied backward to 2018-2024 here.

## Frozen production reconstruction

Long-only, score >= 0.75, frozen production sessions and RSI guards, TP/SL 1.3/1.6 ATR, 90-minute maximum hold, one position, 15-minute post-loss cooldown, spread gate, and 30-point fallback where historical spread is unavailable.

Historical broker-account state, exact fills/slippage/commission, and the server-time-to-UTC mapping are unavailable; the ledger is the closest causal reconstruction, not a byte-for-byte broker ledger.

## Pooled portfolio comparison (2018-2024; calendar-day denominator)

| Portfolio | Prod trades | Added | Uplift | Combined/day | Combined WR | Combined PF | Combined Mean-R | Combined PnL-R | Max DD-R | Expansion PF | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| near_miss_union | 763 | 1325 | 173.7% | 0.8166 | 46.26% | 0.75 | -0.1350 | -281.85 | -284.57 | 0.77 | FAIL |
| near_miss_plus_reentry | 763 | 1325 | 173.7% | 0.8166 | 46.26% | 0.75 | -0.1350 | -281.85 | -284.57 | 0.77 | FAIL |
| near_miss_persistence | 763 | 1282 | 168.0% | 0.7998 | 46.16% | 0.75 | -0.1377 | -281.68 | -285.41 | 0.76 | FAIL |
| all_predeclared | 763 | 1950 | 255.6% | 1.0610 | 44.78% | 0.73 | -0.1546 | -419.44 | -424.22 | 0.73 | FAIL |
| reentry_only | 763 | 7 | 0.9% | 0.3011 | 47.40% | 0.72 | -0.1495 | -115.08 | -116.46 | 0.87 | FAIL |
| near_miss_pullback | 763 | 205 | 26.9% | 0.3786 | 45.66% | 0.70 | -0.1659 | -160.57 | -162.60 | 0.63 | FAIL |

## Diagnostic least-degraded portfolio by fold

Accepted internal portfolio: `None`; diagnostic reference: `near_miss_union`.

| Fold | Layer | Trades | Trades/day | WR | PF | Mean-R | PnL-R | Max DD-R | TP | SL | Timeout | Preempt |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018_2020 | production | 385 | 0.3513 | 41.30% | 0.62 | -0.2268 | -87.33 | -87.33 | 145 | 218 | 22 | 0 |
| 2018_2020 | expansion | 692 | 0.6314 | 42.77% | 0.74 | -0.1474 | -102.03 | -102.77 | 281 | 366 | 24 | 21 |
| 2018_2020 | combined | 1077 | 0.9827 | 42.25% | 0.70 | -0.1758 | -189.35 | -189.50 | 426 | 584 | 46 | 21 |
| 2021_2022 | production | 120 | 0.1644 | 51.67% | 0.77 | -0.1137 | -13.64 | -22.20 | 59 | 58 | 3 | 0 |
| 2021_2022 | expansion | 221 | 0.3027 | 47.06% | 0.81 | -0.1045 | -23.11 | -36.50 | 100 | 111 | 1 | 9 |
| 2021_2022 | combined | 341 | 0.4671 | 48.68% | 0.79 | -0.1078 | -36.75 | -49.02 | 159 | 169 | 4 | 9 |
| 2023_2024 | production | 258 | 0.3529 | 54.65% | 0.88 | -0.0535 | -13.81 | -25.94 | 141 | 117 | 0 | 0 |
| 2023_2024 | expansion | 412 | 0.5636 | 49.51% | 0.81 | -0.1018 | -41.95 | -50.79 | 198 | 207 | 1 | 6 |
| 2023_2024 | combined | 670 | 0.9166 | 51.49% | 0.83 | -0.0832 | -55.75 | -66.28 | 339 | 324 | 1 | 6 |

## Required answers

1. 主要限制是高門檻後的訊號稀疏與持倉占用：4131 個合格 M1 rows 中有 2448 個被既有持倉占用。
2. 固定 0.65-0.75 near-miss 帶共有 132370 rows；套用其他 production guards 後為 61267 rows。
3. 具有正 pooled expectancy 的 near-miss selector：沒有
4. Re-entry 可執行 7 筆，WR 42.86%、PF 0.87、Mean-R -0.0434。
5. 原始候選最多的新增來源是 near_miss_persistence（1842 events）；真正可用增量仍以 non-overlapping executable trades 為準。
6. 在 combined PF > 1 且 Mean-R > 0 的候選中，最大 frequency uplift 為 0.0%。
7. 維持 combined WR >= 60% 且增加頻率的組合：沒有
8. 沒有通過凍結 gate 的最佳組合；最少劣化的 diagnostic reference 是 near_miss_union：combined 2088 trades、0.8166/day、WR 46.26%、PF 0.75。
9. 僅當 paper_shadow_gate 與獨立 validator 都通過才值得建立 sidecar；本輪 paper_shadow_gate=FAIL。即使內部結果漂亮，也必須先收集 cutoff 後完全未看的 forward data。

## Method constraints

- Production always has priority. An expansion position is closed at market when a frozen production entry arrives, so the production ledger is unchanged.
- No TP/SL, production threshold, production model, or production session was changed.
- The candidate list was frozen before outcome calculation; no broad threshold/model sweep was run.
- Historical spread uses the Gen19 observed-value method with a 30-point fallback. This is not proof of exact historical fill cost.
- Untouched-forward cutoff `2026-09-01T02:00:00Z` was not inspected.
