# GOLD# 模型研究與 MT5 Forward Test

本專案包含 GOLD# 模型訓練、chronological walk-forward 研究、可執行事件回測、候選晉級稽核，以及 MT5 測試帳戶 runner。

> `gemini.py` 目前 `DRY_RUN = False`，會實際送出委託。啟動前務必確認 MT5 連線的是測試帳戶。

## `gemini.py` 與目前模型是怎麼來的

`gemini.py` 是 MT5 執行器，不是模型訓練程式。現在的執行鏈如下：

```text
MT5 GOLD# 歷史 bars
  -> gold_long_recent_walk_forward.py（訓練與選模）
  -> gold_long_recent_candidate_xgb.json（31-feature long binary XGBoost）
  -> gemini.py（session／RSI／spread／risk／position guards）
  -> gemini_signal_log.csv + gemini_trade_history.csv
```

Git 與模型沿革：

| 日期 | 事件 | 可追溯證據 |
|---|---|---|
| 2026-05-18 | 建立第一版 `gemini.py`，使用 `gold_barrier_final_xgb.json` 與 meta-regime overlay | commit `0e6423d`、[VERSION_LOG.md](VERSION_LOG.md) |
| 2026-06-06 | 加入 MT5 成交歷史同步與較完整的交易日誌 | commit `152b05a` |
| 2026-08-25 06:49 UTC | 執行 dedicated-long walk-forward，產生目前模型 | [gold_long_recent_walk_forward.py](gold_long_recent_walk_forward.py)、[gold_long_recent_walk_forward.md](gold_long_recent_walk_forward.md) |
| 2026-08-30 | 將訓練程式與模型 artifact 納入 Git | commit `8f15e4a` |
| 2026-09-01 | `gemini.py` 切換成目前的 long-binary 模型與 1.4% risk／0.75 threshold 設定 | commit `d741caf` |

重要說明：目前模型原始訓練報告的 `promotion_pass` 是 `false`，原因是 test 只有 15 筆交易，未達當時要求的 20 筆最低樣本。它後來被指定為目前測試帳戶的 operational／forward baseline；這不等於它已通過現在 [AGENTS.md](AGENTS.md) 定義的完整 production promotion gate。Generation 8–21 與後續擴頻研究也都沒有產生可取代它的合格候選。

## 目前 Forward-Test 設定

以下以 [gemini.py](gemini.py) 目前常數為準：

| 項目 | 設定 |
|---|---|
| Symbol / magic | `GOLD#` / `20260514` |
| Model | `gold_long_recent_candidate_xgb.json` |
| Output / direction | `long_binary` / long only |
| Meta overlay | 關閉 |
| 主要進場門檻 | `P(long) >= 0.75` |
| Risk | 每筆 `1.4%` |
| TP / SL | `1.3 ATR / 1.6 ATR`；最低價格距離 `1.5 / 0.6` |
| 最長持倉 | `90` 分鐘 |
| Position limit | 最多 `1` 個 GOLD# position |
| Session | UTC 小時 `0,1,2,3,4,8,9,11,12,17,18,19,20,22,23`；週一至週五 |
| RSI guards | `RSI >= 22`，排除 `35–45` |
| Spread guards | soft `45` points、hard `100` points、spread/TP 不超過 `0.25` |
| Risk guards | 日損上限 `5%`；虧損後 cooldown `15` 分鐘；最近 30 筆且至少 18 筆時，PF `< 1.15` 將風險乘數降為 `0.5` |
| Time / mode | UTC logs；`DRY_RUN = False` |

`LOWER_CONF_THRESHOLD = 1.01`、`ENABLE_EXPANDED_ENTRY = False`、meta overlay 關閉，因此目前沒有低門檻、expanded 或 meta 旁路訊號。

## 目前模型的完整訓練出生證明

### Artifact 身分

| 欄位 | 值 |
|---|---|
| Model file | `gold_long_recent_candidate_xgb.json` |
| SHA-256 | `2dc32e3b3c0ea6ca8fa2e30187bebf8ff3f7e7e03109b39b3f70f013e3a755f2` |
| XGBoost artifact version | `3.2.0` |
| Objective | `binary:logistic` |
| Boosted rounds | `220` |
| Number of features | `31` |
| Generated at | `2026-08-25T06:49:39.192110Z` |

### Chronological split

這個模型不是用完整 2014–2026 訓練；它是 2025–2026 recent-window model，沒有 random split。

| 階段 | 時間範圍（UTC） | 用途 | 已保存 rows |
|---|---|---|---:|
| Selection fit | 2025-01-01 至 2026-04-01 前 | 訓練 validation model；尾端 purge 240 bars | 原報告未保存 |
| Validation | 2026-04-01 至 2026-06-01 前 | 選 threshold、TP/SL、hold、session | 57,752 |
| Final fit | 2025-01-01 至 2026-06-01 前 | 重新訓練目前保存的 model；尾端 purge 240 bars | 原報告未保存 |
| Test | 2026-06-01 至 2026-08-25 05:49 | frozen evaluation | 83,911 |

原始 MT5 bar snapshot 與 Python package lock 沒有一併保存，因此現在重新執行腳本會取得不同結束時間的資料，不能保證 byte-for-byte 產生相同 SHA-256。Git 中保存的是實際被 runner 載入的模型 artifact、訓練程式與摘要報告。

### 標籤、特徵與模型

訓練 positive class 是 `BARRIER_TARGET == 1` 的 long clean-barrier event：

- Horizon：未來 `240` 個 M1 bars。
- Label TP：`max(1.8 ATR, 1.0 price)`。
- Label SL：`max(1.2 ATR, 0.8 price)`。
- Long positive：未來區間最高價到達 TP，且整個區間最低價沒有到達 SL。
- 其他事件全部作為 binary negative class；class-balanced sample weights。

這個舊標籤使用未來區間 extrema 判定 clean event，不是後來 Generation 研究採用的逐 bar TP／SL first-touch label。選模回測則使用盤中 `HIGH`／`LOW` 的 executable first-touch 邏輯，以及另外選出的 runtime `1.3/1.6 ATR` exit profile。也就是說，training label 與 execution exit 並非同一組 TP/SL 語義；這是目前模型已知的研究限制，不能在文件中把它誤稱為完整 execution-aligned first-touch model。

31 個 entry-time features 全部由已完成 bars 產生：

- 11 個 M1 features：`M1_RSI`、`ATR`、`MACD_HIST`、`BB_WIDTH`、`BIAS_20`、`BODY_PCT`、`ROC_5`、`VOLA_RATIO`、`HOUR_SIN`、`HOUR_COS`、`DAY_OF_WEEK`。
- 20 個 MTF trend flags：`Daily`、`H12`、`H1`、`H2`、`H3`、`H4`、`H6`、`H8`、`M10`、`M12`、`M15`、`M20`、`M2`、`M30`、`M3`、`M4`、`M5`、`M6`、`Monthly`、`Weekly`。
- Higher-timeframe bars 使用 backward as-of 對齊；feature matrix 在決策前 shift，避免把當下未完成 bar 當成已知資訊。

XGBoost 固定參數：

```text
n_estimators=220
learning_rate=0.05
max_depth=4
min_child_weight=80
subsample=0.85
colsample_bytree=0.85
random_state=42
tree_method=hist
```

### 選模範圍與結果

Validation 內測試的固定組合為：

- Threshold：`0.55, 0.60, 0.65, 0.70, 0.75, 0.80`。
- TP ATR：`0.9, 1.1, 1.3`。
- SL ATR：`1.6, 2.0`。
- Max hold：`90, 180`。
- Session profile：4 組。
- 合計 `288` 個組合，其中 `30` 個通過 validation qualification。

選中參數是 threshold `0.75`、TP/SL `1.3/1.6 ATR`、hold `90`、expanded weekday/session、risk `1.4%`。

| Fold | Executable trades | Win rate | PF | PnL | Max DD | TP | SL | Timeout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 20 | 70.00% | 1.756 | +82.03 | -6.73% | 14 | 6 | 0 |
| Test | 15 | 66.67% | 1.345 | +34.25 | -6.93% | 9 | 5 | 1 |
| Test +10-point cost | 15 | 66.67% | 1.332 | +33.06 | -6.97% | 9 | 5 | 1 |

原始 gate 要求 validation 至少 15 筆、test 至少 20 筆、WR `>= 60%`、PF `>= 1.15`、正 PnL、DD 不超過 20%，且 +10-point cost stress 的 PF `>= 1.05`。所有經濟指標通過，但 test 只有 15 筆，所以最終是 `FAIL`，不是正式 promotion pass。

## 完整研究世代紀錄

所有表內歷史區間都已被反覆查看，現在只能視為 development／research history。`FAIL` 仍是需要保留的研究結果；新 Generation 編號不會自動產生新的獨立 OOS 證據。

| 世代／分支 | 核心研究 | 結果摘要 | 最終狀態 |
|---|---|---|---|
| Initial 2026-05-18 | 三分類 barrier model + meta-regime sizing | 曾作為最初 demo baseline；之後由 recent long model 取代 | 歷史版本 |
| Gen1–5 | long/short × trend/pullback、time decay、blend、方向門檻、rolling calibration/champion | 全部未 qualified；選出的 Gen4 仍為總 PnL `-374.17`、worst PF `0.80` | FAIL |
| Gen6 | 四個 Expected-R experts + rolling top-k + realized-R champion | 各主要 fold PF `0.55–0.70`，PnL 全負 | FAIL |
| Gen7 | Event-only probability-to-Expected-R ranking | qualified candidates `0`，各主要 fold PF `0.56–0.63` | FAIL |
| Gen8 | M5 residual model | recent diagnostic 43 trades、62.79%、PF 1.17、PnL +37.58；完整 gate 未通過 | FAIL |
| Gen9 | Repo 中沒有獨立編號、可凍結重建的 Gen9 report | 不臆造缺失指標 | 未形成可稽核 candidate |
| Gen10 | Adaptive portfolio | recent 10 trades、30.00%、PF 0.29、PnL -64.15 | FAIL |
| Gen11 | Execution-aligned model | recent 9 trades、22.22%、PF 0.10、PnL -86.31 | FAIL |
| Gen12 | Executable-event Expected-R | recent 51 trades、60.78%，但 PF 0.93、PnL -15.34 | FAIL |
| Gen13 | Directional exits | recent 60 trades、46.67%、PF 0.71、PnL -70.29 | FAIL |
| Gen14 | Precision-constrained frequency + loser meta-filter | recent 82 trades、63.41%、PF 1.18，但早期 folds 49–59% 且 PnL 全負；qualified `0` | FAIL |
| Gen15 | False-positive / missed-winner mining | 沒有 qualified candidate；fallback 與 parent 完全相同，沒有 unique added trades 或 loser removal | FAIL |
| Gen16 | Independent signal families | qualified `0`；多數 folds 約 48–50% 且負 expectancy | FAIL |
| Gen17 | Cross-regime normalized features | short trend continuation pooled 206 trades、60.19%，但初始 PF 0.89、Mean-R -0.0452；無跨 regime champion | FAIL |
| Gen18 | Payoff audit + calibration-robust ranking | 證實 TP-first WR 與 realized economics 必須分開；rank candidates 仍為負 PF／Mean-R | FAIL |
| Gen19 | Observed-spread cost-aware break-even | Gen17 cohort 重算為 206 trades、60.19%、PF 0.994、Mean-R -0.0024、edge -0.14pp；近 break-even 但仍不合格 | FAIL |
| Gen20 | Direct `P(net-R>0)` / `E(net-R)` learning | 所有 fixed-cohort selectors PF `0.691–0.994`；Phase 5 全失敗，沒有 frequency expansion | FAIL |
| Gen21 | New information: microstructure + XAG ablation | technical control mean Spearman 0.034；新增 family 更差，沒有 candidate construction | FAIL |
| TICKVOL study | 凍結 `TV_ACCEL` 與最小 ablation | raw H1 三 fold 同向，但加到 technical control 後只改善 1/3 folds、mean delta -0.0691；不建立 Gen22 | CLOSED / FAIL |
| GEMINI Frequency Expansion 1 | 固定 production score 的 near-miss/re-entry 擴頻 | 沒有 combined PF > 1、Mean-R > 0 且增加 frequency 的組合；validator FAIL | RESEARCH_ONLY / FAIL |

主要報告索引：

- Gen1–5：[gold_regime_experts_iterative.md](gold_regime_experts_iterative.md)
- Gen6：[gold_expected_r_walk_forward.md](gold_expected_r_walk_forward.md)
- Gen7：[gold_event_rank_walk_forward.md](gold_event_rank_walk_forward.md)
- Gen8：[gold_generation8_residual_walk_forward.md](gold_generation8_residual_walk_forward.md)
- Gen10–13：[gold_generation10_adaptive_portfolio.md](gold_generation10_adaptive_portfolio.md)、[gold_generation11_execution_aligned.md](gold_generation11_execution_aligned.md)、[gold_generation12_executable_events.md](gold_generation12_executable_events.md)、[gold_generation13_directional_exits.md](gold_generation13_directional_exits.md)
- Gen14–16：[gold_generation14_precision_frequency.md](gold_generation14_precision_frequency.md)、[gold_generation15_signal_mining.md](gold_generation15_signal_mining.md)、[gold_generation16_independent_families.md](gold_generation16_independent_families.md)、[Gen16 validator](gold_generation16_walk_forward_validation.md)
- Gen17–18：[gold_generation17_cross_regime.md](gold_generation17_cross_regime.md)、[Gen17 validator](gold_generation17_walk_forward_validation.md)、[gold_generation18_payoff_alignment.md](gold_generation18_payoff_alignment.md)、[Gen18 validator](gold_generation18_walk_forward_validation.md)
- Gen19–21：[gold_generation19_cost_aware.md](gold_generation19_cost_aware.md)、[Gen19 validator](gold_generation19_walk_forward_validation.md)、[gold_generation20_direct_net_edge.md](gold_generation20_direct_net_edge.md)、[Gen20 validator](gold_generation20_walk_forward_validation.md)、[gold_generation21_new_information.md](gold_generation21_new_information.md)、[Gen21 validator](gold_generation21_walk_forward_validation.md)
- 後續封閉研究：[gold_tickvol_information_study.md](gold_tickvol_information_study.md)、[TICKVOL validator](gold_tickvol_information_validation.md)、[gold_gemini_frequency_expansion1.md](gold_gemini_frequency_expansion1.md)、[frequency validator](gold_gemini_frequency_expansion1_validation.md)
- 其他 long/short 與 full-history 失敗實驗：[gold_long_model_optimization.md](gold_long_model_optimization.md)、[gold_long_aligned_model_optimization.md](gold_long_aligned_model_optimization.md)、[gold_long_rule_optimization.md](gold_long_rule_optimization.md)、[gold_long_full_history_walk_forward.md](gold_long_full_history_walk_forward.md)、[gold_long_full_history_first_touch_walk_forward.md](gold_long_full_history_first_touch_walk_forward.md)、[gold_short_recent_walk_forward.md](gold_short_recent_walk_forward.md)

## 驗證與資料狀態

目前沒有真正 untouched historical outer interval：

- 2014–2026 已查看資料全部是 development／research history。
- 過去叫做 `holdout` 或 `recent` 的區間已被多代研究查看，不能重新宣稱 untouched OOS。
- Untouched-forward cutoff 是 `2026-09-01T02:00:00Z`。cutoff 後資料在 candidate 完全凍結前，不得用來做 feature、model、threshold、family、calibration 選擇，也不得查看 outcome correlation、WR、PF 或 PnL 來修改模型。
- 若 cutoff 後結果被用來除錯或選模，該區間必須立刻改列 development data，並重新設定更晚的 forward cutoff。

因此目前最準確的說法是：`gemini.py` 是正在測試帳戶運作的既有 operational baseline；歷史報告提供 development evidence，但沒有任何現行或新世代模型擁有完整 untouched-test production claim。

## 重新訓練與執行

建議使用 Python 3.10。若 `python` 不在 PATH，可使用 `py -3.10` 或 repo 的 `.venv`。

啟動 MT5 runner：

```powershell
py -3.10 gemini.py
```

重新執行 current-model training pipeline：

```powershell
py -3.10 gold_long_recent_walk_forward.py
```

注意：訓練腳本會連線 `D:\XM2\terminal64.exe`、以當下最新 tick 作為資料結束時間，並覆寫 `gold_long_recent_candidate_xgb.json` 與本機報告。若要重現研究，應先在獨立 branch／artifact path 凍結資料與輸出，不能直接覆寫正在 forward test 的模型。

## 主要檔案

- [gemini.py](gemini.py)：MT5 forward-test runner；不負責訓練。
- [gold_long_recent_walk_forward.py](gold_long_recent_walk_forward.py)：目前 model artifact 的訓練、validation 選模與 test gate。
- [gold_long_recent_walk_forward.md](gold_long_recent_walk_forward.md)：目前模型原始摘要與 `promotion_pass = FAIL`。
- `gold_long_recent_candidate_xgb.json`：目前 runner 載入的已追蹤 XGBoost artifact。
- [barrier_classifier_strategy.py](barrier_classifier_strategy.py)：舊 clean-barrier label、後續 first-touch labels 與 executable evaluation 共用邏輯。
- [AGENTS.md](AGENTS.md)：目前研究目標、economic guardrails、chronology 與 promotion policy。

## 日誌與 Git 資料政策

`gemini.py` 在本機持續追加：

- `gemini_signal_log.csv`
- `gemini_trade_history.csv`
- `runtime_logs/`

Runner 重啟不代表舊 CSV 自動消失，但同一份 log 可能跨越不同 model／threshold／risk 設定。分析時必須以每次啟動行、model hash、設定變更時間與 Git commit 切分 era，不能把 5/18 以來所有 rows 當作目前 0.75 long-binary 版本的成績。

執行日誌可能包含帳戶與 broker 細節，而且會持續增長，因此不納入 Git。大型 `GOLD#_*.csv` 歷史資料與生成式研究 JSON 也預設只留本機；需要重現時應使用明確版本的 artifact storage 或 Git LFS，不能把缺少的原始 snapshot 當成已保存。
