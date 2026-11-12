# XM 研究記憶（自動產生，請勿手改）

由 `research_memory_v1.py render` 從 `research_memory/ledger.jsonl` 產生。事件數 64；帳本完整性：OK。

分類規則：**已驗證觀察**＝有證據且未撤回；**假說**＝尚未有結論；**失敗**＝被推翻或中止；**資料重複使用**＝評估區間在登錄前已被看過，只能當開發證據。

## 1. 已驗證的觀察

- **O-001** XM GOLD# costs: live spread median 31 points ($0.31, p90 33), commission 0, long swap -78.59 points/night (short +16.81), triple on Wednesday; swap_mode reports 0 but swap is charged  
  證據：gemini_signal_log.csv order fills (114) since 2026-06; MT5 deal history 252 rows; symbol_info; user confirmation
- **O-002** Shadow replay of gemini matches live fills: same 16 trades +0.133R replay vs +0.131R live; score parity 99.9%; replay spread about 6.5 points optimistic  
  證據：forward_reports/20261108_shadow_replay_v1_oct_nov
- **O-003** gemini M1 (0.65) has about zero-to-negative net expectancy: 321 replay trades Jun-Nov 2026, -0.031R, PF 0.93; break-even win rate about 55% with 1.3/1.6 ATR exits; spread is 6-10% of the TP distance  
  證據：forward_reports/20261103_shadow_replay_v1_thr065_jun_nov; STRATEGY_RESEARCH_2026_11.md section 3
- **O-005** Combo A does not transfer: unchanged on 12 other XM markets it is negative or about 0 on 11; on gold with swap its 2014-2023 mean is -0.086R  
  證據：forward_reports/20261111_cross_asset_combo_v1
- **O-007** All earlier 'new information' families (macro event timing, US Treasury real rates, CFTC COT, USD FX pressure, microstructure, TICKVOL) were evaluated only as extra features of the short-horizon (M1, about 90 min) long classifier; their closure says nothing about their native horizon  
  證據：TRAINING_RUNS.md rows 20260906T060148Z, 20260909T140906Z, 20260913T141817Z; gold_generation21_new_information.md; gold_tickvol_information_study.md
- **O-008** Public rule families on gold with XM costs (20 classic rules, 4 web strategies incl. news straddle with slippage, gold/silver ratio, daily 16-asset trend): none positive in both 2014-2023 and 2024+ except the EMA20/50 breakout long  
  證據：forward_reports/20261108_rules_lab_v1, 20261111_web_strategies_v1, 20261111_ratio_lab_v1, 20261105_trend_daily_v1
- **O-009** E-001 的 H1_C_frozen 判定無效：2018-2023 與它的訓練區間（fit 2014-2022、門檻挑選 2023）重疊，+0.44R 時機殘差是樣本內結果。只看樣本外 2024+：H1_C 殘差 +0.083R（CI -0.05~0.21），同月隨機進場有 15% 不輸它，RANDOM_2PCT 對照組自己也有 +0.11R。  
  證據：frozen/h1_C_long_v1/h1_lab_v1.py 第 8 行 splits；training_runs/20261111T111328Z_research_e001_exp_edge_decomposition_v1/report.md
- **O-010** 2024+ 單一區間的時機殘差噪音約 ±0.1R（隨機進場就能得到 +0.11R），幾百筆交易無法分辨 0.1R 等級的時機能力。  
  證據：E-001 RANDOM_2PCT recent_2024_on 列
- **O-011** E-001 與 E-002 在研究程式尚未 commit 的狀態下執行（manifest git_dirty=true，基準 commit c78f0c1），不符合 AGENTS.md 先 commit/push 再計算的規定。兩者都保存了 training_script 快照與 sha256，程式也與預先登記的雜湊相符，所以結果可追溯；但不能宣稱由某個 commit 完整重現。  
  證據：training_runs/20261111T111328Z_research_e001_exp_edge_decomposition_v1/manifest.json、training_runs/20261111T140822Z_research_e002_exp_information_horizon_v1/manifest.json；之後 research_run_v1 改為未 commit/push 時拒絕執行
- **O-012** research_eval_v1.py 在 E-001 執行之後只在檔尾新增 daily_stats()（E-005~E-007 使用），E-001 用到的函式未改；因此目前檔案的 sha256 不等於 E-001 登記值，這是預期的，不代表 E-001 結果被改動。  
  證據：git 提交 2026-11-11 研究治理版本；E-001 preregistration code_files

## 2. 尚未驗證的假說（依 Research Priority Engine 排序）

分數綜合資訊增益、與過去研究的差異、可反證性、資料品質與取得成本、經濟價值、工程簡單度，扣掉資料探勘風險與同族群已試次數。失敗但能排除重要可能性的實驗同樣得分。公式見 `research_planner_v1.py`。

| ID | 假說 | 族群 | 狀態 | 優先分 | 推翻條件 |
|---|---|---|---|---|---|
| H-014 | 黃金隱含波動率（CBOE GVZ）與實現波動的差（變異數風險溢酬）預測未來 5-20 日的實現波動與報酬 | new_information_implied_vol | open | 39.0 | 把 GVZ 加進 HAR 後 QLIKE 沒有下降（DM &#124;t&#124; < 2），且溢酬對 20 日報酬的 IC &#124;t&#124; < 1 |
| H-008 | 收盤前平倉、只做盤中多單，比一直持有多單（付過夜利息）的風險調整後報酬更好 | cost_structure | inconclusive | 38.5 | 2018-2022 成本 x1 下 INTRADAY_LONG 的 Sharpe 不高於 ALWAYS_LONG |
| H-013 | 可以預測未來 1 小時的波動是否大於 k 倍點差（不預測方向），而 gemini 的虧損集中在預測『不值得交易』的時段 | tradeability | open | 36.0 | 樣本外波動大小分類 AUC <= 0.55，或低可交易性組的 gemini 回放 R 沒有比其他組低 0.05R 以上 |
| H-018 | 波動預測能改善交易決策：倉位風險控制、實際回撤、辨識不值得交易的環境，且改善足以抵銷額外交易成本 | second_moment_decision_value | open | 35.0 | 2018-2023 以凍結預測做風險目標倉位與不交易濾網，在相同平均曝險下，扣除額外換手成本後 Sharpe 不高於固定風險版本，且最大回撤降低 < 10% |
| H-009 | 點差相對同時段常態異常擴大時，接下來的波動與滑價較大，是不值得進場的時刻 | microstructure | open | 34.0 | 點差 z 值最高 10% 時段之後 30 分鐘的實現波動比 < 1.2，或排除這些時段進場使既有策略回放淨 R 改善 < 0.02R |
| H-012 | 低流動時段的單根急動（> 3 倍 ATR）在 15-60 分鐘內部分回補，扣除當下實際點差後期望為正 | transient_anomaly | open | 33.0 | 2018-2022 以實際點差計的平均淨回補 <= 0；事件數 < 200 則判定樣本不足 |
| H-016 | 波動調整倉位只在特定市況（如 2024 年後的高波動趨勢）改善風險調整後報酬；前瞻資料上仍成立 | second_moment_regime | open | 32.0 | 2026-11-12 起至少 12 個月的前瞻資料上，MANAGED 對 CONST_EQUAL_EXPOSURE 的 Sharpe 差 <= 0 |
| H-003 | US macro surprises (actual vs forecast) cause post-release gold drift | event_surprise | open | 31.0 | post-release returns in the surprise direction are not positive beyond spread at the pre-declared horizon in 2014-2022 |
| H-010 | 重要美國數據公布前後的波動放大可由事件類型和預期差大小預測；公布前後停止進場能改善 M1 策略 | event_volatility | open | 31.0 | 公布後／公布前實現波動比與 &#124;預期差&#124; 的等級相關 <= 0，或 ±30 分鐘不交易使回放淨 R 改善 < 0.02R |
| H-017 | H1 C runner 在換日前平倉、隔日重新評估，能在今日 swap 水準下改善每筆淨 R | cost_structure | open | 29.5 | 在 2024+ 回放與前瞻 demo 上，換日前平倉版本的平均淨 R 不高於原版，或差距 < 0.02R |
| H-007 | For M1 trading, execution cost (hour-of-day spread, entry method) matters more than the model | execution | open | 28.0 | restricting to the cheapest hours does not raise replay mean R by more than 0.03R |
| H-002 | Exogenous macro data work at their native weekly horizon | exogenous_horizon | underpowered | 25.5 | discovery (2016-2022) IC in the wrong direction or &#124;t&#124; < 1 |
| H-006 | Strategy timing skill persists over months (premise of any regime switch or bandit) | adaptivity | parked | 20.0 | rank correlation of consecutive 3-month residuals <= 0 |
| H-019 | 對波動水準不敏感的 HAR（log-HAR 或無截距比例形式）凍結參數後不輸 EWMA | second_moment_forecast | open | 19.5 | 前瞻至少 250 個完整交易日，凍結 log-HAR 的 QLIKE 不低於 EWMA_094（DM 單尾 p >= 0.05） |
| H-015 | 同一組價格／技術指標特徵的方向預測變體（Gen1-21、S4、經典規則、網路策略、LSTM、組合規則）含有可交易的進場時機優勢 | price_direction_variants | exhausted_needs_new_data | 5.0 | 前瞻資料上，預先凍結的候選時機殘差（扣當月漂移）連續 6 個月 <= 0 |

## 2b. 證據狀態分類（結論不得超出這個範圍）

- **證據推翻**：H-004 Gold's drift concentrates in specific sessions; intraday-only longs capture it without swap；H-005 Scaling long exposure by inverse forecast volatility improves risk-adjusted drift harvesting；H-011 HAR 實現波動模型對次日波動的預測優於 EWMA，且高波動狀態下的報酬分布（偏態、尾部）顯著不同
- **未找到顯著優勢（不代表不存在）**：H-001 Candidate H1 edges are drift exposure, not timing skill
- **樣本不足，無法判定**：H-002 Exogenous macro data work at their native weekly horizon
- **已過度研究，需要新資料才能判定**：H-015 同一組價格／技術指標特徵的方向預測變體（Gen1-21、S4、經典規則、網路策略、LSTM、組合規則）含有可交易的進場時機優勢

## 3. 失敗的實驗與被推翻的假說

- **H-004** Gold's drift concentrates in specific sessions; intraday-only longs capture it without swap（refuted）
- **H-005** Scaling long exposure by inverse forecast volatility improves risk-adjusted drift harvesting（refuted）
- **H-011** HAR 實現波動模型對次日波動的預測優於 EWMA，且高波動狀態下的報酬分布（偏態、尾部）顯著不同（refuted）
- 實驗 **E-001**（H-001）：timing skill beyond monthly drift (2018-2023 CI > 0 and 2024+ > 0): ['H1_C_frozen']; controls OK  
  失敗原因：`none`；學到：timing skill exists beyond drift for H1_C_frozen
- 實驗 **E-007**（H-005）：Sharpe MANAGED - CONST_EQUAL_EXPOSURE: 2018-2023 -0.061 (CI [-0.292, 0.162]), 2024+ 0.524; cost x3.0 2018-2023 -0.072; shuffled >= managed 0.65  
  失敗原因：`information`；學到：inverse-vol sizing at equal exposure: refuted
- 實驗 **E-005**（H-004）：refuted: sessions with excess over proportional drift passing all rules: none; best 2018-2022 excess t 0.75 (Sidak p over 8: 0.872)  
  失敗原因：`information`；學到：direction question: refuted; see report for cost sensitivity
- 實驗 **E-008**（H-011）：refuted: FROZEN h=1 2018-2023 QLIKE HAR/EWMA_FIT 1.1392 (DM t 2.94, p 0.9984), HAR better in 1/6 years, regimes {'low': 1.1849, 'mid': 1.1215, 'high': 0.74}; DM t lag 22 2.26; scaled-Parkinson agrees True (descriptive)  
  失敗原因：`information`；學到：HAR vs EWMA forecasting (layer 1): refuted

## 4. 受資料重複使用影響的結論

- 實驗 **E-001**：refuted（證據等級 `reuse_contaminated`，登錄前已有 2 次使用重疊）：timing skill beyond monthly drift (2018-2023 CI > 0 and 2024+ > 0): ['H1_C_frozen']; controls OK
- 實驗 **E-002**：inconclusive（證據等級 `development`，登錄前已有 5 次使用重疊）：weekly-horizon verdicts: {'REAL10Y_CHG_20D': 'refuted', 'COT_MM_NET_PCT_OI': 'inconclusive'}
- 實驗 **E-006**：inconclusive（證據等級 `development`，登錄前已有 14 次使用重疊）：inconclusive: Sharpe INTRADAY_LONG vs ALWAYS_LONG (cost x1): 2018-2022 0.152 vs -0.029, 2023+ 0.999 vs 0.773; better in both windows at cost x1.5, live31 and low swap: False
- 實驗 **E-007**：refuted（證據等級 `development`，登錄前已有 18 次使用重疊）：Sharpe MANAGED - CONST_EQUAL_EXPOSURE: 2018-2023 -0.061 (CI [-0.292, 0.162]), 2024+ 0.524; cost x3.0 2018-2023 -0.072; shuffled >= managed 0.65
- 實驗 **E-005**：refuted（證據等級 `development`，登錄前已有 12 次使用重疊）：refuted: sessions with excess over proportional drift passing all rules: none; best 2018-2022 excess t 0.75 (Sidak p over 8: 0.872)
- 實驗 **E-008**：refuted（證據等級 `development`，登錄前已有 20 次使用重疊）：refuted: FROZEN h=1 2018-2023 QLIKE HAR/EWMA_FIT 1.1392 (DM t 2.94, p 0.9984), HAR better in 1/6 years, regimes {'low': 1.1849, 'mid': 1.1215, 'high': 0.74}; DM t lag 22 2.26; scaled-Parkinson agrees True (descriptive)

## 5. 已登錄、尚未有結果的實驗

- ~~E-003~~ 已作廢（執行前發現設計缺陷，見決策紀錄）
- ~~E-004~~ 已作廢（執行前發現設計缺陷，見決策紀錄）

## 6. 所有實驗結果與下一個問題

- **E-001**（H-001）`refuted` / `reuse_contaminated`：timing skill beyond monthly drift (2018-2023 CI > 0 and 2024+ > 0): ['H1_C_frozen']; controls OK  
  下一個問題：is the skill stable forward (fresh data only)?  
  紀錄：`training_runs/20261111T111328Z_research_e001_exp_edge_decomposition_v1`
- **E-002**（H-002）`inconclusive` / `development`：weekly-horizon verdicts: {'REAL10Y_CHG_20D': 'refuted', 'COT_MM_NET_PCT_OI': 'inconclusive'}  
  下一個問題：if weekly horizon also fails, these exogenous families carry no usable edge for XM gold CFDs  
  紀錄：`training_runs/20261111T140822Z_research_e002_exp_information_horizon_v1`
- **E-006**（H-008）`inconclusive` / `development`：inconclusive: Sharpe INTRADAY_LONG vs ALWAYS_LONG (cost x1): 2018-2022 0.152 vs -0.029, 2023+ 0.999 vs 0.773; better in both windows at cost x1.5, live31 and low swap: False  
  下一個問題：session structure carries no usable edge at these costs; prefer H-003/H-009+ questions  
  紀錄：`training_runs/20261111T144849Z_research_e006_exp_session_structure_v2`
- **E-007**（H-005）`refuted` / `development`：Sharpe MANAGED - CONST_EQUAL_EXPOSURE: 2018-2023 -0.061 (CI [-0.292, 0.162]), 2024+ 0.524; cost x3.0 2018-2023 -0.072; shuffled >= managed 0.65  
  下一個問題：second-moment sizing alone is not an edge; test H-011 forecasting quality  
  紀錄：`training_runs/20261111T150249Z_research_e007_exp_vol_managed_v2`
- **E-005**（H-004）`refuted` / `development`：refuted: sessions with excess over proportional drift passing all rules: none; best 2018-2022 excess t 0.75 (Sidak p over 8: 0.872)  
  下一個問題：session structure carries no usable edge at these costs; prefer H-003/H-009+ questions  
  紀錄：`training_runs/20261111T150317Z_research_e005_exp_session_structure_v2`
- **E-008**（H-011）`refuted` / `development`：refuted: FROZEN h=1 2018-2023 QLIKE HAR/EWMA_FIT 1.1392 (DM t 2.94, p 0.9984), HAR better in 1/6 years, regimes {'low': 1.1849, 'mid': 1.1215, 'high': 0.74}; DM t lag 22 2.26; scaled-Parkinson agrees True (descriptive)  
  下一個問題：keep EWMA as the volatility input for H-014/H-018; test whether GVZ adds information  
  紀錄：`training_runs/20261112T020611Z_research_e008_exp_vol_forecast_v1`

## 7. 決策紀錄

- **D-001** Stop new price-only short-horizon timing variants (new indicators, thresholds, TP/SL, model swaps on the same features)：Gen1-21, the S4 family and the 2026-11 labs (O-008) cover this space; every positive result so far looks like drift (O-004). New work must add information, change the cost structure, or test a premise
- **D-002** Only data after 2026-11-12 or newly acquired sources can be confirmatory：all GOLD# history up to 2026-11-11 was inspected; the ledger refuses confirmatory preregistrations on used windows
- **D-003** Start with E-001 and E-002：highest information gain per cost: E-001 tells whether price-only timing has any skill beyond drift; E-002 whether the closed exogenous families were closed for the wrong reason
- **D-004** 更正 E-001：排除樣本內的 H1_C 後，沒有候選在兩個區間都有正時機殘差，H-001（候選優勢只是順勢報酬）視為成立；E-002 實質利率在週線也失敗，COT 僅弱且未勝過一直做多，H-002 inconclusive；H-006 前提不成立，暫停。：實驗設計錯誤：模型候選的評估區間必須排除其訓練區間。之後的預先登記須寫明每個候選的訓練區間，並只在其後的資料評估。
- **D-005** Start E-003 (H-004) and E-004 (H-005)：D-004 left price-only timing closed; these test cost structure and second-moment sizing, neither of which predicts direction
- **D-006** 依證據強度重新分類並縮小結論範圍：H-001 改為『未找到顯著優勢』、H-002 改為『樣本不足』；D-001 的範圍縮小為『同一組價格特徵的方向預測參數／模型變體已過度研究，需要新資料』；作廢 E-003、E-004（執行前審查發現設計缺陷，從未產生結果）；H-004 只保留時段方向問題，成本問題移到 H-008：H-001：只檢驗了 3 個固定候選（TP3/SL1.5/48h），H1_C 樣本外只有 274 筆，無法偵測 < 0.15R 的時機能力；這不代表價格型策略或機器學習沒有研究價值。H-002：實質利率部分符合預先登記的推翻條件，COT 方向在兩段一致但 |t| < 1.4，資料只到 2024。D-001：更多同類變體在同一段歷史資料上無法產生獨立證據，只能用 2026-11-12 之後的資料或新資訊、新預測目標（H-011、H-013）判定。E-003 缺：成本與方向未分開、無方向曝險對照、缺換日段、無成本壓力測試與 Sharpe/Calmar/回撤；E-004 缺：與 w=1 比較的風險預算不同、無權重打亂對照、成本與 swap 只有單一假設。
- **D-007** E-005、E-006、E-007 判讀：H-004 推翻（範圍：4 個固定時段、每天 >= 約 4 bp 的方向效果）；H-005 依預先登記規則推翻（2018-2023 無改善），但 2024+ 有強烈效果，另立 H-016 用前瞻資料確認；H-008 維持 inconclusive（結論取決於 swap 水準：今日利率下盤中持有在所有點差情境都較好，swap 減半時 2018-2022 沒有優勢），另立 H-017 檢驗套用到 H1 C runner：不在看到結果後改判定規則；E-007 兩段期間相反，屬於 regime_dependence，不能用已看過的 2024+ 再確認；E-005 的推翻門檻偏寬（審查員註記），所以只排除大效果；E-006 的唯一失敗格是 2018-2022 swap x0.5，因此結論應限定在『今日 swap 水準』
- **D-008** H-011 分兩層：E-008 只檢驗統計預測能力（H-011 的預測子句）；條件報酬分布子句延後；交易決策價值另立 H-018，於 E-008 有結果後才預先登記 E-009。E-008 結果如何對應下一步：supported → E-009 用 HAR 凍結預測；refuted／not_significant → E-009 改用 EWMA，問題變成『任何波動預測是否改善決策』；無論哪個結果，H-014（GVZ）以 E-008 中表現最好的『FROZEN 2014-2017 參數』模型為基準，不得改用看過 2024+ 後選的模型：避免把『預測誤差較小』和『可以賺錢』混在同一個實驗；兩層都要能各自被推翻
- **D-009** E-008 依預先登記規則判為 refuted：凍結 2014-2017 參數的 HAR 在 2018-2023 h=1 QLIKE 顯著差於 EWMA_FIT（比值 1.139，DM t 2.94，lag 22 t 2.26，6 年中僅 1 年較好）。H-011 預測子句設為 refuted，範圍限於『此設定、凍結參數』；不代表 HAR 類模型無用，也不涵蓋條件報酬分布子句。依 D-008：E-009（H-018）改用 EWMA_094 作波動輸入，問題是『任何波動預測是否改善決策』；H-014（GVZ）以 EWMA_094 為基準。EWMA_094 是 E-008 主要窗口中最佳的 FROZEN 模型，且沒有任何擬合參數。不得改用 RECURSIVE HAR 或 2024+ 表現較好的模型。H-019 只能用前瞻資料判定：照預先登記執行，不在看到結果後改寫判定；EWMA_094 原本就是預先宣告的對照，選它不引入新的擬合；但它是依 2018-2023 結果選出，所以 E-009／H-014 在 2018-2023 的結果須標示此選擇

## 8. 資料區間使用帳（越多次越不能當獨立驗證）

| 區間 | 使用者 |
|---|---|
| 2012-01-01 → 2026-11-12 | pre-ledger research |
| 2018-01-01 → 2024-01-01 | E-001 |
| 2024-01-01 → 2026-11-12 | E-001 |
| 2016-01-01 → 2023-01-01 | E-002 |
| 2023-01-01 → 2025-01-01 | E-002 |
| 2018-01-01 → 2023-01-01 | E-003 |
| 2023-01-01 → 2099-01-01 | E-003 |
| 2018-01-01 → 2024-01-01 | E-004 |
| 2024-01-01 → 2099-01-01 | E-004 |
| 2018-01-01 → 2023-01-01 | E-005 |
| 2023-01-01 → 2099-01-01 | E-005 |
| 2018-01-01 → 2023-01-01 | E-006 |
| 2023-01-01 → 2099-01-01 | E-006 |
| 2018-01-01 → 2024-01-01 | E-007 |
| 2024-01-01 → 2099-01-01 | E-007 |
| 2018-01-01 → 2024-01-01 | E-008 |
| 2024-01-01 → 2099-01-01 | E-008 |

## 9. 研究空間覆蓋與自動產生的候選問題

探索次數（歷史研究＋帳本中已標註的假說）。0 代表從未研究過。

| 資訊來源 | 取得狀態 | 漲跌方向 | 未來波動 | 報酬分布／尾部 | 值不值得交易 | 成本與執行 | 市況轉換 |
|---|---|---|---|---|---|---|---|
| 價格 K 線 | in_hand | 29 | 5 | 1 | 2 | 3 | 3 |
| Tick／點差動態 | in_hand (MT5 ticks, months of history) | 2 | 1 | 0 | 1 | 2 | 0 |
| 經濟事件實際值／預期值 | export_needed (ExportUsdCalendarValues) | 3 | 1 | 0 | 1 | 0 | 0 |
| 利率／總經 | in_hand (to 2024-12) | 3 | 0 | 0 | 0 | 0 | 0 |
| COT 部位 | in_hand (to 2024-12) | 2 | 0 | 0 | 0 | 0 | 0 |
| 跨市場（美元、白銀、指數） | in_hand | 2 | 0 | 0 | 0 | 0 | 0 |
| 隱含波動率（CBOE GVZ） | new_source (FRED GVZCLS, free) | 1 | 1 | 0 | 0 | 0 | 0 |
| 新聞文字 | new_source (no feed yet) | 0 | 0 | 0 | 0 | 0 | 0 |

**自動產生的候選研究問題**（尚未成為假說；要成為假說需補上機制、推翻條件，並通過審查）：

- COT 部位 是否含有預測 GOLD# **未來波動** 的資訊？（資料：in_hand (to 2024-12)）
- COT 部位 是否含有預測 GOLD# **報酬分布／尾部** 的資訊？（資料：in_hand (to 2024-12)）
- 跨市場（美元、白銀、指數） 是否含有預測 GOLD# **未來波動** 的資訊？（資料：in_hand）
- 跨市場（美元、白銀、指數） 是否含有預測 GOLD# **報酬分布／尾部** 的資訊？（資料：in_hand）
- 利率／總經 是否含有預測 GOLD# **未來波動** 的資訊？（資料：in_hand (to 2024-12)）
- 利率／總經 是否含有預測 GOLD# **報酬分布／尾部** 的資訊？（資料：in_hand (to 2024-12)）
- Tick／點差動態 是否含有預測 GOLD# **報酬分布／尾部** 的資訊？（資料：in_hand (MT5 ticks, months of history)）
- Tick／點差動態 是否含有預測 GOLD# **市況轉換** 的資訊？（資料：in_hand (MT5 ticks, months of history)）
