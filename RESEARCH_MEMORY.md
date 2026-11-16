# XM 研究記憶（自動產生，請勿手改）

由 `research_memory_v1.py render` 從 `research_memory/ledger.jsonl` 產生。事件數 107；帳本完整性：OK。

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
- **O-016** XM demo 帳戶的 GOLD# swap_mode = 0（SYMBOL_SWAP_MODE_DISABLED），雖然 swap_long 報 -78.59。唯讀檢查 400 天內 575 筆已平倉部位：9 筆跨過伺服器換日（多空皆有，含跨週三三倍夜的多單 2026-04-14→04-16），swap 全為 0。E-001、E-006、E-007 以每年約 7% 的 swap 計成本，對這個帳戶不成立；真實帳戶可能不同，需另行確認  
  證據：read-only MT5 symbol_info and history_deals_get on 2026-11-12 (no orders sent)
- **O-018** E-013（依預先登記判為 supported）：2014-2017 的 VR_day 0.434（CI 0.39-0.49），2018-2023 為 0.932（CI 0.85-1.02），差 0.498（CI 0.41-0.60），週頻差 0.458。逐年 VR_day：2014 0.67、2015 0.45、2016 0.35、2017 0.36、2018 0.69（過渡）、2019-2023 0.88-1.02、2024-2026 1.02-1.16。2018 年前伺服器 0 時與 1 時兩根 H1 合占全部 H1 報酬平方的 58%（各約 29%），2018 年後約 9%；0 時那根只在 2018 年前存在（每日 24 根→23 根，變更月 2018-04）；0 時與 1 時份額相等，呈現換日時段報價先跳開再回歸的型態。RV_H1／Parkinson 2014-2017 為 2.30，之後約 1.0。經變異比調整後，2018-2023 對 2014-2017 的 RV 水準比由 0.465 變為 0.999：所謂 2018 年後波動減半幾乎全部來自這個型態。SPREAD 欄位在 2013-2018 有一半以上為 0（變更月 2018-10），屬成本欄位缺失。XM 自家交叉匯率殘差 2017 下半年 1.49，2018 與 2020 約 0.55，2023 年後約 0.002，品質逐年改善。按預先登記的措辭，這是『與報價雜訊相符的日內反轉過多』，形式上不能排除真實的日內均值回歸；但集中在換日一小時並在下一小時回歸、規模達日變異的一半以上，解讀上指向換日時段的報價瑕疵。外部資料源無法取得，資料源層級差異未判定  
  證據：training_runs/20261112T154049Z_research_e013_exp_data_quality_v1/report.md
- **O-019** 訓練資料稽核（讀程式與紀錄，未重跑）：H1 C（frozen/h1_C_long_v1）以 GOLD# H1 擬合 2014-01-01 至 2022-12-31（依持有期 purge），2023 挑門檻，2024-01-01 起測試。16 個特徵中 ATR_PCT、BB_WIDTH、VOLA_RATIO、BODY_PCT、ROC_5、HOUR_SIN/COS 直接取自 H1 K 棒，標籤是以 H1 ATR 為單位（TP 3.0／SL 1.5）在 H1 高低價上的先觸及，兩者都暴露在 E-013 的換日時段型態（2014-2017 的 0、1 時 K 棒）之下；2018 年前 SPREAD=0，因此以價格的 0.008% 作成本下限。結論：H1 C 的擬合期受影響（約 5 年中有 4 年在 2014-2017、1 年在過渡年 2018），2023 門檻期與 2024+ 測試期不受影響。gemini.py 載入的 gold_long_recent_candidate_xgb.json 由 gold_long_recent_walk_forward.py 以 M1 訓練，資料起點 2025-01-01（驗證 2026-04-01、測試 2026-06-01），高階週期暖機最多回推 760 天（約 2022-12），全部在 2019 年之後，不受 E-013 揭露的問題影響  
  證據：frozen/h1_C_long_v1/h1_lab_v1.py（docstring 與 FEATURES）、frozen/h1_C_long_v1/config.json、gold_long_recent_walk_forward.json（data.start／validation_start／test_start）、gold_recent_walk_forward.py

## 2. 尚未驗證的假說（依 Research Priority Engine 排序）

分數綜合資訊增益、與過去研究的差異、可反證性、資料品質與取得成本、經濟價值、工程簡單度，扣掉資料探勘風險與同族群已試次數。失敗但能排除重要可能性的實驗同樣得分。公式見 `research_planner_v1.py`。
「原始分」是輸入值算出的分數；「系統自動調整」由依賴圖推得，每項都附決策與證據編號，屬於系統建議，狀態變更仍需決策核准（見 2c）。前提失效的假說不排序（見 2a）。

| ID | 項目 | 狀態 | 原始分 | 系統自動調整（證據） | 最終分 | 推翻條件 |
|---|---|---|---|---|---|---|
| H-013 | 可以預測未來 1 小時的波動是否大於 k 倍點差（不預測方向），而 gemini 的虧損集中在預測『不值得交易』的時段 | open | 36.0 | — | 36.0 | 樣本外波動大小分類 AUC <= 0.55，或低可交易性組的 gemini 回放 R 沒有比其他組低 0.05R 以上 |
| H-009 | 點差相對同時段常態異常擴大時，接下來的波動與滑價較大，是不值得進場的時刻 | testing | 34.0 | — | 34.0 | 點差 z 值最高 10% 時段之後 30 分鐘的實現波動比 < 1.2，或排除這些時段進場使既有策略回放淨 R 改善 < 0.02R |
| H-012 | 低流動時段的單根急動（> 3 倍 ATR）在 15-60 分鐘內部分回補，扣除當下實際點差後期望為正 | open | 33.0 | — | 33.0 | 2018-2022 以實際點差計的平均淨回補 <= 0；事件數 < 200 則判定樣本不足 |
| H-016 | 波動調整倉位只在特定市況（如 2024 年後的高波動趨勢）改善風險調整後報酬；前瞻資料上仍成立 | open | 32.0 | — | 32.0 | 2026-11-12 起至少 12 個月的前瞻資料上，MANAGED 對 CONST_EQUAL_EXPOSURE 的 Sharpe 差 <= 0 |
| H-003 | US macro surprises (actual vs forecast) cause post-release gold drift | open | 31.0 | — | 31.0 | post-release returns in the surprise direction are not positive beyond spread at the pre-declared horizon in 2014-2022 |
| H-010 | 重要美國數據公布前後的波動放大可由事件類型和預期差大小預測；公布前後停止進場能改善 M1 策略 | open | 31.0 | — | 31.0 | 公布後／公布前實現波動比與 &#124;預期差&#124; 的等級相關 <= 0，或 ±30 分鐘不交易使回放淨 R 改善 < 0.02R |
| H-007 | For M1 trading, execution cost (hour-of-day spread, entry method) matters more than the model | open | 28.0 | — | 28.0 | restricting to the cheapest hours does not raise replay mean R by more than 0.03R |
| H-002 | Exogenous macro data work at their native weekly horizon | underpowered | 25.5 | — | 25.5 | discovery (2016-2022) IC in the wrong direction or &#124;t&#124; < 1 |
| data:gold_h1_2014_2017 | 上游修復：處理 data:gold_h1_2014_2017 並重新驗證 H-018、H-019、H-020、H-022、model:h1_c | repair | — | +0 換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%）（D-017, O-018, E-013） | 25.5 | — |
| data:gold_h1_2018 | 上游修復：處理 data:gold_h1_2018 並重新驗證 H-018、model:h1_c | repair | — | +0 過渡年（VR_day 0.69）；D-016 規定 H1 擬合與校準自 2019-01-01 起（D-017, O-018） | 23.0 | — |
| data:gold_spread_field_pre_2018_10 | 上游修復：處理 data:gold_spread_field_pre_2018_10 並重新驗證 H-018、model:h1_c | repair | — | +0 SPREAD 欄位一半以上為 0（D-017, O-018） | 23.0 | — |
| H-006 | Strategy timing skill persists over months (premise of any regime switch or bandit) | parked | 20.0 | — | 20.0 | rank correlation of consecutive 3-month residuals <= 0 |
| H-022 | 只做水準重新校準（每月一個尺度參數，用過去資料）的 EWMA_094 預測，在前瞻資料上優於凍結與未校準版本 | needs_revalidation | 23.5 | -4 資料品質：上游資料有缺陷，data_quality 降為 1（D-017, E-013, O-018, data:gold_h1_2014_2017）<br>-5 依賴證據可信度：H-022 → O-014 → E-008 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 凍結 HAR 以 2014-2017 擬合）；H-022 → O-014 → E-010 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 凍結 EWMA／GVZ 模型以 2014-2017 擬合）；H-022 → O-015 → E-011 → E-008 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 凍結 HAR 以 2014-2017 擬合）（D-017, E-013, O-018） | 14.5 | 前瞻至少 250 個完整交易日，校準版 QLIKE 不低於原始 EWMA_094（DM 單尾 p >= 0.05） |
| H-018 | 波動預測能改善交易決策：倉位風險控制、實際回撤、辨識不值得交易的環境，且改善足以抵銷額外交易成本 | needs_revalidation | 22.5 | -4 資料品質：上游資料有缺陷，data_quality 降為 1（D-017, E-013, O-018, data:gold_h1_2014_2017, data:gold_h1_2018, data:gold_spread_field_pre_2018_10）<br>-5 依賴證據可信度：H-018 → E-012 → model:h1_c → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 擬合 2014-2022；2023 門檻與 2024+ 測試不受影響，保留）；H-018 → E-012 → model:h1_c → data:gold_h1_2018（defective：過渡年（VR_day 0.69）；D-016 規定 H1 擬合與校準自 2019-01-01 起）；H-018 → E-012 → model:h1_c → data:gold_spread_field_pre_2018_10（defective：SPREAD 欄位一半以上為 0 / 以 0.008% 成本下限代替）（D-017, E-013, O-018） | 13.5 | 2018-2023 以凍結預測做風險目標倉位與不交易濾網，在相同平均曝險下，扣除額外換手成本後 Sharpe 不高於固定風險版本，且最大回撤降低 < 10% |
| H-020 | E-008／E-010 固定參數模型的退化可分解為可量測的來源（目標分布、共變量、概念、校準、資料品質），且主要來源可在前瞻資料上單獨檢驗 | needs_revalidation | 19.5 | -3 資料品質：上游資料有缺陷，data_quality 降為 1（D-017, E-013, O-018, data:gold_h1_2014_2017）<br>-5 依賴證據可信度：H-020 → O-013 → E-008 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 凍結 HAR 以 2014-2017 擬合）；H-020 → O-014 → E-008 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 凍結 HAR 以 2014-2017 擬合）；H-020 → O-014 → E-010 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 凍結 EWMA／GVZ 模型以 2014-2017 擬合）（D-017, E-013, O-018） | 11.5 | 描述性診斷本身不判定；後續每個解釋各自預先登記前瞻檢驗 |
| H-019 | 對波動水準不敏感的 HAR（log-HAR 或無截距比例形式）凍結參數後不輸 EWMA | needs_revalidation | 19.5 | -4 資料品質：上游資料有缺陷，data_quality 降為 1（D-017, E-013, O-018, data:gold_h1_2014_2017）<br>-5 依賴證據可信度：H-019 → O-013 → E-008 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 凍結 HAR 以 2014-2017 擬合）（D-017, E-013, O-018） | 10.5 | 前瞻至少 250 個完整交易日，凍結 log-HAR 的 QLIKE 不低於 EWMA_094（DM 單尾 p >= 0.05） |
| H-015 | 同一組價格／技術指標特徵的方向預測變體（Gen1-21、S4、經典規則、網路策略、LSTM、組合規則）含有可交易的進場時機優勢 | exhausted_needs_new_data | 5.0 | — | 5.0 | 前瞻資料上，預先凍結的候選時機殘差（扣當月漂移）連續 6 個月 <= 0 |

## 2a. 前提失效（不排序；不等於假說被推翻）

- **H-008** 收盤前平倉、只做盤中多單，比一直持有多單（付過夜利息）的風險調整後報酬更好（核准狀態 premise_invalid）：H-008 → premise:overnight_swap_charged（invalid：本 demo 帳戶 swap_mode=DISABLED，跨換日部位未被收 swap（真實帳戶另需確認） / 隔夜持有多單要付 swap）
- **H-017** H1 C runner 在換日前平倉、隔日重新評估，能在今日 swap 水準下改善每筆淨 R（核准狀態 premise_invalid）：H-017 → premise:overnight_swap_charged（invalid：本 demo 帳戶 swap_mode=DISABLED，跨換日部位未被收 swap（真實帳戶另需確認））；H-017 → model:h1_c → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 擬合 2014-2022；2023 門檻與 2024+ 測試不受影響，保留）；H-017 → model:h1_c → data:gold_h1_2018（defective：過渡年（VR_day 0.69）；D-016 規定 H1 擬合與校準自 2019-01-01 起）

## 2c. 系統自動建議（尚未核准；核准須以決策 set_status 記錄）

| 節點 | 核准狀態 | 系統建議 | 原因（依賴路徑） | 證據 |
|---|---|---|---|---|
| E-001 | — | annotate_only | E-001 → model:h1_c → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 擬合 2014-2022；2023 門檻與 2024+ 測試不受影響，保留）；E-001 → model:h1_c → data:gold_h1_2018（defective：過渡年（VR_day 0.69）；D-016 規定 H1 擬合與校準自 2019-01-01 起）；E-001 → model:h1_c → data:gold_spread_field_pre_2018_10（defective：SPREAD 欄位一半以上為 0 / 以 0.008% 成本下限代替）；E-001 → data:gold_h1_2018（defective：過渡年（VR_day 0.69）；D-016 規定 H1 擬合與校準自 2019-01-01 起 / 開發期 2018-2023 的 H1 進場與報酬） | D-017, E-013, O-018 |
| E-005 | — | annotate_only | E-005 → data:gold_h1_2018（defective：過渡年（VR_day 0.69）；D-016 規定 H1 擬合與校準自 2019-01-01 起 / 時段報酬以 H1 計算，開發期 2018-2022） | D-017, O-018 |
| E-006 | — | annotate_only | E-006 → data:gold_h1_2018（defective：過渡年（VR_day 0.69）；D-016 規定 H1 擬合與校準自 2019-01-01 起 / 盤中／持有比較以 H1 計算，開發期 2018-2022） | D-017, O-018 |
| E-008 | — | annotate_only | E-008 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 凍結 HAR 以 2014-2017 擬合） | D-017, E-013, O-018 |
| E-010 | — | annotate_only | E-010 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 凍結 EWMA／GVZ 模型以 2014-2017 擬合） | D-017, E-013, O-018 |
| E-011 | — | annotate_only | E-011 → E-008 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 凍結 HAR 以 2014-2017 擬合）；E-011 → E-010 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 凍結 EWMA／GVZ 模型以 2014-2017 擬合）；E-011 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 與 2014-2017 分布比較） | D-017, E-013, O-018 |
| E-012 | — | annotate_only | E-012 → model:h1_c → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 擬合 2014-2022；2023 門檻與 2024+ 測試不受影響，保留）；E-012 → model:h1_c → data:gold_h1_2018（defective：過渡年（VR_day 0.69）；D-016 規定 H1 擬合與校準自 2019-01-01 起）；E-012 → model:h1_c → data:gold_spread_field_pre_2018_10（defective：SPREAD 欄位一半以上為 0 / 以 0.008% 成本下限代替）；E-012 → data:gold_h1_2018（defective：過渡年（VR_day 0.69）；D-016 規定 H1 擬合與校準自 2019-01-01 起 / 校準窗 2018-2023） | D-017, E-013, O-018 |
| H-001 | not_significant | annotate_only | H-001 → E-001 → model:h1_c → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 擬合 2014-2022；2023 門檻與 2024+ 測試不受影響，保留）；H-001 → E-001 → model:h1_c → data:gold_h1_2018（defective：過渡年（VR_day 0.69）；D-016 規定 H1 擬合與校準自 2019-01-01 起）；H-001 → E-001 → model:h1_c → data:gold_spread_field_pre_2018_10（defective：SPREAD 欄位一半以上為 0 / 以 0.008% 成本下限代替）；H-001 → E-001 → data:gold_h1_2018（defective：過渡年（VR_day 0.69）；D-016 規定 H1 擬合與校準自 2019-01-01 起 / 開發期 2018-2023 的 H1 進場與報酬） | D-017, E-013, O-018 |
| H-004 | refuted | annotate_only | H-004 → E-005 → data:gold_h1_2018（defective：過渡年（VR_day 0.69）；D-016 規定 H1 擬合與校準自 2019-01-01 起 / 時段報酬以 H1 計算，開發期 2018-2022） | D-017, O-018 |
| H-011 | refuted | annotate_only | H-011 → E-008 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 凍結 HAR 以 2014-2017 擬合） | D-017, E-013, O-018 |
| H-014 | refuted | annotate_only | H-014 → E-010 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 凍結 EWMA／GVZ 模型以 2014-2017 擬合） | D-017, E-013, O-018 |
| O-013 | — | annotate_only | O-013 → E-008 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 凍結 HAR 以 2014-2017 擬合） | D-017, E-013, O-018 |
| O-014 | — | annotate_only | O-014 → E-008 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 凍結 HAR 以 2014-2017 擬合）；O-014 → E-010 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 凍結 EWMA／GVZ 模型以 2014-2017 擬合） | D-017, E-013, O-018 |
| O-015 | — | annotate_only | O-015 → E-011 → E-008 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 凍結 HAR 以 2014-2017 擬合）；O-015 → E-011 → E-010 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 凍結 EWMA／GVZ 模型以 2014-2017 擬合）；O-015 → E-011 → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 與 2014-2017 分布比較） | D-017, E-013, O-018 |
| O-017 | — | annotate_only | O-017 → E-012 → model:h1_c → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 擬合 2014-2022；2023 門檻與 2024+ 測試不受影響，保留）；O-017 → E-012 → model:h1_c → data:gold_h1_2018（defective：過渡年（VR_day 0.69）；D-016 規定 H1 擬合與校準自 2019-01-01 起）；O-017 → E-012 → model:h1_c → data:gold_spread_field_pre_2018_10（defective：SPREAD 欄位一半以上為 0 / 以 0.008% 成本下限代替）；O-017 → E-012 → data:gold_h1_2018（defective：過渡年（VR_day 0.69）；D-016 規定 H1 擬合與校準自 2019-01-01 起 / 校準窗 2018-2023） | D-017, E-013, O-018 |
| model:h1_c | — | annotate_only | model:h1_c → data:gold_h1_2014_2017（defective：換日時段型態放大 H1 實現波動（VR_day 0.43，0/1 時占 58%） / 擬合 2014-2022；2023 門檻與 2024+ 測試不受影響，保留）；model:h1_c → data:gold_h1_2018（defective：過渡年（VR_day 0.69）；D-016 規定 H1 擬合與校準自 2019-01-01 起）；model:h1_c → data:gold_spread_field_pre_2018_10（defective：SPREAD 欄位一半以上為 0 / 以 0.008% 成本下限代替） | D-017, E-013, O-018 |

`annotate_only`：已有結論的假說、實驗結果、觀察與模型只標註其證據受影響，不改變狀態，也不宣稱過去策略失效。

## 2d. 重排紀錄（每次改變引擎輸入的決策造成的名次與分數變化）

- **D-004**：H-001 None → 未列入（名次 1 → —）
- **D-006**：H-002 None → 25.5（名次 8 → 12）；H-003 None → 31.0（名次 9 → 8）；H-004 None → 28.0（名次 10 → 10）；H-005 None → 38.0（名次 11 → 3）；H-006 None → 20.0（名次 12 → 13）；H-007 None → 28.0（名次 13 → 11）
- **D-007**：H-004 26.5 → 未列入（名次 13 → —）；H-005 36.5 → 未列入（名次 4 → —）
- **D-009**：H-011 36.5 → 未列入（名次 3 → —）
- **D-011**：H-014 37.5 → 未列入（名次 2 → —）
- **D-013**：H-017 29.5 → 24.5（名次 10 → 12）
- **D-014**：H-018 33.5 → 20.5（名次 4 → 13）
- **D-016**：H-021 31.5 → 未列入（名次 6 → —）
- **D-017**：H-008 38.5 → 未列入（名次 1 → —）；H-017 24.5 → 未列入（名次 10 → —）；H-018 20.5 → 11.5（名次 12 → 14）；H-019 19.5 → 10.5（名次 14 → 16）；H-020 19.5 → 11.5（名次 15 → 15）；H-022 23.5 → 14.5（名次 11 → 13）；data:gold_h1_2014_2017 未列入 → 25.5（名次 — → 9）；data:gold_h1_2018 未列入 → 21.0（名次 — → 10）；data:gold_spread_field_pre_2018_10 未列入 → 21.0（名次 — → 11）
- **D-019**：H-018 11.5 → 13.5（名次 14 → 14）；data:gold_h1_2018 21.0 → 23.0（名次 10 → 10）；data:gold_spread_field_pre_2018_10 21.0 → 23.0（名次 11 → 11）

## 2b. 證據狀態分類（結論不得超出這個範圍）

- **證據推翻**：H-004 Gold's drift concentrates in specific sessions; intraday-only longs capture it without swap；H-005 Scaling long exposure by inverse forecast volatility improves risk-adjusted drift harvesting；H-011 HAR 實現波動模型對次日波動的預測優於 EWMA，且高波動狀態下的報酬分布（偏態、尾部）顯著不同；H-014 黃金隱含波動率（CBOE GVZ）與實現波動的差（變異數風險溢酬）預測未來 5-20 日的實現波動與報酬
- **未找到顯著優勢（不代表不存在）**：H-001 Candidate H1 edges are drift exposure, not timing skill
- **樣本不足，無法判定**：H-002 Exogenous macro data work at their native weekly horizon
- **已過度研究，需要新資料才能判定**：H-015 同一組價格／技術指標特徵的方向預測變體（Gen1-21、S4、經典規則、網路策略、LSTM、組合規則）含有可交易的進場時機優勢
- **前提失效（不等於假說被推翻）**：H-008 收盤前平倉、只做盤中多單，比一直持有多單（付過夜利息）的風險調整後報酬更好；H-017 H1 C runner 在換日前平倉、隔日重新評估，能在今日 swap 水準下改善每筆淨 R
- **依賴的證據受影響，需要重新驗證（不等於假說被推翻）**：H-018 波動預測能改善交易決策：倉位風險控制、實際回撤、辨識不值得交易的環境，且改善足以抵銷額外交易成本；H-019 對波動水準不敏感的 HAR（log-HAR 或無截距比例形式）凍結參數後不輸 EWMA；H-020 E-008／E-010 固定參數模型的退化可分解為可量測的來源（目標分布、共變量、概念、校準、資料品質），且主要來源可在前瞻資料上單獨檢驗；H-022 只做水準重新校準（每月一個尺度參數，用過去資料）的 EWMA_094 預測，在前瞻資料上優於凍結與未校準版本

## 3. 失敗的實驗與被推翻的假說

- **H-004** Gold's drift concentrates in specific sessions; intraday-only longs capture it without swap（refuted）
- **H-005** Scaling long exposure by inverse forecast volatility improves risk-adjusted drift harvesting（refuted）
- **H-011** HAR 實現波動模型對次日波動的預測優於 EWMA，且高波動狀態下的報酬分布（偏態、尾部）顯著不同（refuted）
- **H-014** 黃金隱含波動率（CBOE GVZ）與實現波動的差（變異數風險溢酬）預測未來 5-20 日的實現波動與報酬（refuted）
- 實驗 **E-001**（H-001）：timing skill beyond monthly drift (2018-2023 CI > 0 and 2024+ > 0): ['H1_C_frozen']; controls OK  
  失敗原因：`none`；學到：timing skill exists beyond drift for H1_C_frozen
- 實驗 **E-007**（H-005）：Sharpe MANAGED - CONST_EQUAL_EXPOSURE: 2018-2023 -0.061 (CI [-0.292, 0.162]), 2024+ 0.524; cost x3.0 2018-2023 -0.072; shuffled >= managed 0.65  
  失敗原因：`information`；學到：inverse-vol sizing at equal exposure: refuted
- 實驗 **E-005**（H-004）：refuted: sessions with excess over proportional drift passing all rules: none; best 2018-2022 excess t 0.75 (Sidak p over 8: 0.872)  
  失敗原因：`information`；學到：direction question: refuted; see report for cost sensitivity
- 實驗 **E-008**（H-011）：refuted: FROZEN h=1 2018-2023 QLIKE HAR/EWMA_FIT 1.1392 (DM t 2.94, p 0.9984), HAR better in 1/6 years, regimes {'low': 1.1849, 'mid': 1.1215, 'high': 0.74}; DM t lag 22 2.26; scaled-Parkinson agrees True (descriptive)  
  失敗原因：`information`；學到：HAR vs EWMA forecasting (layer 1): refuted
- 實驗 **E-010**（H-014）：refuted: general: refuted (A/R 1.1248, DM t 3.09, Sidak p 1.0, gain -12.48% ub95 -5.83%, years 2/6, lag2 A/R 1.1183, A/P 1.116); extreme: refuted (A/R 1.5593, DM t 5.4, Sidak p 1.0, gain -55.93% ub95 -38.91%, years 0/6, lag2 A/R 1.4282, A/P 1.5058); regime_up: refuted (A/R 1.1101, DM t 2.47, Sidak p 1.0, gain -11.01% ub95 -3.68%, years 2/6, lag2 A/R 1.1025, A/P 1.1069)  
  失敗原因：`information`；學到：GVZ increment over price history: {'general': 'refuted', 'extreme': 'refuted', 'regime_up': 'refuted'}
- 實驗 **E-009**（H-018）：aborted before any evaluation: the pre-run guard found GOLD# swap_mode 0 (DISABLED) where the design assumed swap in points; bars were fetched but no signal, trade or P&L was computed  
  失敗原因：`method`；學到：verify broker cost settings against real deals before preregistering costs; the runner did not catch SystemExit from a module (fixed)

## 4. 受資料重複使用影響的結論

- 實驗 **E-001**：refuted（證據等級 `reuse_contaminated`，登錄前已有 2 次使用重疊）：timing skill beyond monthly drift (2018-2023 CI > 0 and 2024+ > 0): ['H1_C_frozen']; controls OK
- 實驗 **E-002**：inconclusive（證據等級 `development`，登錄前已有 5 次使用重疊）：weekly-horizon verdicts: {'REAL10Y_CHG_20D': 'refuted', 'COT_MM_NET_PCT_OI': 'inconclusive'}
- 實驗 **E-006**：inconclusive（證據等級 `development`，登錄前已有 14 次使用重疊）：inconclusive: Sharpe INTRADAY_LONG vs ALWAYS_LONG (cost x1): 2018-2022 0.152 vs -0.029, 2023+ 0.999 vs 0.773; better in both windows at cost x1.5, live31 and low swap: False
- 實驗 **E-007**：refuted（證據等級 `development`，登錄前已有 18 次使用重疊）：Sharpe MANAGED - CONST_EQUAL_EXPOSURE: 2018-2023 -0.061 (CI [-0.292, 0.162]), 2024+ 0.524; cost x3.0 2018-2023 -0.072; shuffled >= managed 0.65
- 實驗 **E-005**：refuted（證據等級 `development`，登錄前已有 12 次使用重疊）：refuted: sessions with excess over proportional drift passing all rules: none; best 2018-2022 excess t 0.75 (Sidak p over 8: 0.872)
- 實驗 **E-008**：refuted（證據等級 `development`，登錄前已有 20 次使用重疊）：refuted: FROZEN h=1 2018-2023 QLIKE HAR/EWMA_FIT 1.1392 (DM t 2.94, p 0.9984), HAR better in 1/6 years, regimes {'low': 1.1849, 'mid': 1.1215, 'high': 0.74}; DM t lag 22 2.26; scaled-Parkinson agrees True (descriptive)
- 實驗 **E-010**：refuted（證據等級 `development`，登錄前已有 22 次使用重疊）：refuted: general: refuted (A/R 1.1248, DM t 3.09, Sidak p 1.0, gain -12.48% ub95 -5.83%, years 2/6, lag2 A/R 1.1183, A/P 1.116); extreme: refuted (A/R 1.5593, DM t 5.4, Sidak p 1.0, gain -55.93% ub95 -38.91%, years 0/6, lag2 A/R 1.4282, A/P 1.5058); regime_up: refuted (A/R 1.1101, DM t 2.47, Sidak p 1.0, gain -11.01% ub95 -3.68%, years 2/6, lag2 A/R 1.1025, A/P 1.1069)
- 實驗 **E-011**：inconclusive（證據等級 `development`，登錄前已有 20 次使用重疊）：descriptive diagnostic of fixed-parameter drift; see report tables (no verdict)
- 實驗 **E-009**：aborted（證據等級 `development`，登錄前已有 10 次使用重疊）：aborted before any evaluation: the pre-run guard found GOLD# swap_mode 0 (DISABLED) where the design assumed swap in points; bars were fetched but no signal, trade or P&L was computed
- 實驗 **E-012**：inconclusive（證據等級 `development`，登錄前已有 12 次使用重疊）：underpowered: 2024+ 274 trades / 33 months with trades; Sharpe EWMA_RISK 1.291 vs FIXED_RISK 1.087 (delta 0.204, 95% month-block CI [-0.29, 0.617]); vol-matched MDD -5.401% vs -4.877%, CVaR5 -91.025 vs -93.608 bp; alpha on FIXED+market {'alpha_bp': 0.625, 'nw_t': 1.41, 'betas': [0.7317, -0.0121]}; delta Sharpe spread x2 0.202, quoted swap x1.5 0.207; descriptive EWMA_TARGET vs FIXED_RISK {'delta_sharpe': 0.032, 'ci': [-0.278, 0.344]}, vs CONSTANT_NOTIONAL {'delta_sharpe': 0.111, 'ci': [-0.412, 0.621]}; hour R^2 of weight ratios {'EWMA_RISK': 0.072, 'EWMA_TARGET': 0.065, 'CONSTANT_NOTIONAL': 0.081}
- 實驗 **E-013**：supported（證據等級 `development`，登錄前已有 22 次使用重疊）：supported: VR_day P0 0.4341 P1 0.932 (delta 0.4979, CI [0.405, 0.5979]), VR_week delta 0.4577 (CI [0.2796, 0.6539]); RV level P1/P0 0.4653, VR-adjusted 0.999; change months {'spread_zero_share': '2018-10', 'h1_bars_per_day': '2018-04', 'm1_zero_return_share': '2020-03'}; findings: M1 spread field: confirmed_defect; H1 spread field: confirmed_defect; D1 spread field: confirmed_defect; session length: confirmed_collection_change; RV level shift: intraday_reversal_excess_consistent_with_quote_noise; external cross-check: undetermined

## 5. 已登錄、尚未有結果的實驗

- ~~E-003~~ 已作廢（執行前發現設計缺陷，見決策紀錄）
- ~~E-004~~ 已作廢（執行前發現設計缺陷，見決策紀錄）
- **E-014** → H-009（exploratory）：`exp_spread_cost_v1`；審查：通過，可執行  
  判定規則：Sidak over trials_declared=4 confidence intervals that enter decisions (C1, C2, volatility ratio, increment) times the family count passed by the runner: CI level 0.95^(1/k), month-block bootstrap (2000 draws). Q2 (the H-009 volatility clause, its own threshold 1.2 on volatility = sqrt of realized variance): for each test month, using only the 90 days ending 40 minutes before the month, z = (spread - hour-of-week mean) / max(hour-of-week sd, 1 point) for cells with >= 100 bars; anomaly = z above that window's 90th percentile (strict). Normalized volatility = sqrt(next-30-minute RV) / that window's hour-of-week median sqrt(RV). vol_ratio = anomaly-weighted geometric mean over months of the within-month ratio of mean normalized volatility after anomalous vs other minutes. Supported as vol_clause_not_refuted (ledger outcome supported, meaning only that the volatility clause survived; the replay clause is untested) only if all hold: vol_ratio >= 1.2, CI lower bound > 1, vol_ratio excluding server hours 23/0/1 >= 1.2, at least 75% of test years >= 1.2, and the test anomaly share within [5%, 20%]; labelled vol_clause_not_refuted_but_redundant if the OLS coefficient of the anomaly flag on log next RV, controlling log past-30-minute RV, demeaned within hour-of-week x month, flat windows dropped, has a CI lower bound <= 0. refuted if the CI upper bound < 1.2. inconclusive_sample if < 2000 anomalous test minutes or < 24 months with both groups; otherwise inconclusive. Q1 (cost model): static_generalizes if the hour-of-week median C1 (train 2019-2022) has test MAE / train-global-median MAE <= 0.95 with CI upper < 1, below 1 excluding rollover hours and in every test year, and its hour-of-week mean has |mean cost bias| <= 10% in every test year; adaptive_only if only C2 (monthly, previous 90 days) passes the same rules against the same window's global median; else not_reliable. Q3: the bar SPREAD field is labelled min/max/first/last/mean of the minute's tick spreads if, on minutes whose tick spread changed, one candidate matches >= 90% and leads the next by >= 0.2; else undetermined. H-013 cost input ready only if Q1 is not not_reliable, Q3 is determined and >= 100 demo fills were priced against the last tick within 60 s before the fill; if the field is not the minute mean, H-013 must scale it by the reported tick-mean / bar-mean factor. Spread 0 is missing in every statistic, never zero cost. Robustness reported, not decisive: median-based ratio, mid-price RV, increment with a past 4-hour RV control, quote lag by fill reason. A not-redundant increment does not show information beyond a multi-day volatility state.
  - 審查 `mechanical` → **approve**：data reuse: evaluation windows overlap 19 earlier uses; evidence is development only
  - 審查 `critic_agent_6` → **revise**：B1 fixed train-window norms let period-level drift (spread and volatility rising together across 2023-2026) pass as anomalies (synthetic: Q4 spread +60%, vol +50%, unrelated per minute -> ratio 2.15, supported); robustness tables not in decide；B2 statistic was a variance ratio while H-009 states a volatility ratio; 1.2 on variance = 1.095 on volatility；B3 adaptive cost model C2 compared with the stale train median, so tracking the level alone could pass; rollover hours may carry C1；notes: field-semantics ties, unit factor for H-013, flat windows, min cell bars, quote lag, ledger wording, Sidak count, mid-price robustness；round 1 was on the draft before preregistration; scripts adv.py/null.py in the session scratchpad
  - 審查 `critic_agent_6` → **approve**：B1 resolved: norms trail per test month (90 days ending 40 min before), within-month ratios, hour-of-week x month demeaning, excl-rollover / 75% of years / anomaly-share band in decide; mid-month level shifts give inconclusive；B2 resolved: volatility = sqrt(RV), threshold 1.2, median ratio reported; B3 resolved: C2 vs same-window global median, excl-rollover MAE < 1；D-016 compliant: Q2 data from 2022-10, Q1 train 2019-2022, descriptive spread-only from 2018-10；note: a multi-day spread/vol state switching every 10 days passes as not refuted, which matches H-009's wording; not-redundant must not be read as information beyond a multi-day volatility state (descriptive 4h control added)；notes applied after approval: MID ffill instead of 0, prereg text aligned; round 3 confirmed increment_sums() reproduces the decisive coefficient exactly

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
- **E-010**（H-014）`refuted` / `development`：refuted: general: refuted (A/R 1.1248, DM t 3.09, Sidak p 1.0, gain -12.48% ub95 -5.83%, years 2/6, lag2 A/R 1.1183, A/P 1.116); extreme: refuted (A/R 1.5593, DM t 5.4, Sidak p 1.0, gain -55.93% ub95 -38.91%, years 0/6, lag2 A/R 1.4282, A/P 1.5058); regime_up: refuted (A/R 1.1101, DM t 2.47, Sidak p 1.0, gain -11.01% ub95 -3.68%, years 2/6, lag2 A/R 1.1025, A/P 1.1069)  
  下一個問題：H-018 with EWMA_094 stays next; a supported question names GVZ only for that question and needs forward data before any runner use  
  紀錄：`training_runs/20261112T130455Z_research_e010_exp_gvz_increment_v1`
- **E-011**（H-020）`inconclusive` / `development`：descriptive diagnostic of fixed-parameter drift; see report tables (no verdict)  
  下一個問題：pick the dominant measurable part and preregister a forward-data test for it  
  紀錄：`training_runs/20261112T135155Z_research_e011_exp_fixed_param_drift_v1`
- **E-009**（H-018）`aborted` / `development`：aborted before any evaluation: the pre-run guard found GOLD# swap_mode 0 (DISABLED) where the design assumed swap in points; bars were fetched but no signal, trade or P&L was computed  
  下一個問題：E-012: same design with the account's actual swap (0) and the quoted swap as stress  
  紀錄：`training_runs/20261112T135141Z_research_e009_exp_vol_sizing_v1`
- **E-012**（H-018）`inconclusive` / `development`：underpowered: 2024+ 274 trades / 33 months with trades; Sharpe EWMA_RISK 1.291 vs FIXED_RISK 1.087 (delta 0.204, 95% month-block CI [-0.29, 0.617]); vol-matched MDD -5.401% vs -4.877%, CVaR5 -91.025 vs -93.608 bp; alpha on FIXED+market {'alpha_bp': 0.625, 'nw_t': 1.41, 'betas': [0.7317, -0.0121]}; delta Sharpe spread x2 0.202, quoted swap x1.5 0.207; descriptive EWMA_TARGET vs FIXED_RISK {'delta_sharpe': 0.032, 'ci': [-0.278, 0.344]}, vs CONSTANT_NOTIONAL {'delta_sharpe': 0.111, 'ci': [-0.412, 0.621]}; hour R^2 of weight ratios {'EWMA_RISK': 0.072, 'EWMA_TARGET': 0.065, 'CONSTANT_NOTIONAL': 0.081}  
  下一個問題：if no sizing value: turn to questions with a different economic mechanism (H-008/H-013/H-009)  
  紀錄：`training_runs/20261112T142415Z_research_e012_exp_vol_sizing_v1`
- **E-013**（H-021）`supported` / `development`：supported: VR_day P0 0.4341 P1 0.932 (delta 0.4979, CI [0.405, 0.5979]), VR_week delta 0.4577 (CI [0.2796, 0.6539]); RV level P1/P0 0.4653, VR-adjusted 0.999; change months {'spread_zero_share': '2018-10', 'h1_bars_per_day': '2018-04', 'm1_zero_return_share': '2020-03'}; findings: M1 spread field: confirmed_defect; H1 spread field: confirmed_defect; D1 spread field: confirmed_defect; session length: confirmed_collection_change; RV level shift: intraday_reversal_excess_consistent_with_quote_noise; external cross-check: undetermined  
  下一個問題：excluding or re-weighting any period needs its own preregistration; past results stay as recorded and reruns on other periods are not confirmatory  
  紀錄：`training_runs/20261112T154049Z_research_e013_exp_data_quality_v1`

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
- **D-010** H-014 先只檢驗波動子句，問題是『資訊增量』：同一種擬合、同一批資料列下，加入 GVZ 的模型是否優於只用價格歷史（EWMA_094）的模型；不是單純追求比 EWMA 更低的誤差。分三個預先指定的問題：一般波動（h=5）、極端波動（未來 5 日超過過去 90 百分位）、狀態轉變（未來 22 日波動升到過去 22 日的 1.5 倍以上），Sidak 校正 3 次。溢酬預測報酬的子句延後。GVZ 時間規則：起點 t 只用美國日期 <= t-1 日的值（歷史發布時間無法驗證，採保守延遲），另以再晚一筆的 lag 2 作為必要條件。實驗編號 E-010；E-009 保留給 H-018。H-018 保留為下一個獨立方向：在固定交易訊號下，檢驗 EWMA_094 波動預測能否改善風險調整後報酬、最大回撤或尾部風險，核心不是預測準確度；E-010 結果不改變 H-018 的波動輸入（仍為 EWMA_094），除非某個問題 supported 且經前瞻資料確認。GVZ 與任何未驗證模型都不接入 Runner：把重新校準的效果和新資訊的效果分開；預先固定問題、期間、參數與狀態，避免事後挑選
- **D-011** E-010 依預先登記規則三題皆 refuted（general A/R 1.125 DM t 3.09；extreme 1.559 t 5.40；regime_up 1.110 t 2.47；lag 2 與對照 P 方向一致）。H-014 波動子句設為 refuted，範圍限於『GVZ 以 log 水準、lag 1、2014-2017 凍結係數加入EWMA_094 模型』；依審查意見，這表示凍結的 GVZ 模型樣本外較差，不代表 GVZ 不含資訊；報酬子句未檢驗。2024+ 的 general A/R 0.98、h22 0.92 只是描述，不得用來挽回結論。H-018 不變：下一個獨立研究方向，以 EWMA_094（無擬合參數）作波動輸入，在固定交易訊號下檢驗風險調整後報酬、最大回撤與尾部風險，編號 E-009。GVZ 的尺度不變形式（相對自身過去的變化）屬事後想法，若要研究須另立假設並只用前瞻資料判定；目前不登記：照預先登記執行；兩次失敗都指向凍結的水準校準，但這是看過結果後的歸納，不能在同一段資料上再驗證
- **D-012** 暫停尋找新的波動預測模型。E-009（H-018）：固定 H1 C 凍結訊號與出場（TP 3／SL 1.5 ATR／48h，一次一筆），只改變進場時固定的部位大小。主要比較為巢狀設計：EWMA_RISK = 固定風險倉位 × (σ_ref/σ_EWMA) vs 固定風險（固定風險本身已以 H1 ATR 調整；審查發現直接以 EWMA 取代 ATR 主要只是按進場小時重新加權，所以改成保留同一個 ATR 停損、只加入日線 EWMA 縮放）。純目標波動（EWMA_TARGET）與固定名目（CONSTANT_NOTIONAL，runner 固定手數的對應）只作描述，各自附信賴區間。不測不交易濾網，因為它會改變交易樣本。判定只用 2024+（H1 C 在 2014-2022 擬合、2023 選門檻）；2018-2023 為樣本內，只作描述。H-018 原 falsified_if 的點估計寫法由 E-009 預先登記的檢定式規則取代。E-011（H-020）為獨立的描述性診斷，不判定任何假設，不混入 E-009。若 E-009 顯示部位管理沒有額外價值，轉向不同經濟機制的問題（H-008／H-013／H-009）：訊號固定才能把部位管理的價值和方向預測分開；同一套交易、同一成本、同一槓桿上限，並以事前預算對齊與事後波動對齊兩種尺度報告
- **D-013** E-009 在任何計算前被 swap 單位檢查中止（GOLD# swap_mode 為 DISABLED），記為 aborted；Runner 修正為也能接住模組拋出的 SystemExit。E-012 沿用 E-009 已審查的設計，只改 swap：基準情境用帳戶實際 swap（0），報價 swap_long×1.5 改為壓力情境。O-016 推翻了 E-001／E-006／E-007 的 swap 成本前提：H-008（E-006 的『盤中優於持有』主要來自 swap）需以帳戶實際成本重新檢驗，狀態維持 inconclusive；H-017（換日前平倉以省 swap）的前提不成立，設為 parked。E-011 的判讀記為 O-015，可檢驗的後續方向登記為 H-021（資料品質，可用一致性比較檢驗）與 H-022（只用前瞻資料）：成本前提必須以真實成交紀錄驗證；中止發生在看到任何結果之前，所以沿用同一設計不構成事後調整。揭露：E-009 與 E-012 之間已看過 E-011 的描述結果（O-015，波動水準，沒有部位管理的損益）；trials_in_family 會把這次中止算成 H-018 家族的一次嘗試
- **D-014** H-018 設為 underpowered：E-012 既不能排除 0.2 Sharpe 量級的改善，也不能排除同量級的惡化；風險面（同波動回撤、集中度）沒有改善跡象。以 H1 C 每年約 100 筆的頻率，要把 CI 半寬從約 0.45 縮到 0.2 需要約 5 倍資料（十年以上），所以 H-018 以 H1 C 繼續累積的資訊增益很低，調降優先序；不更動任何 Runner。依使用者指示，部位管理沒有可證實的額外價值，研究轉向不同經濟機制：優先順序依 Priority Engine，其中 H-008 必須先以帳戶實際 swap（O-016）重新設計。：照預先登記執行；underpowered 不是推翻，但在可預見的資料量下也無法判定，繼續投入的價值低
- **D-015** 使用者決定的順序：H-021（歷史資料可信度）→ H-009（點差異常與可執行性）→ H-013（交易經濟優勢判斷，主要方向），並改進 Priority Engine（前提有效性、資料可用性、假說依賴、已有反證與範圍、研究多樣性、資料品質對下游的影響；失效假說自動轉入適當狀態並保留原因與證據）。E-013 是資料診斷：不訓練模型、不評估策略、不得用來重新宣稱任何過去策略具有獨立樣本外優勢。外部獨立黃金資料（Stooq／Dukascopy／Yahoo）在 2026-11-12 從本機無法取得，改以 GOLD# 的 M1／H1／D1 互相核對與 GOLD#／EURUSD／XAUEUR# 交叉匯率恆等式核對；能只靠外部資料判定的部分歸為無法判定：後續所有研究都依賴歷史資料；先確認哪些缺陷已證實、哪些是真實市場差異、哪些無法判定
- **D-016** H-021 設為 supported（範圍：2018 年前 H1 層級的價格統計受換日時段型態嚴重放大）。資料使用規則：任何以 H1（或更細）價格統計擬合、校準或估計成本的研究，預設只用 2019-01-01 之後的資料；2018 為過渡年；2019 之前的期間只能在預先登記中說明理由並把它當成獨立的資料品質條件。點差研究只能用 2018-10 之後（之前欄位缺失）。受影響的既有結論（保留原紀錄，不改寫，只註記）：E-008／E-010 的凍結模型以 2014-2017 擬合，主要學到被放大的波動水準；O-013 至 O-015 中『固定參數退化／目標分布改變／校準漂移』大部分應歸因於資料而非市場；H-019／H-022 的前提需重新審視；H1 C 的訓練期 2014-2022 包含受影響期間（其 2024+ 評估不因此失效，但模型可能學到換日瑕疵），Demo Runner 不更動，是否重新訓練由使用者決定並另行預先登記。不得以 E-013 重新宣稱任何過去策略的樣本外優勢：後續研究的資料基礎必須先可靠；把資料缺陷、收集方式改變與未判定部分分開記錄
- **D-017** Priority Engine 依賴圖（第一版）與上游標記。資料：2014-2017 H1 與 2018 H1（過渡年）標為 defective，2018-10 前的 SPREAD 欄位標為 defective，2019+ H1 與 2025+ M1 為乾淨葉節點；前提：本 demo 帳戶隔夜收 swap 標為 invalid。依賴：E-008／E-010 以 2014-2017 擬合；E-011 重建兩者的凍結參數；E-012 用 H1 C 訊號並以 2018-2023 校準；E-001 用 H1 C；model:h1_c 以 2014-2022 擬合；model:gemini 以 2025+ M1；O-013／O-014／O-015／O-017 分別以 E-008、E-008+E-010、E-011、E-012 為證據；H-019、H-020、H-022 以這些觀察為前提；H-018 以 E-012／O-017 為證據；H-008、H-017 以隔夜收 swap 為前提；H-011、H-014 的結論以 E-008、E-010 為證據。只宣告依賴，不改變任何實驗結果或已結案假說的狀態：E-013 證明上游資料出問題時，下游假說仍可能被當成有效證據；讓引擎沿依賴路徑追蹤受影響的證據並據以重排
- **D-018** 核准（使用者 2026-11-16 指示）：H-008 設為 premise_invalid（隔夜收 swap 的前提在本帳戶不成立，停止依原假設推進）；H-019、H-022 設為 needs_revalidation（前提來自受 2014-2017 資料影響的觀察；不等於被推翻）。model:h1_c 經 D-017 標為訓練資料可能受污染，2024+ 評估保留，Demo Runner 不變，是否重訓另行預先登記。引擎另建議 H-017 premise_invalid、H-018 與 H-020 needs_revalidation，屬系統建議，未在本決策核准：區分已核准的研究決策與系統自動建議；needs_revalidation 只代表證據基礎需要以 2019+ 資料重新驗證
- **D-019** 核准（使用者 2026-11-16）：H-017 設為 premise_invalid，範圍僅限已確認的本 demo 帳戶條件（swap_mode=DISABLED，O-016），不擴及其他 XM 帳戶；H-018、H-020 設為 needs_revalidation（依賴受影響資料，不是被推翻）。H-009 設為 testing（E-014）。E-001～E-007 依賴稽核：E-001、E-005、E-006 的開發期 2018-2022/2023 以 H1 計算，含過渡年 2018，宣告依賴 data:gold_h1_2018（E-001 另依賴 model:h1_c）；E-002、E-007 只用 D1，不受影響，宣告依賴 data:gold_d1_all；E-003、E-004 已作廢不宣告。H-001、H-004 的結論分別以 E-001、E-005 為證據。歷史結論與觀察一律不改寫：使用者核准系統建議；補齊依賴圖讓受 2018 過渡年影響的證據被標註

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
| 2018-01-01 → 2024-01-01 | E-010 |
| 2024-01-01 → 2099-01-01 | E-010 |
| 2024-01-01 → 2099-01-01 | E-009 |
| 2018-01-01 → 2099-01-01 | E-011 |
| 2024-01-01 → 2099-01-01 | E-012 |
| 2014-01-01 → 2099-01-01 | E-013 |
| 2023-01-01 → 2099-01-01 | E-014 |

## 9. 研究空間覆蓋與自動產生的候選問題

探索次數（歷史研究＋帳本中已標註的假說）。0 代表從未研究過。

| 資訊來源 | 取得狀態 | 漲跌方向 | 未來波動 | 報酬分布／尾部 | 值不值得交易 | 成本與執行 | 市況轉換 |
|---|---|---|---|---|---|---|---|
| 價格 K 線 | in_hand | 29 | 7 | 1 | 2 | 3 | 3 |
| Tick／點差動態 | in_hand (MT5 ticks, months of history) | 2 | 1 | 0 | 1 | 2 | 0 |
| 經濟事件實際值／預期值 | export_needed (ExportUsdCalendarValues) | 3 | 1 | 0 | 1 | 0 | 0 |
| 利率／總經 | in_hand (to 2024-12) | 3 | 0 | 0 | 0 | 0 | 0 |
| COT 部位 | in_hand (to 2024-12) | 2 | 0 | 0 | 0 | 0 | 0 |
| 跨市場（美元、白銀、指數） | in_hand | 2 | 0 | 0 | 0 | 0 | 0 |
| 隱含波動率（CBOE GVZ） | new_source (FRED GVZCLS, free) | 1 | 2 | 0 | 0 | 0 | 0 |
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
