# H-013 新版預先登記草案（v2.5，草案，尚未登記、尚未核准）

v2.5 修訂（2026-11-18）依 `修改意見.md` 第五輪：
- 11.0：E-017 的 AUC 判定期對齊 E-016 的成本驗證期 `[2025-01-01, 凍結日)`，附窗口用途表；
- 11.2：分開定義成本估計報價、覆蓋率目標、決策時的特徵與閘門報價，附人工時間線；
- 11.4：共同分鐘不再要求有效閘門報價（那本身就是篩選）；
- 11.6：k = 4 改寫為設計假設；說明對照組模型的角色；
- 11.7：E-016／E-017 候選 preregistration JSON 與離線 `critique()` 結果。

v2.4 修訂（2026-11-18）依 `修改意見.md` 第四輪：
- 特徵與標籤的有效性規則（排除 ±inf；CLOSE 無效時相關窗口一律 NaN）；
- 第 9 節改成決策表；
- 新增第 11 節方案 1 草稿（橋接驗證 E-016 與子題 E-017，候選編號）；
- 新增第 12 節兩階段審查流程。

v2.3 修訂（2026-11-17）依 `修改意見.md` 第三輪：
- 閘門輸入邊界收斂；
- AUC 主要比較改用共同資料列；
- 驗證範圍說明（第 10 節）。

v2.2 修訂（2026-11-17）依 `修改意見.md` 第二輪：
- 閘門改用決策當下的報價；
- 標籤缺值不再被當成 0；
- 前推切分限制在指定區間內；
- 補上方案 1 的橋接驗證與判定範圍。

狀態：使用者只核准**研究設計與可行性檢查**（D-021 `set_approval: design_only`）。這份文件不是預先登記事件，
也不是帳本決策；第 9 節的決策草稿尚未寫入帳本。正式驗證要等第 8 節的前置條件全部成立，研究執行器
（`research_run_v1.py`）在那之前會拒絕執行。

v2.1 修訂（2026-11-17）依 `CLAUDE_RESEARCH_MODIFICATION_BRIEF_2026_11_17.md`：
- P0-1：模型樣本身分改成逐段表（第 4 節）。
- P0-2：補上成本契約與公式（第 3 節）。
- P0-3：明定時間語義，提出兩個方案（第 2、6 節）。
- P1：補上可行性表與檢定力（第 5 節），並新增離線模組 `exp_h013_tradeability_v1.py`。

## 1. 原始假說（帳本 H-013，不改寫）

- 主張：可以預測未來 1 小時的波動是否大於 k 倍點差（不預測方向），而 gemini 的虧損集中在預測為「不值得交易」的時段。
- 機制：波動太小時，點差佔停利的比例過高，任何方向判斷都難以覆蓋成本。
- 推翻條件（**不變**）：樣本外波動大小分類 AUC ≤ 0.55，或低可交易性組的 gemini 回放 R 沒有比其他組低 0.05R 以上。
- 預期收益機制：不是找到新的方向優勢，而是**在成本占比過高的時段不交易**，讓既有策略少付不划算的成本。
  它的價值上限是「避開的淨虧損」，前提是方向策略本身在其他時段不虧。

## 2. 兩個問題與時間語義

| | 問題 A：事後診斷 | 問題 B：決策時點預測 |
|---|---|---|
| 問什麼 | 未來 h 分鐘**實際**的波動是否足以覆蓋成本 | 在決策當下，只用**當時已知**的資料，能否預測未來 h 分鐘值得交易 |
| 欄位 | `y_ex_post` | `p_pred`（逐月前推、只用過去擬合） |
| 標籤時間 | Y(t) = 1 若 bid 價在 (t, t+h] 的高低區間 > k × 成本(t)；窗口內少於 h 根 M1（休市、週末、資料尾端）時沒有標籤 | 同一個 Y(t)，只作為預測目標 |
| 特徵時間 | 不適用（用到未來） | 嚴格早於 t 的 bid 收盤實現波動（30／240／1440 根）、伺服器小時、**排定**出場時刻 t+h 的換日旗標、t 之前最後一筆報價、當日 C95（只用 t 當日之前的 tick 日） |
| 判定 | 只有描述，不做推翻判定 | 原始推翻條件（AUC ≤ 0.55） |

原假說 h = 60 分鐘。若問題 B 的特徵含有任何標籤或未來欄位（`y_*`、`exit_at`、`spike_exit`），
`question_b()` 會直接拒絕（測試 `test_purged_folds_and_prediction_on_synthetic_volatility`）。

**AUC 的主要比較**（v2.3）：所有模型都在同一組資料列上計分。這組資料列要有有效標籤，而且每個模型都有有限的預測值。
各模型用自己全部可用資料列算出的 AUC 只列為輔助結果（`auxiliary_all_available`），不能拿來比較模型。
每個 fold 報告總列數、有效標籤數、共同評估數與排除原因（沒有標籤、模型沒有擬合、特徵缺失），不能只看合併後的 AUC。

E-015 的 E95x 依「實際出場報價時刻」判斷是否落在換日時段。跨過每日休市時，實際出場時刻是重開時間，事後才知道。
預測流程因此改用排定出場時刻 t+h 的換日旗標（`roll_exit_scheduled`）。兩者的一致率要在正式執行時報告；
在那之前，E95x 在預測流程裡的覆蓋率不能直接沿用 E-015 的數字。

**時間尺度的限制**：E-015 驗證的是固定 +60 分鐘出場的報價點差。它不證明 gemini 中位數 4.9 分鐘的持倉、
TP／SL 提前出場、或最長 90 分鐘出場的成本也被覆蓋。

## 3. 成本契約（P0-2）

單位：點（0.01 美元／盎司，即 gemini.py 的 (ask − bid) / point）。**一筆來回只付一次點差。**
程式在 `exp_h013_tradeability_v1.net_points()`，三種價格基準用同一組報價時損益相同
（測試 `test_cost_contract_is_one_spread_and_matches_by_hand`）。

| 價格基準 | 多單 | 空單 | 另外扣 |
|---|---|---|---|
| 實際成交價（fill） | 成交價差，點差已在價格裡，**不再扣** | 同左 | 佣金、手續費、隔夜利息（此 demo 帳戶皆 0；真實帳戶未確認） |
| bid 價格路徑 | 毛利 − **進場**點差（ask 進、bid 出） | 毛利 − **出場**點差（bid 進、ask 出） | 同上 |
| mid 價格路徑 | 毛利 − ½ 進場點差 − ½ 出場點差 | 同左 | 同上 |

手算範例（測試裡的人工報價）：
- 報價：進場 bid 2000.00／ask 2000.30（30 點），出場 bid 2001.00／ask 2001.40（40 點）。
- 多單：成交 2000.30 → 2001.00 = **+70 點**；bid 路徑 100 − 30 = 70；mid 路徑 105 − 35 = 70。
- 空單：成交 2000.00 → 2001.40 = **−140 點**；bid 路徑 −100 − 40 = −140；mid 路徑 −105 − 35 = −140。

在研究中套用成本上界時：
- **gemini 回放（只做多，bid 路徑）**：成本 = 進場 C95。出場上界只對空單有意義，不加。
- **問題 A／B 的標籤（不分方向）**：成本(t) = max(C95(t), E95x(t) × 2)，取多單與空單中較貴的一邊，
  **不是** C95 + E95x。
- **上界缺失時**：任一個上界缺失就是 NaN，標記為不可評估。不填 0，也不改用 1.21 常數。
- **壓力情境**：×1.5、×2 是情境，不是經驗證的成本。D-022 的「出場 × 2」也是未驗證的壓力值。
- **即時報價閘門**（v2.2 修正）：閘門用**決策當下已收到的最後一筆有效報價**（`quote_at`）。
  - 決策時刻 = bar 開盤 + 預先登記的決策延遲；只要當分鐘的第一筆報價在決策前到達，就會被納入。
  - 晚於決策時刻的 tick 不能影響決策；報價時間晚於決策時間會直接報錯。
  - `quote_before`（上一個有報價分鐘的最後一筆）只作為預測特徵，**不能**當閘門報價。
    例：上一分鐘 30 點、當分鐘第一筆 200 點、C95 35 點時，舊版放行，新版阻擋。
  - 記錄報價時間與決策時間。
  - 不可交易原因（依序判斷）：
    - 沒有報價（`no_quote`，包括完全沒有有效 tick）；
    - 點差不是有限正值（`invalid_quote`）；
    - 報價過期（`stale_quote`，時效門檻在預先登記時固定，草案建議 60 秒）；
    - C95 不是有限正值（`no_bound`）；
    - 高於 C95（`cost_gate`）。
  - 有效 tick 必須時間、bid、ask 都是有限值，且 bid > 0、ask > bid。
  - 決策時間不是有限值、時效設定不是有限非負值時直接報錯，不會靜默放行。
  - 因閘門而沒有交易和因模型濾網而沒有交易（`model_filter`）分開記錄（`skip_reason`）。
  - 決策延遲與報價時效兩個參數要在登記前固定，不能看結果後調整。

D-022 寫了「進場 C95、出場 E95x × 2、一筆來回只計一次點差」，但沒有給合計公式。上面兩條公式是依同一原則寫成的，
沒有改變 D-022 的實質內容，但仍需使用者確認後才能寫進帳本（第 9 節草稿 1）。

## 4. 模型與資料的樣本身分（P0-1）

來源：`gold_long_recent_walk_forward.py`（第 152–212 行）、`gold_long_recent_walk_forward.json`、`gemini.py`、
`gemini_trade_history.csv`、git 歷史、`training_runs/20260929T160233Z_gold_s4_source_binding_validator_repair_v1/source_binding_records_v1.json`。

| 區間 | 身分 | 依據 |
|---|---|---|
| 2025-01-01 至 2026-03-31 | **模型樣本內**（訓練） | 存檔的模型是 `test_model`，用 `TIME_DT < 2026-06-01` 擬合，只扣掉最後 240 根（HORIZON） |
| 2026-04-01 至 2026-05-31 | **門檻選擇期，且也在存檔模型的訓練資料內** | 參數（門檻 0.75、TP 1.3／SL 1.6 ATR、時段）用 validation 模型選出，之後**重新擬合**時把這兩個月也放進訓練 |
| 2026-06-01 至 2026-08-25 | 模型層級樣本外，**但已被看過** | 報告寫為 untouched test。結果是 15 筆、PF 1.34，promotion gate FAIL，之後模型仍被採用 |
| 2026-08-25 之後 | 模型層級樣本外 | 但 gemini 的門檻在 2026-10-26（→ 0.70，commit 003c2ff「修改參數」，沒有記錄依據）和 2026-11-11（→ 0.65，風險 2%，commit c78f0c1，和 2026-11 策略研究一起提交）兩次調整。調整時是否參考了這段期間的表現：**Unknown**。保守起見，視為設定選擇用資料 |
| 2026-11-11 之後 | 目前設定的最新段 | 設定凍結在 commit c78f0c1；沒有預先登記、沒有凍結紀錄，**不符合**「真正前瞻資料」的條件 |

模型檔綁定：
- `gold_long_recent_candidate_xgb.json` 的 SHA-256 = `2dc32e3b…a755f2`，和 2026-09-29 source-binding 紀錄一致。
- 檔案修改時間 2026-08-25 14:49:39（+08）和報告 `generated_at` 06:49:39Z 同一秒，所以推定由這次執行產生。
- 但報告本身沒有記錄雜湊，這是**時間戳推定，不是雜湊綁定**。
- meta 模型 `gold_meta_regime_xgb.json` 在 gemini.py 中關閉（`USE_META_OVERLAY = False`），不影響交易。

**demo 交易的模型身分**（進場筆數，伺服器時間）：

| 期間 | 筆數 | 下單時的模型 |
|---|---|---|
| 2026-05-15 至 05-31 | 28 | **Unknown**：比現存模型檔早，git 歷史從 2026-09-01 開始 |
| 2026-06-01 至 08-24 | 64 | **Unknown**（同上） |
| 2026-08-25 至 11-10 | 51 | 推定為現存模型；門檻 0.75／0.70 |
| 2026-11-11 之後 | 17 | 現存模型，門檻 0.65 |

結論：
- 約 92 筆 demo 交易**不能**歸屬到目前的模型，最多約 68 筆可以。
- 任何「gemini 在低可交易時段虧損較多」的檢驗，都必須按模型與設定版本分層，不能把 160 筆混在一起。
- 歷史回放：2025-01 至 2026-05 是樣本內；2026-06 之後是已看過的開發證據。兩者都不能主張交易能力。

## 5. 資料可行性與檢定力（P1-1）

| 資料 | 可用範圍 | 缺失／限制 |
|---|---|---|
| M1／H1 價格 | 2019-01-01 起（D-016） | 價格可用不代表有成本上界 |
| tick 報價 | 2023-08-04 起 | E-015 每 2 個交易日抽 1 天。H-013 每個評估日都需要 t 之前的報價，所以要抓**每一天**的 tick（約為 E-015 的 2 倍） |
| C95 | 第一個有 ≥ 20 個 tick 日在前 90 天內的日子起：每日抓取約 2023-09 起；E-015 的抽樣方式約 2023-10 起 | 更早的日子 = 不可評估，不補值 |
| `gemini_trade_history.csv` | 2026-05-15 至 2026-11-17，160 筆進場，全為多單 | 模型身分見第 4 節 |
| `gemini_signal_log.csv` | 2026-05-18 起，99,899 列 | `spread_points`、`tp_distance`、`sl_distance` 只在 198 列下單相關事件中有值（缺失 99.8%）。**更正 v2 的說法**：它不能在每個決策時點重建成本占比，只能描述有下單動作的時點 |

檢定力（`exp_h013_tradeability_v1.power` / `n_required`）：
- 設定：效果量 0.05R（原條款，不變），單尾 5%，R 標準差 σ，低可交易組占比 p，交易相依的設計效果 deff。
- 單一組差的標準誤 = σ·√(deff / (n·p·(1−p)))。

| σ | p | deff | 80% 檢定力所需筆數 | 157 筆的檢定力 |
|---|---|---|---|---|
| 0.9 | 0.25 | 1.0 | 10,684 | < 15% |
| 0.9 | 0.25 | 1.5 | 約 16,000 | 更低 |

v2 寫的「約 7,000 筆」是「差距 = 2 個標準誤」的估計，不是 80% 檢定力，也不是通用門檻。
正式登記時要把 σ（只用選擇期以前的資料估計）、p、deff、檢定方式和效果量一起固定。
**筆數不足時，結論只能是 underpowered，不能放寬 0.05R。**

## 6. 兩個可選方案（P0-3，由使用者選擇，不能事後依結果挑選）

| | 方案 1：保留原假說 | 方案 2：與實際持倉相符的新假說 |
|---|---|---|
| 時間尺度 | h = 60 分鐘，AUC 0.55、0.05R 不變 | 新假說（例如 h = 15 分鐘，或依實際出場時間），需要新編號與新預先登記 |
| 問題 B（AUC） | 可以正式檢驗（判定期 2025-01 起，和橋接驗證期相同；2023-10 起的資料只做訓練，見 11.0） | 同左，但成本上界要另做出場時間相依的驗證，**不能沿用 E-015 的 supported** |
| 第二條款（gemini 0.05R） | 樣本不足，預先宣告為 underpowered；只做描述，並按模型版本分層 | 同樣受交易樣本限制；可以改用預先登記的回放，但回放期全在樣本內或已看過 |
| 成本輸入 | 以 E-015 C95／E95x 為基礎，但定義有改動，需要橋接驗證（見下表） | 需要新的成本驗證（新實驗） |
| 可判定的範圍 | 只有「可交易性能否預測」這個子題 | 子題與新假說的經濟條款 |
| 優點 | 成本工作只需要橋接驗證，不必新做完整的成本研究 | 和 gemini 的行為一致，經濟意義較直接 |
| 缺點 | 和 gemini 5 分鐘的持倉不一致，經濟條款判定不了 | 多一個成本實驗，而且仍受交易樣本限制 |

**方案 1 和 E-015 的定義差異**（v2.2 補充）：

| 項目 | E-015 | 方案 1 | 需要 |
|---|---|---|---|
| 進場目標（分鐘第一筆報價）、C95 公式、90 天窗口、最少 20 個 tick 日、每格最少 200 分鐘 | 同 | 同 | 不需要；但只證明覆蓋第一筆報價，不證明覆蓋閘門報價或成交（11.2） |
| tick 抽樣 | 每 2 個交易日 1 天 | 每天 | **橋接**：每日抽樣下 C95 的前向覆蓋率要重新計算，並且再次通過 E-015 的門檻 |
| E95x 換日分格 | 實際出場報價時刻 | 排定出場時刻 t+60 | **橋接**：用排定時刻重新計算 E95x 的覆蓋率，並且再次通過門檻。兩種時刻的一致率只是診斷，不能代替這個檢查 |
| 閘門報價 | 不適用 | `quote_at`（決策當下的報價） | 報告閘門排除的比例，以及排除後的覆蓋率 |
| 標籤成本 | 不適用 | max(C95, E95x × 2) | 屬於提案，待使用者確認（第 9 節） |

橋接驗證本身也要預先登記並經 Critic 審查；任何一項沒通過，方案 1 就不能使用該成本輸入。

**判定範圍**：方案 1 只檢驗可交易性的預測子題（問題 B，AUC ≤ 0.55 即推翻）。
- 第二條款預先宣告為 underpowered，樣本不足時 H-013 **整體不能判為 supported**。
- 即使 AUC 通過，最多只能記為「子題通過，整體因經濟條款樣本不足而 inconclusive」。
- AUC ≤ 0.55 時子題被推翻；依原推翻條件的「或」，H-013 整體為 refuted。
- 這套判定規則要寫進新版預先登記。

建議方案 1：先判定「可交易性能否預測」，這一題不需要 gemini 交易紀錄。只有子題通過，才值得投資方案 2 的成本驗證。
新成本公式與方案 1 都還是**提案**，不代表已同意寫入帳本或提升為 formal。

## 7. 交易比較的可執行語義（P1-3，正式執行時才做）

需要預先定義三個版本，在相同資料與成本假設下比較：
1. 原始 gemini；
2. 只加成本閘門；
3. 成本閘門加預測濾網。

- **先報告固定交易集合的分組診斷，再報告重新模擬執行的結果。** 去掉一筆交易會釋放持倉、改變後面的進場，
  所以不能只刪掉虧損單就說得到一個可執行策略。
- **報告項目**：逐期與合計交易數、trades/day、頻率變化、勝率、PF、Mean-R、PnL、最大回撤、區塊信賴區間、×1.5／×2 壓力。
- **分組規則**：分組只用事前固定的規則和樣本外 `p_pred`，不按未來波動或已實現 R 分組。
- **尚未實作**：重新模擬需要 gemini 的離線回放器，`exp_h013_tradeability_v1.py` 目前沒有 `run()`。gemini.py 不修改。
- **順序**：見第 12 節的兩階段流程。不能先核准正式實驗，之後才補執行內容。

## 8. 正式驗證前置條件（帳本 D-021 `set_prerequisites`，由 Priority Engine 自動檢查）

| 前置條件 | 目前狀態 | 解除需要的最小工作 |
|---|---|---|
| 使用者核准層級 formal | 未成立（design_only） | 使用者以決策 `set_approval` 核准 formal |
| `cost_upper_bound`：E-015 判定 supported | **已成立**（H-023 狀態仍待使用者核准） | — |
| `prereg_v2_reviewed`：新版預先登記通過機械與獨立 Critic 審查 | 未成立 | 第 12 節：離線審查候選程式 → 使用者選方案 → 凍結並以雜湊登記 → 依帳本流程登錄兩份 review |
| `trade_sample`：交易樣本足以檢驗第二條款 | 未成立（現行模型約 68 筆可歸屬） | 方案 1：預先宣告 underpowered（需要使用者核准把這條改為「描述」）；或累積樣本 |

帳本已把「成本輸入未通過可靠性門檻」（E-014）記為 H-013 開放中的證據品質條件。
這個條件只標示、不重複扣分；要等使用者核准 E-015 的結論後，再以附證據的決策關閉。

## 9. 待使用者決定（**未寫入帳本**）

| 項目 | 建議 | 依據 | 核准後會改什麼 | 仍不授權什麼 |
|---|---|---|---|---|
| 成本公式解釋 | 確認為 D-022 的實作：gemini 多單回放 = 進場 C95；不分方向的標籤 = max(C95, E95x × 2)；上界缺失 = 不可評估 | 第 3 節成本契約；三種價格基準手算一致（測試） | 新增一筆決策，寫明公式；第 11 節橋接驗證與 H-013 程式照此實作 | 不代表 E95x 的排定出場版本已通過（要等橋接驗證）；不改 E-015 判定 |
| 模型身分觀察 | 記錄 | 第 4 節：存檔模型以 2026-06-01 前資料重新擬合；約 92 筆 demo 交易的模型為 Unknown | 新增一筆觀察紀錄（附來源）；第二條款的樣本按模型版本分層 | 不改任何既有實驗結果；不涉及研究結論 |
| 方案選擇 | 方案 1，具體內容為第 11 節 v2.5：E-016 與 E-017 的判定期都是 `[2025-01-01, 凍結日)`，k = 4、logistic、purge 120 分鐘、閘門參數 1 秒／60 秒 | 第 6 節：子題不需要 gemini 交易；成本只需橋接驗證；第二條款兩案都受樣本限制。11.7 的離線 critique 沒有設計阻擋 | 第 11 節成為下一步：寫 E-016／E-017 正式程式，凍結後登記，送機械與獨立審查；`trade_sample` 改為「描述」 | 不授權執行橋接研究或 H-013；不提升 formal；第二條款仍預先宣告 underpowered |
| H-023 狀態 | supported | E-015 依預先登記判定 supported（進場 C95、出場 E95x；尖峰未覆蓋） | 以決策設定 H-023 = supported；H-013 的 `cost_input_unreliable` 條件可附證據關閉 | 不代表橋接驗證已通過；不代表每筆交易都被覆蓋（`spikes_covered = false`） |

## 10. 驗證範圍（v2.3）

- 本草案的程式只用**合成資料**做過單元測試（`test_research_os_v1.H013BuildingBlockTests`）。沒有用真實 tick 或價格算過任何覆蓋率、閘門排除比例或 AUC。
- 指定環境 `.venv` 是 Python 3.11.9／pandas 3.0.5。第二輪修正了時間單位相容性：在 `datetime64[ns]` 輸入下，舊公式 `astype("int64") // 10**9` 和新函式 `epoch_seconds()` 算出相同的秒數。這一點只由公式推導與合成測試（s／ms／us／ns 四種精度）確認。
- E-014／E-015 封存時是 Python 3.10／pandas 2.3。兩次實驗**沒有重跑**，真實資料結果沒有逐值核對；封存的程式、報告與環境紀錄保留原樣。
- 本草案引用的 E-015 數字一律來自封存報告，不是新版程式的計算結果。
- v2.5：11.2 的時間線用合成 tick 跑過 `minute_table`、`quote_at`、`cost_gate`；11.7 的 critique 只讀帳本（呼叫前後列數相同）。都沒有用真實資料。

## 11. 方案 1 草稿：橋接驗證與 H-013 子題（v2.5，**未選定、未登記、不授權執行**）

> **2026-11-19 定位調整**：研究重點改為可執行的淨獲利（`STRATEGY_RESEARCH_2026_11.md` 第 8 節）。E-016 對首選假說 H-003 不是前置條件，E-017 列為輔助路線；本節草稿保留不刪，暫停推進。

候選編號（查過帳本：目前最大是 E-015、H-023、D-022、O-021；只是候選，不預先佔用）：
- **E-016**：H-023 的橋接驗證；
- **E-017**：H-013 子題（問題 B）。

### 11.0 窗口與用途（v2.5：E-017 主要評估期對齊 E-016 已驗證範圍）

時間基準：全部是 **MT5 伺服器時間**，區間一律半開 `[起, 迄)`。「凍結日」= 登記當天往前的最後一個完整交易日的隔天 00:00，
登記時寫成固定日期；下表與 11.7 的 JSON 暫用 `2099-01-01` 代表。

**建議（v2.5 選定的一套）**：E-017 的 AUC 判定只用 `[2025-01-01, 凍結日)`，和 E-016 的成本驗證期完全相同。
2024 年及更早的資料只做訓練或描述，不參與任何通過判定。不擴大橋接範圍到 2024，原因：
2024 年在 E-015 是校準期，沒有任何前向覆蓋資格；要讓它也有資格，必須另外驗證，而且 2023-08 至 2023-10 的
90 天歷史不完整，校準本身就不足。

| 窗口 `[起, 迄)` | 資料 | 成本估計歷史 | 模型訓練 | 成本驗證（判定） | AUC 評估（判定） | 描述 | 成本資格 |
|---|---|---|---|---|---|---|---|
| `[2019-01-01, 凍結日)` | M1 bar | — | — | — | — | E-016：每月換日尖峰時段（DST 檢查） | 不適用 |
| `[2023-08-07, 凍結日)` | 每日 tick | **是**：第 d 天的 C95／E95x 只用 `[d−90 天, d)` 已完成的 tick 日與出場 | — | — | — | — | 不適用（輸入） |
| `[2023-08-07, 2025-01-01)` | 每日 tick | 同上 | — | — | — | E-016：校準期覆蓋率，只報告 | 無（沒有前向覆蓋資格） |
| `[2023-09-01, 凍結日)` | M1 bar | — | — | — | — | — | E-017 特徵（rv_1440 需要前 1,440 根）與標籤的高低區間 |
| `[2023-10-01, 2025-01-01)` | M1 bar＋每日 tick | 同上 | **是**：E-017 第一個 fold 的訓練資料（逐 fold 擴張） | — | — | — | 無：訓練標籤的成本未經前向驗證，只影響擬合品質，不影響判定 |
| `[2025-01-01, 凍結日)` | 每日 tick | 同上 | E-017 後續 fold 的訓練（只用該 fold 之前、扣除 purge） | **E-016**：C95（每日）與 E95x（排定版）未經閘門篩選的覆蓋率 | **E-017**：逐月前推 AUC | 兩者的逐年、逐月 | **有**：只在 E-016 兩項都通過時成立 |

- E-017 的每一段主要評估 `[2025-01-01, 凍結日)` 都落在 E-016 的成本驗證期內，所以「評估列的成本有資格」⇔ E-016 通過。
- 所有區間都在 E-014／E-015 看過（2025 年後是 E-015 的前向期），**一律是開發證據**，不因日期對齊變成 untouched。

### 11.1 問題與依賴

| 改動 | E-015 | E-016 | 角色 |
|---|---|---|---|
| tick 抽樣 | 每 2 個交易日 1 天 | 每天 | **先決條件**：C95（每日）要通過全部 E-015 門檻 |
| 出場換日分格 | 實際出場報價時刻（`roll_exit`） | 排定出場時刻 t+60（`roll_exit_scheduled`） | **先決條件**：E95x（排定版）要通過全部 E-015 門檻 |
| 決策延遲與報價閘門 | 沒有 | `quote_at` + `cost_gate` | **診斷**：只報告閘門排除比例與排除後的覆蓋率，不參與判定 |
| 兩種換日旗標的一致率 | — | 計算 | **診斷**：不能代替覆蓋率檢查 |
| 尖峰漏判（`spikes_covered`） | 綁定標籤 | 同 | 綁定標籤：沒覆蓋時，閘門與出場 ×2 的規則繼續有效 |

E-017 只有在 E-016 兩個先決條件都通過、而且使用者核准第 9 節的成本公式後才能登記。

### 11.2 三種報價各自的角色（v2.5）

決策時刻 = 分鐘開盤 T + 1,000 ms（決策延遲 1 秒）。同一分鐘可能有好幾筆 tick，三種用途**分開定義**：

| 欄位 | 成本估計的歷史報價與分格 | 未篩選覆蓋率的實際成本目標 | 決策時已知的特徵與閘門報價 |
|---|---|---|---|
| 定義 | 第 d 天以前 90 天、已完成 tick 日的**每分鐘第一筆有效報價**（`minute_table().first`）；出場比值 = 該分鐘 +60 後第一筆 ÷ `quote_before`，只用 `exit_at < d` | 進場：**分鐘內第一筆有效報價**（和 E-015 相同）；出場：T+60 分鐘起第一筆（跨每日休市取重開，跨週末無目標） | 特徵 `quote_before` = **更早分鐘的最後一筆**（也是 E95x 的分母）；閘門報價 = `quote_at`：**時間 ≤ T+1,000 ms 的最後一筆有效報價**，時效 ≤ 60 秒 |
| 分格 | 換日（伺服器 23／0／1 時）與其他；E95x 依**排定**出場時刻分格 | 同左 | — |
| 和 E-015 比 | 只改抽樣（每天）和 E95x 分格鍵 | **不變** | 新增 |

**建議：橋接目標沿用「第一筆報價」**，讓 E-016 只檢驗抽樣與分格兩項改動。限制要寫明：
- E-016 通過只證明 **C95 覆蓋「分鐘第一筆報價」**，不證明它覆蓋決策當下的閘門報價，也不證明覆蓋實際成交（成交在決策之後）。
- 閘門報價是否 ≤ C95 由閘門本身逐筆檢查；未篩選時「閘門報價 > C95」的比例就是 `cost_gate` 排除比例，以診斷報告。
- 實際成交報價（決策後第一筆）沒有任何驗證，列為下游限制：E-017 只做預測、不模擬成交；第 7 節交易比較正式執行前要另外處理。

**人工時間線**（合成 tick，C95 = 35 點；已用 `minute_table` 與 `quote_at`／`cost_gate` 實際算過）：

| 例 | 這一分鐘的 tick（距 T） | 更早的 tick | 成本目標（第一筆） | `quote_before` | 閘門報價（≤ 1.0 秒） | 閘門結果 |
|---|---|---|---|---|---|---|
| A | 0.1 秒 30 點、0.9 秒 45 點、1.2 秒 60 點 | −3 秒 25 點 | **30**（0.1 秒） | 25 | **45**（0.9 秒，100 ms 前） | `cost_gate`（45 > 35） |
| B | 只有 1.2 秒 60 點 | −3 秒 25 點 | **60**（決策時還沒收到） | 25 | **25**（−3 秒那筆） | `ok`：閘門看不到 60 |
| C | 沒有 | −3 秒 25 點 | 沒有（該分鐘不在覆蓋率分母） | — | 25 | `ok` |
| D | 只有 1.2 秒 60 點 | −91 秒 25 點 | 60 | 25 | 25，已 92 秒 | `stale_quote`：不交易，但 60 仍算進**未篩選**覆蓋率 |

- 1.2 秒的 tick 在 A、B、D 都**看不到**：不能當特徵，也不能當閘門報價。
- 例 A：目標（30）和閘門報價（45）不同，所以「第一筆覆蓋率」≠「閘門報價覆蓋率」。
- 例 B：目標存在、閘門卻放行一個實際 60 點的分鐘。這就是上面寫的下游限制。
- 沒有新 tick 時：取更早的最後一筆，超過 60 秒為 `stale_quote`，從來沒有有效 tick 為 `no_quote`；兩者都不交易，不補值。

### 11.3 資料與因果時間（E-016）

- **窗口**：見 11.0。校準期 `[2023-08-07, 2025-01-01)` 只報告；評估期 `[2025-01-01, 凍結日)`。
- **證據等級**：評估期曾在 E-014、E-015 被看過，所以是開發證據。
- **成本估計**：沿用 E-015 規格：90 天歷史窗口、至少 20 個 tick 日、每格至少 200 分鐘、95 分位。
  第 d 天的上界只用 d 之前已經完成的 tick 日；出場比值只用 d 之前已經完成的出場（`exit_at < d`）。
- **預先固定的參數**（登記前固定，不看結果後挑選）：
  - 決策延遲 1 秒，也就是決策時刻 = bar 開盤 + 1,000 ms；
  - 報價時效 60 秒；
  - 沒有報價時為 `no_quote`，不交易；
  - 出場若跨過週末就沒有標籤（沿用 `max_exit_gap_minutes = 180`）；
  - 跨過每日休市的出場取重開後第一筆報價，換日分格用排定時刻。

### 11.4 可比性

- **共同可評估分鐘**：同時有 C95、E95x（排定版）、進場目標、出場目標的分鐘（和 E-015 相同的條件）。
  v2.5 修正：不再要求「有效的 `quote_at` 報價」，因為那本身就是一種篩選。
- 每個候選在共同分鐘上報告三組數字：
  1. 未經閘門篩選的覆蓋率（**判定用這一組**）；
  2. 閘門排除比例，依 `no_quote`／`invalid_quote`／`stale_quote`／`no_bound`／`cost_gate` 分開；
  3. 篩選後（閘門 `ok`）的覆蓋率。
- 不能只報篩選後的數字；篩選後的覆蓋率也不能拿來判定通過。

### 11.5 判定與限制（E-016）

- **門檻**（逐項沿用 E-015）：
  - 覆蓋率 ≥ 0.93，且 CI 下限 ≥ 0.90；
  - 每個評估年度 ≥ 0.90；
  - 換日分鐘 ≥ 0.85，其他分鐘 ≥ 0.90；
  - 整點（HH:00）≥ 0.90；
  - 未覆蓋成本占比 ≤ 0.03；
  - 上界／實際成本 ≤ 1.75。
- **多重比較**：Sidak，`trials_declared = 2`（C95 每日、E95x 排定版），再乘上執行器的家族次數。
- **信賴區間**：月區塊 bootstrap，2,000 次，seed 20261118；另外報告 3 個月區塊的 CI。
- **必須報告**：逐年、逐月（最差月份）、換日、整點的覆蓋率，以及非換日時段超過當日水準 2 倍的尖峰漏判率。
- **結果與下游**：
  - 兩項都通過 → E-017 的成本輸入可用；
  - 任一項失敗 → 方案 1 停止，H-013 的成本前置條件（排定版）不成立；
  - 某格樣本不足（< 200 分鐘或 < 20 個 tick 日）→ 該部分不可評估，視同沒有通過，不補值，也不改用 1.21。
- 橋接通過只代表「分鐘第一筆報價的成本上界在新定義下仍然保守」，**不代表閘門報價或實際成交被覆蓋，也不代表任何策略賺錢**。

### 11.6 H-013 子題（E-017）

- **不變的部分**：h = 60 分鐘，推翻條件 AUC ≤ 0.55；0.05R 條款不變，但預先宣告為 underpowered。
- **決策列**：每個整點（HH:00，伺服器時間）。
- **標籤**：Y(t) = 1 若 bid 在 (t, t+60] 的高低區間 > k × 成本(t)，成本(t) = max(C95, E95x 排定版 × 2)。
  - **k = 4 是設計假設，不是由 gemini 規則推出的門檻**。啟發來自 gemini 的 `MAX_SPREAD_TO_TP_RATIO = 0.25`
    （單方向停利距離至少為當下點差 4 倍才下單），但兩者不等價：
    - 標籤用的是未來 60 分鐘的**高低區間**，不是停利實際可達成的單方向距離；
    - 成本用的是保守上界 max(C95, E95x × 2)，不是決策當下的點差。
  - 所以 k = 4 只是事前固定的唯一候選值，用來限制搜尋；不主張「Y = 1 的時段可交易」或「達到獲利門檻」。不做 k 搜尋。
- **特徵**：`FEATURES` 的 8 項（第 2 節），全部只用 t 之前的資料；`quote_before` 見 11.2。閘門不參與 AUC 的評估列。
- **模型**：
  - 決定用的模型是 `logistic`（StandardScaler + sklearn 預設）；判定只用一個模型，trials = 1。
  - `base_rate_by_hour` 是對照組，只報告。它的角色是說明可預測性有多少來自時段；
    logistic 的 AUC 通過**只代表這個標籤可以預測**，不能單憑它推論特徵帶來額外資訊價值，也不能推論策略有經濟優勢。
- **切分**：逐月前推；purge 120 分鐘（≥ 60 分鐘的標籤窗口）；每次擬合至少 5,000 列；
  訓練資料從 2023-10-01 起逐 fold 擴張；**AUC 評估期 `[2025-01-01, 凍結日)`**（v2.5，原為 2024-04-01 起），報告逐年結果。
- **評估列**：共同資料列（第 2 節）。逐 fold 報告列數與排除原因；非有限特徵、非有限標籤一律排除。
- **整體判定**：
  - logistic AUC（點估計）≤ 0.55 → 子題推翻；依原條款的「或」，H-013 整體為 refuted。
  - AUC > 0.55 且月區塊 CI 下限 > 0.55 → 子題通過；但第二條款 underpowered，所以 H-013 整體為 **inconclusive**，不能判為 supported。
  - 其他情況 → inconclusive。
- **第二條款**：現行模型可歸屬的約 68 筆 demo 交易只做描述，依模型版本分層，不做判定。

### 11.7 候選 preregistration 字典與離線 critique（v2.5）

下面兩個 JSON 是**不完整的候選草稿**，沒有登記，也沒有寫進帳本。欄位依 `research_critic_v1` 所讀的格式。
- 正式程式還沒寫，所以 `code_files` 寫的是預定檔名與 `PENDING_*` 標記，**不是雜湊**；critic 因此會阻擋，這是如實的狀態。
- `2099-01-01` 代表凍結日，登記時換成固定日期。
- 和正文不一致時以正文為準，並視為草稿錯誤。

E-016 候選：

```json
{
  "kind": "preregistration", "id": "E-016", "hypothesis": "H-023", "role": "exploratory", "author": "researcher",
  "draft_status": "candidate, incomplete: module not written, code not frozen, not registered, reviews pending",
  "module": "exp_cost_bridge_v1",
  "code_files": {"exp_cost_bridge_v1.py": "PENDING_NOT_WRITTEN", "exp_cost_bound_v1.py": "PENDING_HASH_AT_FREEZE",
                 "exp_spread_cost_v1.py": "PENDING_HASH_AT_FREEZE", "exp_h013_tradeability_v1.py": "PENDING_HASH_AT_FREEZE"},
  "evaluation_windows": [["2025-01-01", "2099-01-01"]],
  "training_window_note": "Nothing is fitted. Each day d's C95 and E95x use only tick days in [d-90d, d) and exits with exit_at < d; the calibration window [2023-08-07, 2025-01-01) is reported only.",
  "claim_scope": "XM GOLD# demo quotes. A pass shows the daily-sampled C95 and the scheduled-exit E95x still cover the FIRST valid quote of each minute (entry) and the first quote at or after +60 minutes (exit) on 2025+ data. It does not show coverage of the decision-time gate quote (last quote at or before T+1s) or of any fill after the decision; it does not cover weekend holds, slippage, commission or swap, and tests no strategy edge. 2025+ was seen in E-014/E-015: development evidence only.",
  "decision_rule": "Sidak over trials_declared=2 (C95_daily entry, E95x_scheduled exit) times the runner's family count, month-block bootstrap 2000 draws, seed 20261118. On the common evaluable minutes of [2025-01-01, freeze) (C95, E95x, entry and exit targets all present; no gate filter) EACH candidate must, on the UNSCREENED target: coverage >= 0.93 and CI lower bound >= 0.90; every year >= 0.90; rollover minutes >= 0.85 and other minutes >= 0.90; hour-start minutes >= 0.90; uncovered cost share <= 0.03; mean bound / mean cost <= 1.75. Both pass -> the bridged cost input is usable by E-017; either fails or is not evaluable (cell < 200 minutes or < 20 tick days) -> option 1 stops. Gate-excluded share by reason and post-gate coverage are reported, never decisive. spikes_covered stays a binding label as in E-015.",
  "spec": {
    "tick_start": "2023-08-07", "tick_every_nth_day": 1, "calibration": ["2023-08-07", "2025-01-01"],
    "forward": ["2025-01-01", "2099-01-01"], "quantile": 0.95, "adaptive_days": 90, "level_days": 5,
    "min_trailing_tick_days": 20, "min_cell_minutes": 200, "horizon_minutes": 60, "max_exit_gap_minutes": 180,
    "exit_rollover_key": "scheduled", "decision_latency_ms": 1000, "max_quote_age_ms": 60000,
    "bootstrap_draws": 2000, "seed": 20261118,
    "criteria": {"coverage_min": 0.93, "coverage_ci_lower_min": 0.90, "year_coverage_min": 0.90,
                 "rollover_coverage_min": 0.85, "other_coverage_min": 0.90, "hour_start_coverage_min": 0.90,
                 "uncovered_cost_share_max": 0.03, "bound_over_cost_max": 1.75, "high_cost_miss_max": 0.5},
    "trials_declared": 2, "candidates": ["C95_daily", "E95x_scheduled"],
    "selection_rule": "no selection: both candidates must pass; nothing is chosen by forward performance",
    "cost_target": "entry: quoted spread of the first valid tick inside the minute (as E-015); exit: first quote at or after +60 minutes, across the daily break to the reopen, not across a weekend; points",
    "decision_quote": "quote_at: last valid tick with time <= minute open + 1000 ms, age <= 60 s; a gate input and diagnostic, not the coverage target",
    "cost_accounting": "points (0.01 USD/oz), (ask - bid) / point; one spread per round trip (long pays entry, short pays exit). Slippage, commission and swap are not part of the target and are not claimed.",
    "involves_trading": false,
    "not_applicable_reason": "no positions are simulated: bounds are scored on quoted-spread coverage, so direction and swap cannot enter any decision statistic",
    "data_windows": [
      {"window": ["2019-01-01", "2099-01-01"], "uses": ["spread_field"], "purpose": "rollover spike hour per month (DST check), reported"},
      {"window": ["2023-08-07", "2099-01-01"], "uses": ["spread_field", "h1_or_finer_price"], "purpose": "daily tick quotes: minute first quotes, trailing 90-day quantiles, exit ratios, decision-time quotes"}],
    "controls": ["E-015 archived scores (every-2nd-day sampling, actual-exit rollover key) reported beside the bridged ones, not recomputed",
                 "actual vs scheduled exit rollover flag agreement, reported",
                 "calibration-window scores reported beside forward scores",
                 "coverage by year, month, rollover vs other, hour start; month-block and 3-month-block CI"],
    "metrics": ["coverage", "coverage_ci", "coverage_by_year", "worst_month", "coverage_rollover", "coverage_other",
                "coverage_hour_start", "uncovered_cost_share", "bound_over_cost", "high_cost_miss_rate", "spikes_covered",
                "gate_excluded_share_by_reason", "coverage_after_gate", "rollover_flag_agreement", "coverage_ci_3month_blocks"],
    "power_note": "forward 2025-01..2026-11 is ~23 months; daily sampling gives ~480 tick days x ~1300 minutes, but minutes within a day are dependent, so ~23 month blocks carry the CI (roughly +-0.01-0.02 at a true coverage of 0.95, as in E-015). Hour-start minutes are ~1/60 of that (~10k), still enough for a 0.90 threshold; spike miss rate keeps a wide interval and stays a label."
  }
}
```

E-017 候選：

```json
{
  "kind": "preregistration", "id": "E-017", "hypothesis": "H-013", "role": "exploratory", "author": "researcher",
  "draft_status": "candidate, incomplete: run() not written, code not frozen, not registered, reviews pending, requires E-016 supported and the section 9 cost formula approved",
  "module": "exp_h013_tradeability_v1",
  "code_files": {"exp_h013_tradeability_v1.py": "PENDING_RUN_NOT_WRITTEN", "exp_cost_bridge_v1.py": "PENDING_NOT_WRITTEN",
                 "exp_cost_bound_v1.py": "PENDING_HASH_AT_FREEZE", "exp_spread_cost_v1.py": "PENDING_HASH_AT_FREEZE"},
  "evaluation_windows": [["2025-01-01", "2099-01-01"]],
  "training_window_note": "Walk-forward by month: the model for month m is fitted on hourly rows from 2023-10-01 up to the start of m minus a 120-minute purge (>= the 60-minute label window), at least 5000 rows; the first fold trains on [2023-10-01, 2025-01-01). Training-period labels use bounds that have no forward coverage qualification; they affect fit quality only and are never scored. No fold sees its own or later months.",
  "claim_scope": "XM GOLD# hourly decision times (HH:00 server). A pass shows only that the label 'bid high-low range over the next 60 minutes > 4 x max(C95, 2 x E95x)' is predictable above AUC 0.55 from past-only features on 2025+ development data. k=4 is a design assumption, not a take-profit or profit threshold; AUC above the hour-of-day baseline is not shown to add information value, and no strategy or economic edge is claimed. The 0.05R clause is underpowered, so H-013 as a whole cannot be supported by this experiment.",
  "decision_rule": "Single decisive model (trials_declared=1): logistic regression (StandardScaler + sklearn defaults) on the 8 FEATURES, AUC on the common evaluable rows of [2025-01-01, freeze), monthly walk-forward, 120-minute purge, month-block bootstrap 2000 draws, seed 20261118. AUC point estimate <= 0.55 -> sub-question refuted and, by the original 'or' clause, H-013 refuted. AUC > 0.55 and month-block CI lower bound > 0.55 -> sub-question passes but H-013 is inconclusive (0.05R clause underpowered). Otherwise inconclusive. base_rate_by_hour is reported only.",
  "spec": {
    "horizon_minutes": 60, "k": 4, "label_cost": "max(C95, 2 x E95x_scheduled); either bound missing -> no label",
    "decision_times": "every HH:00 minute, server time", "decision_latency_ms": 1000,
    "features": ["rv_30", "rv_240", "rv_1440", "hour_sin", "hour_cos", "roll_exit_sched", "quote_before", "c95"],
    "models": ["logistic", "base_rate_by_hour"], "decisive_model": "logistic", "purge_minutes": 120, "min_train_rows": 5000,
    "train_start": "2023-10-01", "fold": "calendar month", "bootstrap_draws": 2000, "seed": 20261118,
    "trials_declared": 1,
    "cost_target": "label cost per decision row (conservative quoted-spread bound from E-016, points)",
    "cost_accounting": "points (0.01 USD/oz); one spread per round trip; the direction-free label charges the dearer side max(entry C95, 2 x exit E95x), never C95 + E95x; slippage, commission and swap are not included and not claimed",
    "involves_trading": false,
    "not_applicable_reason": "forecast-only: no position is opened, so direction and swap cannot enter the AUC; the cost enters only through the label threshold",
    "data_windows": [
      {"window": ["2023-09-01", "2099-01-01"], "uses": ["h1_or_finer_price"], "purpose": "M1 bid bars: past realized volatility features and the label's high-low range"},
      {"window": ["2023-08-07", "2099-01-01"], "uses": ["spread_field", "h1_or_finer_price"], "purpose": "daily tick quotes: C95 / E95x bounds (90 days before each day) and quote_before"},
      {"window": ["2023-10-01", "2025-01-01"], "uses": ["spread_field", "h1_or_finer_price"], "purpose": "training only for the first fold; never scored"}],
    "controls": ["base_rate_by_hour on the same common rows (how much is time of day)",
                 "per-fold rows, valid labels, common rows and exclusion reasons",
                 "per-model AUC on all available rows labeled auxiliary, never compared",
                 "label base rate per year"],
    "metrics": ["auc_common", "auc_ci_month_block", "auc_by_year", "auc_by_fold", "log_loss", "brier",
                "base_rate_auc", "label_base_rate", "exclusions_by_reason"],
    "power_note": "~23 forward months x ~22 trading days x ~23 hours ~ 11k decision rows. With a label base rate between 0.2 and 0.8 the iid AUC standard error is about 0.006-0.01; month blocks (~23) widen it, so the CI lower bound > 0.55 rule separates a true AUC near 0.58+ from 0.55. The 0.05R clause has < 15% power with ~68 attributable trades (section 5)."
  }
}
```

**離線 critique 結果**（2026-11-18，`.venv` Python 3.11，唯讀；直接呼叫 `research_critic_v1.critique(p, rm.read())`，
沒有用 CLI、沒有 `--record`、沒有寫帳本，critic 未修改）。重跑方式：從本節抽出兩個 JSON 區塊，對每個呼叫 `critique()`。

```text
## E-016 (H-023)
BLOCK code: exp_cost_bridge_v1.py differs from the preregistered hash
BLOCK code: exp_cost_bound_v1.py differs from the preregistered hash
BLOCK code: exp_spread_cost_v1.py differs from the preregistered hash
BLOCK code: exp_h013_tradeability_v1.py differs from the preregistered hash
note  data reuse: evaluation windows overlap 15 earlier uses; evidence is development only
note  family 'cost_bound' already has 1 result(s); the runner widens the Sidak count
note  not ready for a formal run: 已有判定結果 E-015（supported），狀態待使用者核准 -> 使用者以決策 set_status 核准（系統建議 supported）
revise
## E-017 (H-013)
BLOCK code: exp_h013_tradeability_v1.py differs from the preregistered hash
BLOCK code: exp_cost_bridge_v1.py differs from the preregistered hash
BLOCK code: exp_cost_bound_v1.py differs from the preregistered hash
BLOCK code: exp_spread_cost_v1.py differs from the preregistered hash
note  data reuse: evaluation windows overlap 15 earlier uses; evidence is development only
note  not ready for a formal run: 使用者核准層級為 design_only（只可做設計與可行性檢查） -> 使用者以決策 set_approval 核准 formal
note  not ready for a formal run: 前置條件 prereg_v2_reviewed：H-013 新版預先登記（問題 A 事後診斷與問題 B 決策時點預測分開；成本用 E-015 上界；持有時間與 gemini 實際持倉一致）通過機械與獨立 Critic 審查 -> 依 H013_PREREG_DRAFT_V2.md 寫成實驗程式並預先登記 → Critic → 修正 → 鎖定
note  not ready for a formal run: 前置條件 trade_sample：gemini 交易樣本足以檢驗第二條款（低可交易性組回放 R 至少低 0.05R） -> 目前 demo 只有 157 筆來回（2026-05 起，全部多單，中位持倉 4.9 分鐘），差 0.05R 的標準誤約 0.17-0.18R，要約 7,000 筆才到 2 個標準誤；需改用預先登記的歷史回放（並標示模型樣本內期間）或改寫第二條款（需新版預先登記與核准）
revise
```

**狀態分類**：

| 類別 | 項目 | 處理 |
|---|---|---|
| 設計本身需要修改的阻擋 | **無**。所有欄位檢查（訓練窗口、控制組、多重比較、成本記帳、檢定力、判定門檻、資料邊界 D-016、候選與選擇規則）都通過 | — |
| 正式程式、雜湊或核准尚未完成的預期阻擋 | 兩個實驗各 4 項 `code: ... differs from the preregistered hash`：檔案不存在（`exp_cost_bridge_v1.py`）或雜湊是 `PENDING_*` 佔位 | 程式寫好並凍結後，以真實 SHA-256 登記再重跑 |
| 同上（就緒檢查，note） | E-016：H-023 的 E-015 結果待使用者核准狀態。E-017：H-013 只有 `design_only`；`prereg_v2_reviewed` 未成立；`trade_sample` 未成立 | 第 9 節決策表；方案 1 下 `trade_sample` 要由使用者核准改為「描述」 |
| 已看過資料與家族重複測試的證據限制 | 兩者評估期與帳本中 15 次既有使用重疊 → 只能是開發證據；`cost_bound` 家族已有 1 個結果 → 執行器會擴大 Sidak 次數 | 寫進 claim_scope；不能消除，只能標示 |

**人工檢查阻擋**（critic 表達不了、但正式前必須解決）：
- 正式模組不存在：E-016 的 `exp_cost_bridge_v1.py` 和 E-017 的 `run()` 都沒寫。critic 只能以「雜湊不符」間接顯示。
- 凍結日、程式雜湊都還沒固定；`2099-01-01` 只是佔位。
- E-017 的登記前提：E-016 判定兩項都通過，且使用者核准第 9 節的成本公式。critic 不檢查跨實驗的前提。
- 獨立 Critic：**待審**。作者不能自己核准自己。

### 11.8 完整交付的邊界

| 項目 | 現在 | 正式前還缺 |
|---|---|---|
| 成本契約、標籤、特徵、閘門、前推切分、模型集合、檢定力 | `exp_h013_tradeability_v1.py`，合成資料測試 | — |
| 候選 preregistration 與機械 critique | 11.7（不完整草稿、離線結果） | 程式完成後重新產生雜湊並重跑 critique |
| E-016 `run()` | 無 | 新模組 `exp_cost_bridge_v1.py`（不改 E-015 已登記雜湊的 `exp_cost_bound_v1.py`）：每日 tick 抓取、排定版 E95x、共同分鐘、三組覆蓋率、報告 |
| E-017 `run()` | 無 | 每個整點的 C95／E95x、標籤、特徵、`question_b`、AUC 的月區塊 CI、逐 fold 表、報告 |
| 封存產物 | — | `report.md`、`metrics.json`、逐 fold／逐月表、共同資料列遮罩的雜湊、`environment.txt`、manifest（由 `research_run_v1` 產生） |

**還沒做的事**：正式程式、雜湊綁定、獨立審查、使用者核准、真實資料的橋接驗證，以及 H-013 子題計算。
現在只補草稿、合成檢查與離線 critique，不寫入真實的 preregistration，也不提升 formal。

## 12. 審查與登記流程（對齊既有工具）

`research_critic_v1.py E-xxx` 的 CLI 會從帳本查 preregistration，所以只能用在已登記的實驗。流程分兩階段：

**第一階段：設計草稿與候選程式的離線審查**（現在可以做）
- 已完成一次（11.7）：對候選的 preregistration 字典直接呼叫純函式 `research_critic_v1.critique(p, events)`，不加 `--record`，不寫真實帳本。
- 獨立 Critic 的意見以「待審」保留在草稿或回覆裡；作者不能自己核准自己。

**第二階段：登記與正式執行**（**本輪不做**）
1. 使用者選定方案，候選程式就緒；
2. 凍結程式，把 preregistration（含 `code_files` 雜湊）寫進帳本；
3. 執行 `research_critic_v1.py E-xxx --record` 登錄機械審查，再由獨立 Critic 登錄審查；
4. 依 Priority Engine 的前置條件，使用者以決策核准 formal；
5. 使用者手動執行 `run_research_experiment.bat E-xxx`。

任何設計改動若需要新編號，要先查帳本、提出候選編號，不覆寫既有的 preregistration 或歷史 run。
