# Demo Trading Research Foundation V1

本版建立 Demo Forward 資料與執行語義基礎，不是新模型、Generation、回測或 promotion。
策略基準為 commit `003c2ff` 的 `gemini.py`。模型仍為
`gold_long_recent_candidate_xgb.json`，SHA256
`2dc32e3b3c0ea6ca8fa2e30187bebf8ff3f7e7e03109b39b3f70f013e3a755f2`。

## Runtime 交易語義規格

以下描述目前行為；已知差異沒有藉此改成另一套策略。

| 項目 | 現行 runtime 定義／限制 |
|---|---|
| Decision Time | 主迴圈僅在 `utc_now().second == 0` 嘗試預測。`event_time` 是該次取樣前的 UTC 時間，不是 broker fill time。處理延遲沒有補單。 |
| M1 availability | `copy_rates_from_pos(..., start_pos=1, count=400)`；最後一根由 MT5 視為已完成的 M1 產生特徵。`bar_time` 是該 bar 的開盤時間，不是決策時間。 |
| MTF availability | 20 種高週期各自查詢 `start_pos=1, count=80`，使用最後完成 bar 的 CLOSE 與 MA20。各查詢非同一個原子 snapshot；尚未加入 freshness gate。 |
| 時間對齊 | runtime 沒有研究 pipeline 的額外 feature shift。研究先 shift HTF trend，as-of 合併後再 shift 整個矩陣，週期邊界可能不同。測試保留這個差異，不修改模型輸入。 |
| Session | 使用 `bar_time.hour/dayofweek`，小時 `{0,1,2,3,4,8,9,11,12,17,18,19,20,22,23}`、週一至五。API epoch 轉 UTC 的既有方式保留；過去推定 UTC+3 不自動套用到現在。 |
| RSI | 完成 M1 的 rolling-14 RSI；`RSI >= 22`，排除含邊界的 `[35,45]`。 |
| 模型門檻 | Long binary，`CONF_THRESHOLD=0.70`，`edge>=0`，buy>=sell。這是模型分數，不宣稱校準後勝率。lower=1.01、expanded=False、meta=False。 |
| Position occupancy | symbol=GOLD# 且 magic=20260514 的持倉；最多一個。其他 magic 的部位不算入此策略的限制。查詢 None 現在阻擋新倉。 |
| Cooldown | 舊帳本最後一筆損益 <0 時，距離最後出場 <15 分鐘不進場；恰好15分鐘可進場。保留舊 OUT-only 損益定義以避免改變策略。新完整帳本不反向驅動此 guard。 |
| Rolling risk | 舊帳本最近30筆，至少18筆，PF<1.15時 risk multiplier=0.5。其進場費用／部分平倉限制明確保留，後續改用完整帳本須另案確認。 |
| 日損 | 本機午夜至本機現在的 MT5 成交淨損益，加本策略浮動損益；<= -當前balance*5%阻擋新倉，不自動平倉。保留本機日界與浮盈未含全部費用的原行為。歷史查詢失敗不再視為0。 |
| Spread | `(ask-bid)/symbol_info.point`；上限維持 `min(100,max(45,tp*0.25/0.01))`。因此最低TP=1.5可能允許30%，本版只揭露，不更動。 |
| Lot / risk | 每筆balance*0.014*既有risk multiplier；原本固定100與0.01換算、broker volume rounding保留。研究用初始R另由實際fill volume/price到原SL的broker損失試算取得；無法取得就Unknown。 |
| Entry / fill | 在送出時重新查tick，BUY用ask，IOC。request price與result broker price分開保存。部分成交照原runtime返回碼行為，不偷偷改成全額成交；新帳本按deals計量。 |
| TP / SL | BUY request price + max(ATR*1.3,1.5)、-max(ATR*1.6,0.6)。broker管理SL/TP；實際成交價格可能與request不同，帳本使用實際deals。 |
| Timeout | 每次迴圈先管理部位，`time.time()-position.time >= 90*60`發出市價平倉；以實際填單時間為準，並非第90根K棒。trailing=False。 |
| Re-entry | 每個新的完成bar最多預測一次；持倉歸零後，下一次eligible bar仍可進場，無需分數先跌回門檻。即使同一persistent episode亦可能重進。 |
| Signal episode | runtime不以episode限制進場。研究若計算episode，必須另外記錄定義；不可把每根合格bar當成獨立交易。 |
| 同bar TP+SL | 真實結果以broker deal為準。只有OHLC時先後為Unknown；`intrabar_evidence()`另輸出既有研究`stop_first`慣例，不能把慣例當作觀測事實。 |

既有 S0→S1 改 HIGH/LOW、S1→S2 改 barwise entry、S2→S3 改 spread/cost、
S3→S4 改風控、S4→S5 改 OPEN／入場bar／wall-clock。歷史 S5 固定0.75且仍是近似，
不是本版runtime的完整替身。本版不重跑、不修改任何封存 S0–S5 成果。

## 最低安全與啟動版本

`XM_EXPECTED_LOGIN` 與 `XM_EXPECTED_SERVER` 必須由使用者明確設定。
啟動、每次迴圈與每次order_send都核對預期身分及
`account.trade_mode == ACCOUNT_TRADE_MODE_DEMO`。沒有設定、查詢失敗或身分切換會拒絕交易，
不自動把目前帳號當作預期帳號。這次開發沒有查詢實際帳號，也沒有設定這兩個值。

所有委託集中在 `send_demo_order()`；DRY_RUN在此及開倉／平倉入口阻擋，
不會送出開倉、逾時平倉或未來的SLTP請求。持倉／成交歷史查詢None會明確丟出安全錯誤。

每個通過帳戶驗證的Gemini啟動建立新目錄：
`runtime_logs/demo_foundation_v1/<UTC+UUID run_id>/`。
`manifest.json`保存Git commit/dirty、runtime source hash、model hash、configuration hash、
策略常數、threshold、TP/SL/risk、帳戶/server/currency與UTC啟動時間。
`signals.jsonl`含run身分；`orders.jsonl`先保存intent，再保存result/error及關聯ID；
`deals.jsonl`含broker deal/order/position ID與`capture_run_id`。
觀測到舊deal不代表它由目前run建立。帳本只有明確entry order metadata才能歸屬原run。

既有CSV只沿用原append用途，不做schema migration或覆寫；header不同時拒絕寫入。
新資料的完整版本追蹤以run目錄為準。既有launcher會截斷stdout，故不能用它取代run manifest。
本版不重啟現行程序；新保護須使用者日後啟動新版本才生效。

## Forward Ledger

`forward_trading_ledger_v1.py`沒有MT5/model import，純粹整理觀測deal，沒有市場模擬。
按account/server/position分組，deal ID去重、Order ID提供明確關聯；不同內容的重複deal標Unknown。
IN、OUT、OUT_BY與分批成交按volume核對；INOUT reversal、出場缺進場、反向／超量平倉、
缺欄位、非有限值、未知時區或不明帳戶範圍皆Unknown。部分平倉剩餘volume>0為open。
不以時間接近或價格相似猜position、不替舊資料建立run ID。

每筆complete交易淨損益=sum(profit+commission+swap+fee)，explicit_cost為負的三項費用總和，
可表達rebate／正swap。spread/slippage已在實際fill與broker profit中，不再扣一次。
open只呈現observed_net_pnl，不宣稱最終交易損益；Unknown保留問題與觀測值但不納入績效。

Net PnL、WR、PF、平均贏虧與Mean-R使用觀測視窗內完成交易，視窗為[start,end)。
Mean-R僅納入每張entry order的實際filled volume與全部entry deals相符，且有有效初始SL損失試算的樣本。
R分母為各entry fills到其原SL的捕捉風險加總，不拿planned budget代替，也不把後來SL當初始SL。
沒有風險資料的交易仍可計算cash PnL，另計missing_risk_samples。

Realized Max Drawdown按視窗內已知deals的現金增減計算，包含open交易已收費用／部分平倉，
同時刻的deals合併成一個cash event；單位是帳戶幣別，不是DD%或含浮動損益的equity DD。
observed_realized_cash_pnl與完成交易cohort的net_pnl可能不同；不得混用。
Trades/day以明確指定視窗的經過日數計算，不以第一筆至最後一筆交易壓縮分母。
不同帳戶/server/currency拒絕混算。缺少完整外部帳戶statement時不宣稱broker歷史完整。
PF無虧損且有獲利時以null配`profit_factor_status=infinite`表達；沒有樣本則undefined。

此工具沒有預設資料路徑；手動使用需明確指定`--deals`（可重複）、`--orders`、`--signals`、
`--start`、`--end`、`--output`。JSONL與CSV皆支援；output必須是新目錄，拒絕holdout路徑。
實際交易績效分析／研究仍遵守AGENTS與手動核准流程；不得把任意舊資料重新稱為untouched。
本次只用合成資料呼叫帳本函式，沒有對現有交易紀錄計算新績效。

## 隔離驗證

在`D:\XM\數據`執行：

```powershell
.\.venv\Scripts\python.exe -B -m unittest -v test_demo_trading_foundation_v1
```

測試在import Gemini前以Fake MT5與Fake XGBoost取代真實套件；禁止initialize與模型建構。
只使用暫存檔與合成資料，涵蓋最低安全、委託與版本關聯、成本／部分平倉、缺漏資料、
明確視窗績效、已完成bars、MTF邊界差異、0.70邊界與既有策略回歸。
這不是訓練、歷史回測、策略PASS或任何實際MT5驗證。
