# Forward Test Ledger Verification V2

`forward_test_report_v2.py` 是獨立、標準函式庫報告入口；不匯入 Gemini、MT5 或 XGBoost。
僅讀取明確指定的成交／訊號／委託／manifest，不模擬交易、不改參數、不啟動訓練。
使用者已明確授權本次唯讀實際成交對帳與描述統計；這不是歷史策略回測或 promotion。

## 執行

在 `D:\XM\數據`：

```powershell
.\.venv\Scripts\python.exe -B forward_test_report_v2.py --deals gemini_trade_history.csv --signals gemini_signal_log.csv --start 2026-06-08T00:00:00Z --end 2026-10-27T00:00:00Z --recorded-start 2026-05-15T00:00:00 --recorded-end 2026-06-06T00:00:00 --output forward_reports/NEW_UNIQUE_DIRECTORY
```

`--deals`、`--signals`、`--orders`、`--manifests` 可重複指定。沒有隱藏的預設資料來源。
輸出目錄必須不存在；工具拒絕覆寫、拒絕任何 `future_holdout` 路徑。
UTC與無時區記錄的觀測視窗分開指定，不為舊時間猜UTC offset。
未指定無時區視窗時，可呈現原檔數值／日期分組，但該分組trades/day為Unknown。
期間為[start,end)，頻率分母使用明確視窗，不縮成有交易的日數。

## 兩層證據

V1的嚴格`state`不放寬：缺帳戶、幣別或時區仍然Unknown。
V2新增`accounting_state`，只在明確Deal／Order／Position關聯、完整數字成本、方向與volume核對
均通過時，標示「記錄內數值可對帳」；這不補造帳戶、Server、幣別或版本。

因此檔案記錄的Net PnL、WR與PF可以有有效樣本，同時所有交易的嚴格狀態仍為Unknown。
這些值的單位為`recorded_file_units`；未查驗帳戶幣別前不得標USD或稱為已認證的單一Demo帳戶績效。
不同已知／未知scope或幣別的資料拒絕混算，須先分組明確輸入。
無法排序的時間情境不混算全期間drawdown、連敗或頻率，另列各時間分組。

成本使用broker成交欄位`profit+commission+swap+fee`，不再扣spread。
成本欄缺值、數值無效、同ID內容衝突、部分進場歷史缺失、反向／超量出場與混合時間情境會排除並列原因。
成本欄全部為0，只證明檔案記錄0，不代表外部費用完整保留。
INOUT reversal仍為Unknown；不能把完整volume平衡誤稱已核對broker statement全部歷史。

每項指標含有效`n`。Mean-R只使用V1捕捉的完整初始SL風險，不以舊signal risk_budget推估。
已實現回撤基於已觀測deal現金流，含進場費用與部分平倉，沒有浮動PnL／百分比DD。
PF無虧損時以null及infinite狀態表達；零樣本undefined。
平均贏虧與Payoff Ratio使用完整平倉的記錄淨損益。

## 版本與診斷

Signal只以明確deal/order/position ID關聯；ID相互矛盾、多組不同分數或關聯不完整時Unknown。
分數連結不等於模型版本：必須有相符Run ID、模型hash、設定hash、帳戶/server及有效manifest，
才能歸屬門檻、模型與程式。設定hash需重算一致，不能拿目前Git HEAD／0.70設定補歷史版本。

Session固定為「記錄UTC小時」00–08、08–16、16–24；非經獨立認證的市場時區。
無時區記錄Session=Unknown。分數帶固定為<0.70、[0.70,0.75)、[0.75,1]與Unknown，
舊模型可能混雜，因此不能據此宣稱0.70優於0.75。
持倉區間固定<=15、15–60、60–90、>90分鐘，分開顯示原始時間情境，僅作描述。
小於20筆標示樣本不足；超過20筆仍不代表統計顯著或可預測正期望。
每日／每週保留明確視窗內沒有成交紀錄的日期，不能據此保證實際沒有交易。

## 輸出與重現

- `report.md`、`report.json`：結果、樣本數、對帳與限制。
- `summary.csv`、`trades.csv`：逐指標與逐position追溯。
- `daily.csv`、`weekly.csv`、`sessions.csv`、`score_bands.csv`、`holding_bands.csv`。
- `time_partitions.csv`、`frequency_distribution.csv`：分開時間情境的回撤、連敗、頻率。
- `inputs/`：本次實際讀取的原樣快照；原始檔不修改。
- `source/`：報告、V1帳本與必要支援程式的精確快照。
- `manifest.json`：輸入／程式／輸出SHA256、Git commit/dirty、Python版本、視窗與`reproduce_argv`。

輸入若正在被原runtime追加，保留當次讀取bytes與變動標記；欄位數不符或未結束的最後一行會列入rejected，
不自行補造。重現時以inputs快照為準，不重新讀持續變動的來源。
在報告目錄內依manifest的reproduce_argv執行，可在新的reproduced目錄重建數值結果。
新報告的產生時間、來源路徑及Git工作區狀態可以不同；數值、帳本、分組與輸入hash應一致。
報告與快照在本機forward_reports保留，不宣稱已遠端封存。

## 隔離測試

```powershell
.\.venv\Scripts\python.exe -B -m unittest -q test_demo_trading_foundation_v1 test_forward_test_report_v2
```

只使用合成交易與暫存資料，測試成本、部分平倉、重複／缺失、Unknown範圍、時間分組、指標樣本數、
版本歸屬、拒絕holdout、唯讀與快照重現，不讀本機實際交易紀錄。
