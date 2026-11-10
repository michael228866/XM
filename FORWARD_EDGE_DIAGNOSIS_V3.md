# Forward Trading Edge Diagnosis V3

## 範圍

此工具只診斷 V2 已封存的實際 Forward 成交、訊號及指定的佐證檔案。不連線 MT5、不啟動 Gemini、不載入模型、不模擬交易路徑，也不讀市場 bars 或 future holdout。

入口：[forward_edge_diagnosis_v3.py](forward_edge_diagnosis_v3.py)。沿用 V2 解析、時間、金額與輸出工具，不修改 V1/V2 的帳本規則。

## 執行

在本目錄執行，`--output` 必須是不存在的新目錄：

```powershell
.\.venv\Scripts\python.exe -B forward_edge_diagnosis_v3.py `
  --v2-dir forward_reports/20261027_forward_ledger_verification_v2_r2 `
  --output forward_reports/my_forward_edge_diagnosis
```

這個基本命令只使用 V2，不自行尋找其他證據。沒有指定 SDK enum 檔時，出場原因維持 Unknown。

本次包含帳戶情境、歷史成交快照、Git來源、SDK常數及既有日誌的完整成果位於：

`forward_reports/20261029_forward_edge_diagnosis_v3/`

其 `manifest.json` 列出每項補充來源及 SHA256。帳戶情境 manifest 與歷史成交快照都先核對既有 FINALIZED seal；這些 checksum 不等於獨立 broker statement。

## 輸出與重現

- `report.md` / `report.json`：分組績效、證據等級、統計限制、前三項問題及下一個假說。
- `period_diagnosis.csv`：日／ISO週／月、有紀錄期間的統計；沒有紀錄不代表已確認未交易。
- `cost_sensitivity.csv`：記錄基準與每筆額外成本 0.1、0.25、0.5、1、2 的假設情境。
- `trade_attribution.csv` / `coverage.csv`：逐筆與整體版本／帳戶歸屬。
- `exit_classification.csv`：broker reason 與 comment 衝突保留 Unknown，不從持倉時間猜 Timeout。
- `holding_bands.csv` / `loss_concentration.csv`：持倉期間與虧損集中度。
- `session_months.csv` / `score_bands.csv` / `source_versions.csv`：描述性切片與來源設定差異。
- `v2/`、`evidence/`、`source/`：本次使用的精確輸入及程式副本。

在成果目錄執行下列程式，使用保留來源重現到新的 `reproduced/`；不依賴原研究資料路徑或 Git 歷史仍然存在：

```powershell
..\..\.venv\Scripts\python.exe -B -c "import json,subprocess,sys; a=json.load(open('manifest.json',encoding='utf-8'))['reproduce_argv']; a[0]=sys.executable; subprocess.run(a,check=True)"
```

既有輸出目錄拒絕覆寫。V2 output/source/input seal 與補充來源 seal 不符時停止，不產生正式結果。`report.json` 和全部 CSV/Markdown 可逐位元重現；`manifest.json` 的建立時間、路徑與執行環境屬於本次執行紀錄。

## 已確認的資料限制

同一批 252 Deals／126 Positions 記錄內對帳差為 0.00；嚴格帳戶／版本狀態仍全為 Unknown。初始風險完整樣本 N=0，Mean-R 不估算。

| 組別 | N | Net PnL | PF | WR | 損益兩平 WR | 平均每筆 |
|---|---:|---:|---:|---:|---:|---:|
| 早期、時區未知 | 37 | +160.02 | 1.6443 | 72.97% | 62.15% | +4.3249 |
| UTC標記 | 89 | -115.65 | 0.8604 | 55.06% | 58.74% | -1.2994 |

金額是**未驗證帳戶幣別的記錄單位**。不得以合併 +44.37 宣稱現行策略有正期望。

歷史擷取 manifest 提供登入／Server／USD情境，但沒有這126筆成交與該帳戶的可信關聯；Demo trade mode 也未由此確認。Commission、Swap、Fee全部記錄為0，外部原因仍Unknown；來源程式的 fee fallback 不能證明實際費用為0。

沒有可驗證的0.70或0.75成交版本；分數可連結101筆，[0.70,0.75) N=0、>=0.75 N=5，不能進行門檻優劣判定。

## 統計方法

金額沿用 Decimal 彙總。Gross Profit/Loss 是獲利／虧損的淨交易總額；另保留未扣顯式成本的正／負 broker profit。Break-even WR 使用相同交易樣本的平均盈虧，平手交易調整到全樣本分母；這是代數分解，不是獨立估計的成功門檻。

假設成本只對每筆完成交易扣一次，與記錄成本分欄；不估造費用、不再次扣 Spread、不挑選最佳成本參數。

Bootstrap 固定 seed=42、2000次，以完整觀測窗的 ISO日曆週為群集，抽相同數目的週且放回，保留週內全部交易及無紀錄週。保留週內相依，但無法保留跨週相依與市況順序；部分邊界週一併保留。早期僅4個有效週，UTC為12個有效週／21日曆週。兩段平均每筆損益95%描述區間皆跨0，不能證實穩定交易優勢。無限PF邊界輸出null並記錄數量，不能轉成有限PF。

持倉與報酬只有關聯，TP/SL結果本身會決定持倉時間；手數、未知版本與市場情境可能混雜。訊號列不是獨立事件，重疊 gate 次數不能相加成可恢復交易數。Session只使用記錄UTC的8小時區塊，不冒充已驗證市場時區。

本次已查看的歷史資料不是新的 Untouched OOS。不由切片PF推薦交易過濾條件，不建立 Generation，不變更策略。

## 驗證

```powershell
.\.venv\Scripts\python.exe -B -m unittest test_demo_trading_foundation_v1 test_forward_test_report_v2 test_forward_edge_diagnosis_v3
```

本次 Python 3.11及3.10皆通過101項隔離測試，其中23項為V3新增。測試僅使用合成交易與Fake MT5；測試輸出的 `Timeout close: ticket=7` 是Fake MT5案例，不是真實委託。
