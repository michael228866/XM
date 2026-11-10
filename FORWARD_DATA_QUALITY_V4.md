# Forward 資料品質驗收 V4

## 用途與通過標準

入口為 `forward_data_quality_v4.py`，只讀明確指定的紀錄並建立新報告目錄；不匯入 MT5、不啟動 Gemini、不載入模型、不訓練、不回測，也不修改任何交易設定。沿用 V1 Ledger、V2 訊號與版本關聯、V3 封存驗證。

`PASS` 僅代表**指定檔案的資料契約**符合下列要求，不是 broker 認證、獲利證明、策略 promotion 或啟動授權：

1. 每筆交易符合明確提供的 Account／Server／Currency；原始交易生命週期可重建，時間有時區。
2. 每筆新交易的進場訊號、Order ID、原始 Run ID、模型／設定 hash 都能與同一份有效 manifest 關聯。Manifest 必須記錄 Demo mode=0、UTC 啟動時間、Git commit 與有效來源 hash；設定 hash 必須重新計算一致。
3. Manifest 啟動不可晚於進場；`capture_run_id` 不能當作原始 Run ID。不同模型、設定或 Runtime 來源版本，必須分開驗收；同版本重啟可有不同 Run ID。
4. 初始風險必須是已記錄的 broker 計算結果，含完整成交量、實際成交價與請求的初始 SL。計畫風險額度不替代初始風險。
5. 進場決策紀錄有有效分數／時間，Order 有請求報價與實際成交價；成交量及成交金額與原始 Deals 一致。多次成交採完整成交量與價格加權對照，不猜測允許誤差。
6. Profit／Commission／Swap／Fee 皆為有效數值。逐筆 Deal 與明確提供的 broker 匯出核對，ID、帳戶、價格、數量、成本、時間均一致；broker 匯出多出的 Deal 也列為問題。
7. 每筆及總計的原始成交損益與 Ledger 對帳差必須為0，不允許正負差互相抵銷後冒充通過。不另外扣 Spread。
8. 完成交易有可靠的出場 reason；reason/comment 衝突、缺漏或 EXPERT 無法辨認觸發原因時保留 Unknown。不能由持倉時間或模糊 comment 猜 Timeout。

所有適用檢查必須100%通過。尚未平倉的交易保留 `Pending_open_position`，出場檢查不適用，不能算成已完成績效樣本。即使只有未平倉資料通過契約，也沒有足夠的已完成績效樣本。

`NOT_READY` 表示缺證據或資料不一致，缺漏逐筆列出；既有資料不會被補造或修寫。

## 使用方式

### 已封存 V2 舊資料

在專案目錄執行，輸出必須是不存在的新目錄：

```powershell
.\.venv\Scripts\python.exe -B forward_data_quality_v4.py `
  --v2-dir forward_reports/20261027_forward_ledger_verification_v2_r2 `
  --output forward_reports/my_forward_data_quality_v4
```

沒有明確帳戶與版本證據的舊資料預期得到 `NOT_READY`，不會自動套用目前設定。這次實際驗收輸出為 `forward_reports/20261029_forward_data_quality_v4/`。

### 新 Forward 期間

以下為參數範本，請替換 Run 路徑、實際 Demo 帳號／Server／幣別及 broker 匯出；不得從舊績效反推這些值。

```text
python -B forward_data_quality_v4.py
  --deals <RUN_DIR>/deals.jsonl
  --orders <RUN_DIR>/orders.jsonl
  --signals <RUN_DIR>/signals.jsonl
  --manifests <RUN_DIR>/manifest.json
  --broker-deals <NORMALIZED_BROKER_DEALS.csv>
  --expected-login <DEMO_LOGIN>
  --expected-server <EXACT_SERVER>
  --expected-currency <ACCOUNT_CURRENCY>
  --output <NEW_REPORT_DIRECTORY>
```

每種輸入可重複指定多次，用於同一凍結版本的多個 Run。`--v2-dir` 不可與 Runtime 輸入混用；同一檔案不可同時充當 Runtime 與 broker 證據。

Orders 必須為 JSONL，保留 `request` 巢狀 JSON object；Run manifest 為原始 JSON。Deal／Signal 可用符合專案欄位的 CSV 或 JSONL，每檔最後須有換行，避免把寫到一半的最後一列當作完整資料。

### Broker 成交資料契約

`--broker-deals` 接受已正規化 CSV／JSONL，不直接解析 MT5 HTML、XLSX 或任意券商的原生欄名。必要欄位：

```text
account_login,account_server,currency,deal_id,order_id,position_id,
deal_time,symbol,magic,entry,type,volume,price,profit,commission,swap,fee
```

完成交易另外需要 `reason`；`comment` 有記錄時一併保留。時間必須保留原始時區證據，不可把未知時間直接加上 UTC。數值正規化不得填補未記錄費用，不得用單筆淨利取代各成本欄位。

此輸入必須來自使用者另行取得、可追溯的 broker 匯出，且涵蓋與指定 Runtime 相同的成交範圍。工具可以檢查逐筆一致性，不能鑑定檔案真偽、取得權限或證明帳戶所有費用均在成交表內。若有額外帳戶收費，仍須另行核對，不得將零費用欄視為已證實零成本。

目前 Runtime 的 Deals JSONL 沒有保存出場 reason；需要有 reason 的 broker 匯出才能補足驗收證據。工具只做對照，不改寫原始 JSONL。

## 輸出、重現與退出碼

- `report.md`／`report.json`：整體狀態、缺漏數、限制。
- `trades.csv`：逐筆生命週期、帳戶、Run／模型／設定／來源版本、初始風險、出場原因及各檢查狀態。
- `coverage.csv`：每項檢查的有效分母、通過樣本及覆蓋率。
- `issues.csv`：逐筆與整體問題；禁止以 Unknown 當作通過。
- `inputs/`／`source/`／`manifest.json`：原始輸入位元組、使用程式、SHA256、預期帳戶、V2父封存識別與重現命令。

退出碼：`0=PASS`、`2=NOT_READY`、`1=輸入／封存／IO錯誤`。不修改外部交易程序，也不替 Runtime 自動開關下單。

在報告目錄使用專案 Python 執行 manifest 的 `reproduce_argv`，會先驗證保留輸入、程式及預期帳戶，再建立新的 `reproduced/`：

```powershell
..\..\.venv\Scripts\python.exe -B -c "import json,subprocess,sys; a=json.load(open('manifest.json',encoding='utf-8'))['reproduce_argv']; a[0]=sys.executable; r=subprocess.run(a); sys.exit(r.returncode)"
```

重現得到 `NOT_READY` 時，退出碼2是驗收結果，不代表執行器故障。Checksum 是本機完整性檢查，不是外部簽章。

## 隔離測試

```powershell
.\.venv\Scripts\python.exe -B -m unittest test_demo_trading_foundation_v1 test_forward_test_report_v2 test_forward_edge_diagnosis_v3 test_forward_data_quality_v4
```

只使用合成資料與既有 Fake MT5。未執行真實交易，也未確認目前連線的帳戶。

本次 Python 3.11 與 3.10 各通過127項測試，其中26項為V4新增；語法與 `git diff --check` 通過。實際V2資料驗收為 `NOT_READY`、退出碼2：126筆記錄內對帳差0.00，帳戶／版本／初始風險／broker對照通過數皆為0；有效成本欄位126筆不等於已驗證全部費用。

使用保留副本重現後，5個報告檔案逐位元一致，輸入及程式checksum相符。既有交易程式、模型、政策、V1/V2共享模組等7個保護檔案，以及V3報告與使用模組皆驗證未變更。
