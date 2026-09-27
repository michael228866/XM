# 手動訓練政策 v1

模型訓練、重訓、策略評估及 promotion research 由使用者負責。
Codex、排程器、開機流程與 CI 不得自動啟動。使用者雙擊 `RUN_TRAINING.bat`，不必輸入指令。
自動 raw capture、heartbeat 與 chain 檢查照常運作；手動啟動不代表解除 holdout 封鎖，也不代表 production promotion。

目前未指定核准訓練腳本、設定與資料集。Launcher 明確阻擋，而不挑選已凍結 family 或重跑 one-shot。
核准後須在 `training_launcher_config_v1.json` 版本化登錄腳本、validator、設定及每個資料檔的 SHA256。
介面為 `run_training(run_dir, configuration)`，獨立 validator 為 `validate(run_dir)`。
訓練程式須完成 repository manifest/report/metrics 規約；launcher 封存結果，不自動 promotion。

Interpreter 固定為已驗證的 repository `.venv\Scripts\python.exe`，Python 3.11；不自動換成其他環境。
BAT 會留在完成或失敗畫面。沒有建立 EXE、桌面捷徑或訓練排程。

## Holdout guard 的範圍

預設拒絕解析後落在 `future_holdout/gold_s4_v4` 的 open/listdir/scandir，包括相對路徑與 symlink。
訓練程序也拒絕外部程序及後續 ctypes library 載入。
CPython audit hook 不是作業系統隔離；不能宣稱阻擋所有惡意 native I/O 或 hard-link 別名。
因此未審核原生讀檔能力的 workflow 不可啟用；目前所有實際訓練均停用。
正常 Python exception／Ctrl+C 可封存 aborted；Windows 強制結束或斷電無法保證 finally 執行，需保留未完成 run 作後續 aborted 歸檔，不可宣稱 PASS。

## 基礎設施測試紀錄

20260927T025857Z、20260927T030416Z、20260927T030451Z 的
`launcher_infrastructure_fixture` 是合成取消測試，未執行任何模型訓練。
測試最初未將 `runs_root` 明確傳入，因而落在 repository training_runs；
三份 aborted 封存原樣保留。測試已修正為明確使用暫存目錄，不重用這些 run。
