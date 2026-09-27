# GOLD 手動訓練綁定 v1 — PARTIAL

`RUN_TRAINING.bat` 不需參數，但本版本仍顯示 `NOT_READY`，不會下載真實資料或開始訓練。
這是完成 provenance 調查後的明確限制，**不是已完成一鍵訓練**。

## 尚未核准唯一預設

完整候選與逐檔 SHA256 見 `gold_manual_training_provenance_v1.json`。

| 類別 | 腳本 / 已封存 run | 缺口 |
|---|---|---|
| PRODUCTION_MODEL_REPRODUCTION | `gold_long_recent_walk_forward.py` | README 確認產生 protected model；舊 main 直接覆寫 production，原始 raw snapshot/environment 不完整，end 取即時 tick。 |
| B0_BASELINE_RESEARCH | `20260913T141817Z_gemini_cftc_gold_cot_b0_b1_v1/training_script.py` | 正式入口同時跑 B0/B1，比較流程不等於 B0 單獨重訓。 |
| S4_SECONDARY_RESEARCH | `gold_independent_secondary_classifier_v1.py` / `20260915T151723Z_gold_independent_secondary_classifier_v1` | 已執行的固定 discovery；要求 archived B0 及精確 X/y。 |
| S4_SECONDARY_RESEARCH | `gold_secondary_s4_confirmation_v1.py` / `20260919T124853Z_gold_secondary_s4_confirmation_v1` | 最新相關歷史模型訓練實作；固定三 fold confirmation，未建立日常重訓或最後一個 full-fit model 的契約。 |

S4 的新近程度不能自動授權重新執行已凍結研究。所有相關候選使用 GOLD#；GAUCNH# 的獨立研究不可替代。
舊 production 標籤是 240-row barrier；B0/S4 使用 execution-aligned C1，兩者不可混用。

## 資料缺口

C1 manifest 記載 naive broker/export timestamps，historical UTC mapping unresolved。
21 個原始 CSV 身分已記錄，範圍延伸至 2026-05；S4 scoring 為 2018–2024，訓練為各 fold 前 18 個月。
這些資料範圍和標籤成熟度要求不等於可任意縮短或替換 CSV 的許可。
目前 10800-second capture offset 不是整段歷史的時間認證。
新抓 MT5 bars 不能宣稱與 frozen X/y 雜湊相同。本次未載入完整 CSV、fit 或執行策略評估。

`gold_historical_training_data_manager_v1.py` 僅有合成資料測試過的元件：
明確 native symbol/timeframe、UTC cutoff、完整預期 timestamp inventory、OHLC/volume/spread、順序/重複/schema/hash 檢查，缺口 fetch，及不可覆寫的 cache。
完整 market-session inventory 與 clock mapping 必須先認證；不由 endpoints 猜測完整度。
CSV fallback 支援已認證的 canonical UTF-8 schema；MT5 `<DATE>/<TIME>` 等格式尚需明確映射，不能只靠檔名推論。
目前無已核准 requirements、無 eligible 真實 cache、無自動 import 掃描，且未做 live fetch。AUTO_FETCH_STATUS=PARTIAL。
放進 import 的 CSV 不會自動變成合格資料。`UNKNOWN` 不合格。

## 安全與使用者介面

使用者雙擊 BAT → 檢查環境 → 驗證 Explorer/CMD 來源 → 發出 process-local 30 秒一次性憑證 → 顯示具體未核准原因 → 保留視窗。
靜態環境 marker 不是授權憑證；憑證也不是密碼認證系統。
舊未綁定的動態 training import 路徑已移除；修改 JSON 的 APPROVED 值不能打開不存在的安全 adapter。
CHECK_STATUS 分別顯示 capture 與 training readiness。

holdout 樹及 cutoff 檢查已保留；CPython audit hook 不是防惡意 native code 的 OS sandbox。
未來 adapter 必須審核 native I/O，且 training worker 對 holdout 樹零依賴。
production、既有 finalized runs、capture code/policy/task 全部保持不變。

## 解鎖下一步需要的具體決策

1. 核准唯一目的：production 歷史重現、B0-only，或新的 S4 歷史重訓規格（不得重試舊 run）。
2. 針對該目的確定輸出模型／fold contract 與 independent validator。
3. 認證歷史 source clock、所需原始範圍與 coverage，決定嚴格舊 CSV 身分或另行核准 native MT5 替換資料的研究規格。
4. 在上述確定後改接既有訓練函式的 run-local output adapter，再以不訓練測試驗證 user-only 路徑。

本次不以 fixture 成功宣稱真實訓練可用，不建立新 candidate model，不做 production promotion。
