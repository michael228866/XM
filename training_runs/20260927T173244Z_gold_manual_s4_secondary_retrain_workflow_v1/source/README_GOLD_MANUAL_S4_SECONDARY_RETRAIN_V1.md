# GOLD S4 Secondary Retrain v1

本版本是使用者明確指定的 S4 secondary 歷史重現流程。認證 PASS 並提交 approval 後，雙擊 `RUN_TRAINING.bat` 即可；不需輸入命令或參數。
本次實作／認證不訓練真實模型。合成測試使用 fake classifier，不呼叫 XGBoost.fit。

## 固定定義

- 來源：20260915 discovery 的實際訓練函式；20260919 confirmation 僅提供相同三個 model hash 的佐證，不當 trainer。
- GOLD#，不是 GAUCNH#。source ID `LEGACY_XM_GOLD_S4_20260915` 表示原始檔案身分；不把舊檔案的 broker server 推定為目前連線 server。
- 完整 31 欄及順序、float32 特徵、int8 C1 標籤、參數、seed=42、CPU single-thread、220 trees 全部固定。
- 三個 score folds：2018–2020、2021–2022、2023–2024；各自前 18 個月訓練，沿用 C0/C1 嚴格成熟度 purge。
- Secondary training subset：對應 archived B0 的 TRAIN probability < 0.75。B0 僅載入及預測，絕不重訓。
- C1 target：standalone S5 net realized R > 0 為 1，否則 0。
- Gate：B0 >= 0.75，或 B0 < 0.75 且 secondary >= 0.75。沿用 frozen S5，無 threshold tuning。
- 輸出三個 fold specialist。最後一個模型只是最後歷史 fold 的身分，不是用最新日期 full-fit 的 production model。

## 資料裁決

21 個 CSV 的原始 hash 在 C1 finalized manifest 中記錄；S4 discovery 使用相同 frozen X/y/timestamp 重建。
本次確認每個原始 hash，並逐列檢查 schema、排序、重複、OHLC、volume/spread 和首末日期。
這是 `EXACT_PROVENANCE_TRAINING_SOURCE`／`REPRODUCTION_DATASET`，不需要為外觀一致而重新抓資料。

固定 raw cutoff：`2026-05-08T23:57:00`，原始 broker/export naive clock。
最早 raw context 為 `2014-01-01`；實際 fold training 自 `2016-07-01`，最後 train interval 在 `2023-01-01` 前。
歷史 metrics 僅 score 2018–2024。原始 bytes 在 holdout 啟動前已被 finalized run 記錄。
不套用目前 capture 的 10800 offset，不聲稱解決整段歷史的 UTC 映射。

使用者啟動時，建立 content-addressed 獨立歷史 cache，只放這 21 個精確檔案，避免 glob 掃入其他 CSV。
若原檔與 cache 皆缺少，先找 import 中精確 hash 相符檔案，再自動用 native MT5 `copy_rates_range` 嘗試還原。
API query 使用原始 wall-clock 座標，不推定 UTC 等價性；回傳值必須序列化為原始 schema 並通過原始 SHA256 才能使用。
不同 bytes 屬 `NEW_RETRAIN_DATASET`，本版本拒絕。歷史 bars 可能無法從 broker 完整重取；此時顯示缺少檔名與原因並停止，不會靜默換 symbol/timeframe。
正常情況已有全部原檔，不需使用者匯出 CSV。原始資料與 cache 保留在本機，未聲稱存在遠端 raw-data 備份。

## 手動流程與封存

Explorer → BAT → Python launcher → process-local、一次性、30 秒憑證 → approval 與 source/environment 檢查 → exact dataset cache → 新 RUN_ID → secondary fits → 歷史 S5 metrics → 獨立 validator → immutable finalize/register → 窄範圍 Git archive commit/push。

Validator 不 import trainer、不再次 fit，使用原本獨立驗證過的 exact reproduction oracle：
固定 dataset/source/model hashes、每個 fold 的 conditioning/evidence arrays、完整 ledger、historical metrics，逐一核對。
不同結果會 FAIL，不會據此調參。Methodology PASS 不等於經濟效益或 promotion PASS。
NPZ evidence、source/config/metrics/ledger/model JSON/seal 皆封存並提交 Git；大型 raw cache 本機保留且記錄 SHA。
失敗與取消也保留 aborted run；若網路／Git 失敗，顯示錯誤並保留封存，不刪除或重試 validator。
OS 強制關閉無法保證執行 finally；遺留 run 不可重用。

Python audit hook 拒絕 locked holdout 路徑、外部程序與 run 外寫入；只在已審核的 fitting/evaluation 區段啟用。
Native libraries 在進入該區段前載入。這不是對惡意 native code 的 OS sandbox。
XGBoost 輸出經 adapter 限定於當次 run/models，受保護 production hashes 前後與 validator 再驗證。

永不自動 promotion，不修改 gemini.py 或 operational model。Capture code/policy/task 不在本次修改範圍。
CHECK_STATUS 分別顯示 capture、holdout boundary 與 training readiness；不顯示 locked holdout performance。
