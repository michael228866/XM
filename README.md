# GOLD# 模型研究與 MT5 Forward Test

本專案包含 GOLD# 交易模型、完整歷史 walk-forward 研究、候選模型晉級報告，以及 MT5 測試帳戶的 forward-test runner。

> `gemini.py` 目前 `DRY_RUN = False`，會實際送出委託。啟動前務必確認 MT5 連線的是測試帳戶。

## 目前 Forward-Test 版本

以本機 `gemini.py` 的實際設定為準：

- Runner：`gemini.py`
- Model：`gold_long_recent_candidate_xgb.json`
- Model output：`long_binary`
- Meta overlay：關閉
- 方向：目前僅做多
- 進場門檻：`0.75`
- 每筆風險：`1.4%`
- TP／SL：`1.3 ATR / 1.6 ATR`
- 最長持倉：`90` 分鐘
- 最大同時持倉：`1`
- 時區：交易與日誌時間使用 UTC

2026-05-18 至 2026-08-28 的最近一次可比較回測：

| Trades | Win rate | PF | PnL | Max DD |
|---:|---:|---:|---:|---:|
| 20 | 60.00% | 1.08 | +8.84 | -5.50% |

## 模型晉級規則

新模型一律先保留為 `research_only`，不因為近期勝率或交易數較高就替換 `gemini.py`。

晉級必須同時通過：

1. 使用 2014–2026 完整歷史資料，按時間 walk-forward，禁止隨機混合切割。
2. 所有歷史選模區間、獨立 holdout 及近期資料都必須合格。
3. 同時比較交易數、勝率、PF、PnL 與最大回撤；零交易候選不得勝出。
4. 標籤與回測使用 TP／SL 第一觸價，並以盤中 `HIGH` / `LOW` 決定先觸價方向。
5. 加入成本壓力後仍需為正 PnL，且 PF 不得低於 `1.05`。
6. 通過研究門檻後仍先放入測試帳戶 forward test，才能取代現行模型。

## 近期迭代結果

下表為各代在 2026 近期資料的結果；晉級結果仍以全部歷史區間為準。

| 版本 | Trades | Win rate | PF | PnL | 晉級 |
|---|---:|---:|---:|---:|---|
| Generation 8 M5 residual | 43 | 62.79% | 1.17 | +37.58 | FAIL |
| Generation 10 adaptive portfolio | 10 | 30.00% | 0.29 | -64.15 | FAIL |
| Generation 11 execution-aligned | 9 | 22.22% | 0.10 | -86.31 | FAIL |
| Generation 12 executable events | 51 | 60.78% | 0.93 | -15.34 | FAIL |
| Generation 13 directional exits | 60 | 46.67% | 0.71 | -70.29 | FAIL |

Generation 12 成功提高交易頻率與近期勝率，但 PF 低於 1 且 PnL 為負。Generation 13 的對稱停損又明顯降低勝率。兩者都只保留作為研究資料，沒有替換現行 runner。

完整數字請查看：

- `gold_generation8_residual_walk_forward.md`
- `gold_generation10_adaptive_portfolio.md`
- `gold_generation11_execution_aligned.md`
- `gold_generation12_executable_events.md`
- `gold_generation13_directional_exits.md`

## 主要命令

環境建議使用 Python 3.10。Windows 上若 `python` 不在 PATH，可使用 `py -3.10`。

啟動 MT5 forward test：

```powershell
py -3.10 gemini.py
```

執行第十三代快速自我檢查：

```powershell
py -3.10 gold_generation13_directional_exits.py --self-check
```

使用完整歷史資料重新訓練第十三代：

```powershell
py -3.10 gold_generation13_directional_exits.py
```

完整訓練需要：

- 本機歷史 GOLD# CSV 資料。
- `D:\XM2\terminal64.exe` 可連線 MT5，以取得近期保留期資料。
- `MetaTrader5`、`numpy`、`pandas`、`scikit-learn` 與 `xgboost`。

## 主要檔案

- `gemini.py`：MT5 forward-test runner。
- `barrier_classifier_strategy.py`：第一觸價標籤與盤中 HIGH／LOW 回測。
- `gold_generation12_executable_events.py`：非重疊可執行事件 Expected-R 模型。
- `gold_generation13_directional_exits.py`：方向別 rolling TP／SL 冠軍研究。
- `gold_generation13_candidate.json`：第十三代候選設定與 `research_only` 狀態。
- `gold_generation13_directional_exits.json`：完整 walk-forward 與近期評估。

## 日誌與資料政策

`gemini.py` 會在本機產生：

- `gemini_signal_log.csv`
- `gemini_trade_history.csv`

這些檔案可能包含帳戶執行細節，而且會持續增長，因此不納入 Git。大型 `GOLD#_*.csv` 歷史資料也只保留在本機；如需共用，請改用 Git LFS 或獨立 artifact storage。
