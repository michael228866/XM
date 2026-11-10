"""LSTM vs the frozen XGBoost H1 C: same bars, label, splits, threshold rule and baselines; only the model differs.

Two inputs: "full" = the 16 H1-lab features plus last-bar candle shape, "candles" = candle shape only
(learning straight from K-lines). Each row sees its last SEQ_LEN rows; every row holds only closed-bar data.

    python -B lstm_h1_v1.py --output forward_reports/20261108_lstm_h1_v1

Splits: fit < 2023 (last 15% of it for early stopping, purged by horizon), threshold pick 2023, test 2024+.
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import torch
from torch import nn

import h1_lab_v1 as lab
from strategy_lab_v1 import MIN_VALIDATION_TRADES, TOP_SHARES, first_touch, stats, trade

SPEC = lab.STRATEGIES["C_long_3.0_1.5_48h"]
CANDLE_FEATURES = ["C_BODY", "C_UPPER", "C_LOWER", "C_RET"]
INPUTS = {"full": lab.FEATURES + CANDLE_FEATURES, "candles": CANDLE_FEATURES}
SEQ_LEN, HIDDEN, EPOCHS, PATIENCE, BATCH, SEEDS = 48, 32, 15, 3, 512, (0, 1, 2)


def add_candles(frame):
    """Shape of the previous closed bar in ATR units (frame ATR is already the previous bar's)."""
    o, h, l, c = (frame[k].shift(1) for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
    atr = frame["ATR"]
    frame["C_BODY"] = (c - o) / atr
    frame["C_UPPER"] = (h - np.maximum(o, c)) / atr
    frame["C_LOWER"] = (np.minimum(o, c) - l) / atr
    frame["C_RET"] = (c - c.shift(1)) / atr
    return frame


class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.lstm = nn.LSTM(n_in, HIDDEN, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(HIDDEN, 1))

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1]).squeeze(-1)


def windows(x, idx):
    return torch.from_numpy(x[idx[:, None] + np.arange(-SEQ_LEN + 1, 1)])


def fit_predict(x, label, fit_idx, stop_idx, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(x.shape[1])
    pos = label[fit_idx].mean()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    y = torch.from_numpy(label.astype(np.float32))
    best, best_state, bad = np.inf, None, 0
    for epoch in range(EPOCHS):
        net.train()
        for b in np.array_split(rng.permutation(fit_idx), max(1, len(fit_idx) // BATCH)):
            opt.zero_grad()
            loss_fn(net(windows(x, b)), y[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(loss_fn(net(windows(x, stop_idx)), y[stop_idx]))
        print(f"  seed {seed} epoch {epoch} stop-loss {val:.4f}", flush=True)
        if val < best - 1e-4:
            best, best_state, bad = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
        elif (bad := bad + 1) >= PATIENCE:
            break
    net.load_state_dict(best_state)
    net.eval()
    scores = np.full(len(x), np.nan)
    all_idx = np.arange(SEQ_LEN - 1, len(x))
    with torch.no_grad():
        for b in np.array_split(all_idx, max(1, len(all_idx) // 4096)):
            scores[b] = torch.sigmoid(net(windows(x, b))).numpy()
    return scores


def evaluate(frame, scores, r, after):
    times = frame["TIME_DT"]
    val = ((times >= lab.VALIDATION_START) & (times < lab.TEST_START)).to_numpy()
    test = (times >= lab.TEST_START).to_numpy()
    picks = []
    for share in TOP_SHARES:
        threshold = float(np.quantile(scores[val], 1 - share))
        picks.append({"top_share": share, "threshold": round(threshold, 4),
                      **stats(trade(frame[val], scores[val], threshold, r[val], after[val])["r"])})
    eligible = [p for p in picks if p["n"] >= MIN_VALIDATION_TRADES]
    best = max(eligible or picks, key=lambda p: p.get("avg_r", -9))
    taken = trade(frame[test], scores[test], best["threshold"], r[test], after[test])
    return {"validation_candidates": picks, "chosen": best, "test": stats(taken["r"]),
            "baselines": lab.baselines(frame, scores, best["threshold"], SPEC, test)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    end = datetime.now(timezone.utc)
    if not mt5.initialize(path=os.environ.get("XM_TERMINAL_PATH", r"D:\XM2\terminal64.exe")):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        h1 = lab.copy_rates("H1", lab.START - timedelta(days=60), end)
        higher = {tf: lab.copy_rates(tf, lab.START - timedelta(days=900), end) for tf in lab.HIGHER}
    finally:
        mt5.shutdown()
    frame = add_candles(lab.make_features(h1, higher))
    frame = frame[frame["TIME_DT"] >= lab.START.replace(tzinfo=None)].reset_index(drop=True)
    r, after = first_touch(frame, SPEC["side"], SPEC["tp"], SPEC["sl"], SPEC["hold"])
    label = (r > 0).astype(np.int8)
    fit = np.flatnonzero((frame["TIME_DT"] < lab.VALIDATION_START).to_numpy() & np.isfinite(r))
    fit = fit[(fit >= SEQ_LEN - 1) & (fit < fit[-1] - SPEC["hold"])]  # purge labels reaching into validation
    cut = int(len(fit) * 0.85)
    fit_idx, stop_idx = fit[:cut - SPEC["hold"]], fit[cut:]  # purge between fit and early-stop rows too

    args.output.mkdir(parents=True)
    report = {"spec": SPEC, "seq_len": SEQ_LEN, "hidden": HIDDEN, "seeds": SEEDS,
              "xgb_reference": json.loads((Path(__file__).parent / "frozen" / "h1_C_long_v1" / "config.json")
                                          .read_text(encoding="utf-8"))["development_test"], "models": {}}
    for name, cols in INPUTS.items():
        print(f"=== {name} ({len(cols)} inputs)", flush=True)
        raw = frame[cols].to_numpy(np.float32)
        mean, std = np.nanmean(raw[fit_idx], axis=0), np.nanstd(raw[fit_idx], axis=0) + 1e-6
        x = np.nan_to_num(np.clip((raw - mean) / std, -5, 5)).astype(np.float32)
        scores = np.nanmean([fit_predict(x, label, fit_idx, stop_idx, s) for s in SEEDS], axis=0)
        scores = np.nan_to_num(scores, nan=0.0)
        report["models"][name] = result = evaluate(frame, scores, r, after)
        print(json.dumps({k: result[k] for k in ("chosen", "test", "baselines")}, ensure_ascii=False), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
