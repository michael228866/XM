import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import gymnasium as gym
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

DATA_DIR = r"D:\XM\數據"
INITIAL_BALANCE = 10000
SPREAD_POINTS = 30
LABEL_LOOKAHEAD = 90
TARGET_ATR_MULT = 4.0
EXPERT_TRAIN_END_RATIO = 0.70
RL_TRAIN_END_RATIO = 0.85
MAX_HOLD_TICKS = 240
STOP_LOSS_PRICE = -0.2
TAKE_PROFIT_PRICE = None
PPO_MODEL_PATH = "gold_ppo_commander"
CONTINUE_EXISTING_PPO = False
ADDITIONAL_TIMESTEPS = 1_000_000
ENTRY_SIGNAL_THRESHOLD = 0.40
ENTRY_EDGE_THRESHOLD = 0.03
IDLE_PENALTY = 0.001
MISSED_SIGNAL_PENALTY = 0.20
ENTRY_BONUS = 0.15
TARGET_ENTRY_BONUS = 0.12
TARGET_WAIT_PENALTY = 0.08
TARGET_WRONG_PENALTY = 0.12
WRONG_DIRECTION_PENALTY = 0.08
INVALID_ACTION_PENALTY = 0.03
OPEN_ACTION_COST = 0.002
PRICE_CHANGE_REWARD_SCALE = 1.0
HOLD_TIME_PENALTY = 0.001
HOLD_TIME_POWER = 0.7

BASE_FEATURES = [
    "M1_RSI",
    "ATR",
    "MACD_HIST",
    "BB_WIDTH",
    "BIAS_20",
    "BODY_PCT",
    "ROC_5",
    "VOLA_RATIO",
    "HOUR_SIN",
    "HOUR_COS",
    "DAY_OF_WEEK",
]


def add_indicators(df):
    df = df.copy()
    high_low = df["HIGH"] - df["LOW"]
    tr = pd.concat(
        [
            high_low,
            np.abs(df["HIGH"] - df["CLOSE"].shift()),
            np.abs(df["LOW"] - df["CLOSE"].shift()),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["HOUR_SIN"] = np.sin(2 * np.pi * df["TIME_DT"].dt.hour / 24)
    df["HOUR_COS"] = np.cos(2 * np.pi * df["TIME_DT"].dt.hour / 24)
    df["DAY_OF_WEEK"] = df["TIME_DT"].dt.dayofweek / 7.0

    ema12 = df["CLOSE"].ewm(span=12, adjust=False).mean()
    ema26 = df["CLOSE"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["MACD_HIST"] = macd - macd.ewm(span=9, adjust=False).mean()

    ma20 = df["CLOSE"].rolling(20).mean()
    df["BB_WIDTH"] = (df["CLOSE"].rolling(20).std() * 4) / (ma20 + 1e-6)
    df["BIAS_20"] = (df["CLOSE"] - ma20) / (ma20 + 1e-6)
    df["ROC_5"] = df["CLOSE"].pct_change(5)
    candle_range = df["HIGH"] - df["LOW"] + 1e-6
    df["BODY_PCT"] = np.abs(df["CLOSE"] - df["OPEN"]) / candle_range
    return df


def load_and_prepare_data():
    all_files = sorted(
        f for f in os.listdir(DATA_DIR) if f.startswith("GOLD#_") and f.endswith(".csv")
    )
    data_dict = {}

    for filename in all_files:
        tf_name = filename.split("_")[1]
        path = os.path.join(DATA_DIR, filename)
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception as exc:
            print(f"Skipped {filename}: {exc}")
            continue

        df.columns = [c.replace("<", "").replace(">", "").upper() for c in df.columns]
        if "DATE" in df.columns and "TIME" in df.columns:
            df["TIME_DT"] = pd.to_datetime(df["DATE"] + " " + df["TIME"])
        elif "DATE" in df.columns:
            df["TIME_DT"] = pd.to_datetime(df["DATE"])
        else:
            continue

        data_dict[tf_name] = df.sort_values("TIME_DT")
        print(f"Loaded {tf_name}")

    if "M1" not in data_dict:
        raise FileNotFoundError("M1 data file was not found.")

    m1 = add_indicators(data_dict["M1"])
    diff = m1["CLOSE"].diff()
    gain = diff.where(diff > 0, 0).rolling(14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(14).mean()
    m1["M1_RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-6))))
    m1["VOLA_MA"] = m1["ATR"].rolling(240).mean()
    m1["VOLA_RATIO"] = m1["ATR"] / (m1["VOLA_MA"] + 1e-6)

    df = m1.sort_values("TIME_DT")
    mtf_features = []
    for tf_name, tdf in data_dict.items():
        if tf_name == "M1":
            continue
        tdf = add_indicators(tdf)
        trend_col = f"{tf_name}_TREND"
        tdf[trend_col] = np.where(tdf["CLOSE"] > tdf["CLOSE"].rolling(20).mean(), 1, -1)
        tdf[trend_col] = tdf[trend_col].shift(1)
        df = pd.merge_asof(df, tdf[["TIME_DT", trend_col]], on="TIME_DT")
        mtf_features.append(trend_col)

    full_features = BASE_FEATURES + mtf_features
    df[full_features] = df[full_features].shift(1)

    prices = df["CLOSE"].values
    atrs = df["ATR"].values
    target = np.zeros(len(df))
    for offset in range(1, LABEL_LOOKAHEAD + 1):
        future = df["CLOSE"].shift(-offset).values
        mask = target == 0
        target[mask & (future - prices > (TARGET_ATR_MULT * atrs))] = 1
        target[mask & (future - prices < (-TARGET_ATR_MULT * atrs))] = 2
    df["TARGET"] = target.astype(int)

    return df.dropna(subset=full_features + ["TARGET"]), full_features


class Experts:
    def __init__(self, features, device="cuda"):
        self.features = features
        self.device = device
        xgb_device = "cuda" if device == "cuda" else "cpu"
        lgb_device = "gpu" if device == "cuda" else "cpu"
        self.xgb = xgb.XGBClassifier(
            tree_method="hist",
            device=xgb_device,
            n_estimators=500,
            random_state=42,
        )
        self.lgb = lgb.LGBMClassifier(
            device=lgb_device,
            n_estimators=500,
            random_state=42,
        )

    @staticmethod
    def _align_probs(model, probs):
        aligned = np.zeros((len(probs), 3), dtype=np.float32)
        for idx, cls in enumerate(model.classes_):
            aligned[:, int(cls)] = probs[:, idx]
        return aligned

    def train(self, df):
        print(f"Training Experts on {self.device.upper()}...")
        counts = df["TARGET"].value_counts()
        class_weights = {
            cls: len(df) / (len(counts) * count)
            for cls, count in counts.items()
        }
        sample_weight = df["TARGET"].map(class_weights).to_numpy()
        self.xgb.fit(df[self.features], df["TARGET"], sample_weight=sample_weight)
        self.lgb.fit(df[self.features], df["TARGET"], sample_weight=sample_weight)

    def predict_probs(self, df):
        p_xgb = self._align_probs(self.xgb, self.xgb.predict_proba(df[self.features]))
        p_lgb = self._align_probs(self.lgb, self.lgb.predict_proba(df[self.features]))
        return np.hstack([p_xgb, p_lgb])

    def save(self, prefix):
        self.xgb.save_model(f"{prefix}_xgb.json")
        self.lgb.booster_.save_model(f"{prefix}_lgb.txt")
        print(f"Experts saved as {prefix}_xgb.json and {prefix}_lgb.txt")


class GoldDRLEnv(gym.Env):
    def __init__(self, df, expert_probs):
        super().__init__()
        self.df = df
        self.expert_probs = expert_probs
        self.max_hold_ticks = MAX_HOLD_TICKS
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = INITIAL_BALANCE
        self.position = 0
        self.entry_price = 0
        self.hold_ticks = 0
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        probs = self.expert_probs[self.current_step].astype(np.float32)
        curr_price = self.df.iloc[self.current_step]["CLOSE"]
        floating_return = (
            (curr_price - self.entry_price) / (self.entry_price + 1e-6)
            if self.position != 0
            else 0
        )
        state = np.array(
            [
                self.position / 2.0,
                floating_return,
                min(self.balance / INITIAL_BALANCE, 2.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate([probs, state])

    def _expert_signal(self):
        probs = self.expert_probs[self.current_step]
        buy_prob = float((probs[1] + probs[4]) / 2.0)
        sell_prob = float((probs[2] + probs[5]) / 2.0)
        edge = abs(buy_prob - sell_prob)
        if buy_prob >= sell_prob:
            return 1, buy_prob, edge
        return 2, sell_prob, edge

    def _close_position(self, curr_price):
        pnl = (
            curr_price - self.entry_price
            if self.position == 1
            else self.entry_price - curr_price
        )
        trade_reward = (pnl * 100) - (SPREAD_POINTS * 0.01 * 100)
        self.balance += trade_reward
        self.position = 0
        self.entry_price = 0
        self.hold_ticks = 0
        return trade_reward

    def step(self, action):
        curr_price = self.df.iloc[self.current_step]["CLOSE"]
        reward = 0.0
        signal_action, signal_conf, signal_edge = self._expert_signal()
        target_action = int(self.df.iloc[self.current_step].get("TARGET", 0))

        if self.position == 0:
            if action == 0:
                reward -= IDLE_PENALTY
                if signal_conf >= ENTRY_SIGNAL_THRESHOLD and signal_edge >= ENTRY_EDGE_THRESHOLD:
                    reward -= MISSED_SIGNAL_PENALTY * signal_conf
                if target_action in (1, 2):
                    reward -= TARGET_WAIT_PENALTY
            elif action in (1, 2):
                self.position = action
                self.entry_price = curr_price
                self.hold_ticks = 0
                reward -= OPEN_ACTION_COST
                if signal_conf >= ENTRY_SIGNAL_THRESHOLD and signal_edge >= ENTRY_EDGE_THRESHOLD:
                    if action == signal_action:
                        reward += ENTRY_BONUS * signal_conf
                    else:
                        reward -= WRONG_DIRECTION_PENALTY * signal_conf
                if target_action in (1, 2):
                    if action == target_action:
                        reward += TARGET_ENTRY_BONUS
                    else:
                        reward -= TARGET_WRONG_PENALTY
                else:
                    reward -= OPEN_ACTION_COST
        else:
            if action == 0:
                trade_reward = self._close_position(curr_price)
                reward += trade_reward / 10.0
            elif action != self.position:
                reward -= INVALID_ACTION_PENALTY

        if self.position != 0:
            self.hold_ticks += 1
            prev_price = (
                curr_price
                if self.current_step == 0
                else self.df.iloc[self.current_step - 1]["CLOSE"]
            )
            price_change = (
                curr_price - prev_price
                if self.position == 1
                else prev_price - curr_price
            )
            reward += price_change * PRICE_CHANGE_REWARD_SCALE
            reward -= HOLD_TIME_PENALTY * (self.hold_ticks ** HOLD_TIME_POWER)

            float_pnl = (
                curr_price - self.entry_price
                if self.position == 1
                else self.entry_price - curr_price
            )
            if (
                self.hold_ticks > self.max_hold_ticks
                or float_pnl < STOP_LOSS_PRICE
                or (TAKE_PROFIT_PRICE is not None and float_pnl > TAKE_PROFIT_PRICE)
            ):
                trade_reward = self._close_position(curr_price)
                reward += (trade_reward / 10.0) - 1.0

        self.current_step += 1
        done = (self.current_step >= len(self.df) - 1) or (
            self.balance <= INITIAL_BALANCE * 0.7
        )

        return self._get_obs(), reward, done, False, {}


def check_gpu():
    print("Checking GPU Status...")
    if not torch.cuda.is_available():
        print("Torch CUDA is not available. Falling back to CPU.")
        print(
            "Recommended install: python -m pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu128"
        )
        return "cpu"
    try:
        test = torch.randn(32, 32, device="cuda")
        _ = test @ test
        torch.cuda.synchronize()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    except RuntimeError as exc:
        print(f"CUDA is visible but failed a tensor test: {exc}")
        return "cpu"


if __name__ == "__main__":
    train_device = check_gpu()

    print("Preparing Data & Engineering Features...")
    df, full_features = load_and_prepare_data()

    expert_split = int(len(df) * EXPERT_TRAIN_END_RATIO)
    rl_split_end = int(len(df) * RL_TRAIN_END_RATIO)

    train_experts_df = df.iloc[: max(0, expert_split - LABEL_LOOKAHEAD)].copy()
    rl_train_df = df.iloc[expert_split:rl_split_end].copy().reset_index(drop=True)
    print(
        f"Expert rows: {len(train_experts_df):,} | "
        f"RL rows: {len(rl_train_df):,} | "
        f"Test reserved: {len(df) - rl_split_end:,}"
    )

    exp = Experts(full_features, device=train_device)
    exp.train(train_experts_df)
    exp.save("gold_mtf")

    print("Generating Expert Signals for RL Agent...")
    expert_signals = exp.predict_probs(rl_train_df)

    print(f"Training DRL Agent (PPO) - Sniper Mode ({train_device.upper()} Mode)...")
    env = DummyVecEnv([lambda: GoldDRLEnv(rl_train_df, expert_signals)])
    if CONTINUE_EXISTING_PPO and os.path.exists(f"{PPO_MODEL_PATH}.zip"):
        print(f"Continuing PPO training from {PPO_MODEL_PATH}.zip...")
        model = PPO.load(PPO_MODEL_PATH, env=env, device=train_device)
        reset_num_timesteps = False
    else:
        print("Starting a new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=train_device,
            learning_rate=2e-4,
            n_steps=4096,
            batch_size=128,
            ent_coef=0.01,
        )
        reset_num_timesteps = True

    print(f"Sniper Training Started. Additional steps: {ADDITIONAL_TIMESTEPS:,}")
    model.learn(total_timesteps=ADDITIONAL_TIMESTEPS, reset_num_timesteps=reset_num_timesteps)

    model.save(PPO_MODEL_PATH)
    print("Sniper Model (Experts + Commander) Saved!")
