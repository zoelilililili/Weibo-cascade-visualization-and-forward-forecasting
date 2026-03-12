# -*- coding: utf-8 -*-
"""
JSON 微博转发树 -> 特征提取 -> 单回归（XGBoost）预测 meta_crawled_retweets（log1p）
并输出：
- dataset_features.csv
- predictions.csv
- feature_importance_single.csv
- 三张散点图（真实 vs 预测）
同时：
- 支持 Matplotlib 中文字体
- 防泄露：训练时默认丢掉 meta_total_retweets / meta_crawled_retweets / coverage（可配置）
- 新增特征：原微博发布时间（小时/星期/是否周末/是否夜间）+ 原微博作者粉丝数/认证（来自 root.user_profile）
"""

import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================
# 0) Matplotlib 中文字体设置
# =========================
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


def set_matplotlib_chinese_font():
    candidates = [
        "Microsoft YaHei", "微软雅黑",
        "SimHei", "黑体",
        "SimSun", "宋体",
        "STHeiti", "Heiti TC",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[Font] Using Chinese font: {name}")
            return
    print("[Font][Warn] No Chinese font found. Chinese may not render correctly.")
    plt.rcParams["axes.unicode_minus"] = False


set_matplotlib_chinese_font()

# =========================
# 1) XGBoost + sklearn
# =========================
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path


# =========================
# 2) 配置
# =========================
@dataclass
class Config:
    PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
    JSON_DIR = PROJECT_ROOT / "data" / "macro_data"
    OUT_DIR = PROJECT_ROOT / "models" / "macro_predictor" / "outputs"

    OBS_HOURS: float = 2.0
    RANDOM_SEED: int = 312
    TEST_SIZE: float = 0.2

    # 过滤：爬到的节点太少 -> 噪声巨大
    MIN_CRAWLED_NODES: int = 10

    # 是否保存中间特征表
    SAVE_FEATURE_CSV: bool = True

    # 可视化：高亮误差最大的样本数
    TOPK_HIGHLIGHT: int = 15

    # ========= 预测目标字段 =========
    # 这里固定预测 meta_crawled_retweets（如果某个样本没有，就丢弃）
    TARGET_COL_RAW: str = "meta_crawled_retweets"
    TARGET_COL_LOG: str = "log_meta_crawled_retweets"

    # ========== 防泄露：训练时删除这些列 ==========
    # 目标是 meta_crawled_retweets 时：
    # - meta_crawled_retweets 本身必须 drop（否则直接泄露 y）
    # - coverage=crawled/total 也会强泄露（尤其当 total 在特征里）
    # - meta_total_retweets 也会通过 coverage 间接泄露，所以默认也 drop
    DROP_LEAK_COLS: tuple = (
        "meta_total_retweets",
        "meta_crawled_retweets",
        "coverage",
        # 你如果还担心其他 meta 字段，也可以继续加
    )


CFG = Config()


# =========================
# 3) 时间解析
# =========================
def parse_weibo_created_at(s: str) -> Optional[float]:
    """
    Weibo created_at 例如：
    'Sun Dec 14 19:59:32 +0800 2025'
    返回 unix timestamp（秒）
    """
    from datetime import datetime
    try:
        dt = datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
        return dt.timestamp()
    except Exception:
        return None


# =========================
# 4) 基础统计工具
# =========================
def safe_log1p(x: float) -> float:
    return float(np.log1p(max(0.0, x)))


def entropy_from_counts(counts: List[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    p = np.array([c / total for c in counts if c > 0], dtype=float)
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log(p + 1e-12)).sum())


def gini_coefficient(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    if x.size == 0:
        return 0.0
    if np.allclose(x.sum(), 0):
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cumx = np.cumsum(x_sorted)
    return float(1.0 - 2.0 * (cumx.sum() / (n * cumx[-1])) + 1.0 / n)


def burstiness(intervals: np.ndarray) -> float:
    intervals = np.asarray(intervals, dtype=float)
    if intervals.size == 0:
        return 0.0
    mu = intervals.mean()
    sigma = intervals.std()
    denom = sigma + mu
    if denom <= 1e-12:
        return 0.0
    return float((sigma - mu) / denom)


def quantiles(arr: np.ndarray, qs=(0.1, 0.5, 0.9)) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return {f"q{int(q * 100)}": 0.0 for q in qs}
    vals = np.quantile(arr, qs)
    return {f"q{int(q * 100)}": float(v) for q, v in zip(qs, vals)}


# =========================
# 5) JSON -> 特征提取（并产出目标：meta_crawled_retweets）
# =========================
def extract_from_one_json(json_path: str, obs_hours: float) -> Optional[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    meta = obj.get("metadata", {}) if isinstance(obj.get("metadata", {}), dict) else {}
    nodes = obj.get("nodes", [])
    root = obj.get("root", {}) if isinstance(obj.get("root", {}), dict) else {}

    if not isinstance(nodes, list) or len(nodes) == 0:
        return None

    total_retweets = meta.get("total_retweets", None)
    crawled_retweets = meta.get("crawled_retweets", None)

    # ========= 目标：必须有 crawled_retweets =========
    if not isinstance(crawled_retweets, (int, float)):
        return None
    if crawled_retweets < CFG.MIN_CRAWLED_NODES:
        return None

    # ========= 新增：原微博作者 & 发布时间（来自 root） =========
    root_created_at = root.get("created_at")
    root_ts = parse_weibo_created_at(root_created_at) if root_created_at else None

    root_user = root.get("user_profile", {}) if isinstance(root.get("user_profile", {}), dict) else {}
    root_followers = root_user.get("followers_count", None)
    root_verified = root_user.get("verified", None)
    root_verified_type = root_user.get("verified_type", None)

    # root 时间 t0（传播起点）
    t0 = None
    if "created_at" in root:
        t0 = parse_weibo_created_at(root.get("created_at", ""))
    if t0 is None:
        # fallback：在 nodes 里找 is_root
        for n in nodes:
            if n.get("is_root") is True and "created_at" in n:
                t0 = parse_weibo_created_at(n.get("created_at", ""))
                break
    if t0 is None:
        return None

    # ========= 新增：发布时间节奏特征 =========
    from datetime import datetime
    dt0 = datetime.fromtimestamp(t0)
    pub_hour = dt0.hour
    pub_weekday = dt0.weekday()  # 0=Mon
    pub_is_weekend = int(pub_weekday >= 5)
    pub_is_night = int(pub_hour < 6 or pub_hour >= 22)

    # 解析前 obs_hours 的转发事件（排除 root）
    event_ts = []
    event_depth = []
    event_parent = []
    event_user_followers = []
    event_user_verified = []

    for n in nodes:
        if n.get("is_root") is True:
            continue
        ca = n.get("created_at")
        ts = parse_weibo_created_at(ca) if ca else None
        if ts is None:
            continue
        dt_hours = (ts - t0) / 3600.0
        if dt_hours < 0:
            continue
        if dt_hours <= obs_hours:
            event_ts.append(ts)
            event_depth.append(int(n.get("depth", 0)) if n.get("depth") is not None else 0)
            event_parent.append(str(n.get("parent_id")) if n.get("parent_id") is not None else "")

            uf = n.get("user_followers", None)
            if uf is None:
                u = n.get("user", {})
                if isinstance(u, dict):
                    uf = u.get("followers_count", None)
            event_user_followers.append(float(uf) if isinstance(uf, (int, float)) else np.nan)

            uv = n.get("user_verified", None)
            if uv is None:
                u = n.get("user", {})
                if isinstance(u, dict):
                    uv = u.get("verified", None)

            if isinstance(uv, bool):
                event_user_verified.append(int(uv))
            elif isinstance(uv, (int, float)):
                event_user_verified.append(int(uv != 0))
            else:
                event_user_verified.append(np.nan)

    if len(event_ts) < 2:
        return None

    # 排序
    order = np.argsort(event_ts)
    event_ts = np.array(event_ts)[order]
    event_depth = np.array(event_depth)[order]
    event_parent = np.array(event_parent)[order]
    followers = np.array(event_user_followers, dtype=float)[order]
    verified = np.array(event_user_verified, dtype=float)[order]

    t_rel_sec = event_ts - t0
    t_rel_hours = t_rel_sec / 3600.0

    intervals = np.diff(event_ts)
    q_int = quantiles(intervals, qs=(0.1, 0.5, 0.9))

    def time_to_k(k: int) -> float:
        if len(t_rel_hours) < k:
            return float(obs_hours)
        return float(t_rel_hours[k - 1])

    time_to_5 = time_to_k(5)
    time_to_10 = time_to_k(10)
    time_to_20 = time_to_k(20)

    # 分段增量：0~obs_hours 切 12 段
    bins = np.linspace(0, obs_hours, 13)
    counts_per_bin, _ = np.histogram(t_rel_hours, bins=bins)
    cum_per_bin = np.cumsum(counts_per_bin)

    # 树结构：窗口内 parent_id 出度
    parent_counts = {}
    for p in event_parent:
        if p == "" or str(p).lower() == "nan":
            continue
        parent_counts[p] = parent_counts.get(p, 0) + 1
    out_degrees = np.array(list(parent_counts.values()), dtype=float) if len(parent_counts) > 0 else np.array([0.0])
    max_out = float(out_degrees.max()) if out_degrees.size > 0 else 0.0
    top5_mean_out = float(np.mean(np.sort(out_degrees)[-5:])) if out_degrees.size >= 5 else float(out_degrees.mean())

    # 深度统计
    max_depth = int(event_depth.max()) if event_depth.size > 0 else 0
    mean_depth = float(event_depth.mean()) if event_depth.size > 0 else 0.0
    depth_counts = np.bincount(event_depth.astype(int)) if event_depth.size > 0 else np.array([0])
    depth_ent = entropy_from_counts(depth_counts.tolist())

    gini_out = gini_coefficient(out_degrees)
    hhi_out = float(np.sum((out_degrees / (out_degrees.sum() + 1e-12)) ** 2)) if out_degrees.sum() > 0 else 0.0

    # 用户影响力（转发者）
    followers_clean = followers.copy()
    followers_clean[np.isnan(followers_clean)] = 0.0
    log_followers = np.log1p(np.maximum(0.0, followers_clean))
    max_log_followers = float(log_followers.max())
    mean_log_followers = float(log_followers.mean())
    top5_log_followers_mean = float(np.mean(np.sort(log_followers)[-5:])) if log_followers.size >= 5 else mean_log_followers

    idx_max_fol = int(np.argmax(log_followers))
    time_of_max_follower = float(t_rel_hours[idx_max_fol])

    verified_clean = verified.copy()
    verified_clean[np.isnan(verified_clean)] = 0.0
    verified_ratio = float(verified_clean.mean()) if verified_clean.size > 0 else 0.0
    verified_early10 = float(verified_clean[: min(10, len(verified_clean))].sum())

    # coverage（作为 meta 字段保留在 dataset_features.csv；训练时默认会删掉）
    if isinstance(total_retweets, (int, float)) and isinstance(crawled_retweets, (int, float)) and total_retweets > 0:
        coverage = float(crawled_retweets / total_retweets)
    else:
        coverage = np.nan

    feat: Dict[str, Any] = {}
    feat["weibo_id"] = os.path.splitext(os.path.basename(json_path))[0].replace("retweet_tree_", "")
    feat["file"] = os.path.basename(json_path)

    # ========== 目标：crawled ==========
    feat["meta_total_retweets"] = float(total_retweets) if isinstance(total_retweets, (int, float)) else np.nan
    feat["meta_crawled_retweets"] = float(crawled_retweets)
    feat["coverage"] = coverage

    feat[CFG.TARGET_COL_RAW] = float(crawled_retweets)
    feat[CFG.TARGET_COL_LOG] = safe_log1p(float(crawled_retweets))

    # ========= 原微博作者特征 =========
    feat["root_log_followers"] = safe_log1p(float(root_followers) if isinstance(root_followers, (int, float)) else 0.0)
    if isinstance(root_verified, bool):
        feat["root_verified"] = int(root_verified)
    elif isinstance(root_verified, (int, float)):
        feat["root_verified"] = int(root_verified != 0)
    else:
        feat["root_verified"] = 0
    feat["root_verified_type"] = int(root_verified_type) if isinstance(root_verified_type, (int, float)) else -1

    # ========= 发布时间特征 =========
    feat["pub_hour"] = int(pub_hour)
    feat["pub_weekday"] = int(pub_weekday)
    feat["pub_is_weekend"] = int(pub_is_weekend)
    feat["pub_is_night"] = int(pub_is_night)

    # 观测窗口内规模
    feat["n_obs"] = int(len(event_ts))

    # 时间动力学
    feat["time_to_5h"] = time_to_5
    feat["time_to_10h"] = time_to_10
    feat["time_to_20h"] = time_to_20
    feat["interval_mean"] = float(intervals.mean()) if intervals.size > 0 else 0.0
    feat["interval_std"] = float(intervals.std()) if intervals.size > 0 else 0.0
    feat["interval_min"] = float(intervals.min()) if intervals.size > 0 else 0.0
    feat["interval_max"] = float(intervals.max()) if intervals.size > 0 else 0.0
    feat["interval_burstiness"] = burstiness(intervals)
    feat["interval_q10"] = q_int["q10"]
    feat["interval_q50"] = q_int["q50"]
    feat["interval_q90"] = q_int["q90"]

    # 12段增量/累积
    for i in range(12):
        feat[f"bin_inc_{i + 1}"] = int(counts_per_bin[i])
        feat[f"bin_cum_{i + 1}"] = int(cum_per_bin[i])

    # 树结构
    feat["max_depth"] = max_depth
    feat["mean_depth"] = mean_depth
    feat["depth_entropy"] = depth_ent
    feat["max_out_degree"] = max_out
    feat["top5_mean_out_degree"] = top5_mean_out
    feat["gini_out_degree"] = gini_out
    feat["hhi_out_degree"] = hhi_out

    # 转发者影响力
    feat["max_log_followers"] = max_log_followers
    feat["mean_log_followers"] = mean_log_followers
    feat["top5_mean_log_followers"] = top5_log_followers_mean
    feat["time_of_max_follower_h"] = time_of_max_follower
    feat["verified_ratio"] = verified_ratio
    feat["verified_early10"] = verified_early10

    return feat


def load_dataset_from_json_dir(json_dir: str, obs_hours: float) -> pd.DataFrame:
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    print(f"[Info] Found {len(json_files)} json files in: {json_dir}")

    rows = []
    skipped = 0
    for jf in json_files:
        path = os.path.join(json_dir, jf)
        try:
            feat = extract_from_one_json(path, obs_hours=obs_hours)
            if feat is None:
                skipped += 1
                continue
            rows.append(feat)
        except Exception:
            skipped += 1
            continue

    df = pd.DataFrame(rows)
    print(f"[Info] Loaded {len(df)} samples, skipped {skipped}.")
    return df


# =========================
# 6) 训练：单回归模型
# =========================
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_xgb_regressor(X_tr, y_tr, X_val, y_val, seed=42) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )

    # 兼容不同 xgboost 版本 early stopping
    try:
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=120,
        )
    except TypeError:
        try:
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
                callbacks=[xgb.callback.EarlyStopping(rounds=120, save_best=True)],
            )
        except TypeError:
            model.fit(X_tr, y_tr)

    return model


def feature_importance_to_csv(model, feature_names: List[str], out_path: str):
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    df = df.sort_values("importance", ascending=False)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


# =========================
# 7) 可视化：真实 vs 预测
# =========================
def plot_true_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: str,
    title_suffix: str = "",
    topk_highlight: int = 0,
    extra_labels: Optional[List[str]] = None,
):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    y_pred = np.maximum(y_pred, 0.0)
    y_true = np.maximum(y_true, 0.0)

    # 1) 线性坐标
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.6, s=25)
    max_val = float(max(y_true.max(), y_pred.max(), 1.0))
    plt.plot([0, max_val], [0, max_val], "r--", lw=2, label="y = x")
    plt.xlabel("真实 crawled_retweets")
    plt.ylabel("预测 crawled_retweets")
    plt.title(f"真实值 vs 预测值（线性）{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(out_dir, "scatter_true_vs_pred_linear.png")
    plt.savefig(p1, dpi=160)
    plt.show()

    # 2) log-log
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true + 1.0, y_pred + 1.0, alpha=0.6, s=25)
    max_val2 = float(max((y_true + 1.0).max(), (y_pred + 1.0).max(), 2.0))
    plt.plot([1, max_val2], [1, max_val2], "r--", lw=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("真实 crawled_retweets（log）")
    plt.ylabel("预测 crawled_retweets（log）")
    plt.title(f"真实值 vs 预测值（log-log）{title_suffix}")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "scatter_true_vs_pred_loglog.png")
    plt.savefig(p2, dpi=160)
    plt.show()

    # 3) TopK 误差高亮
    if topk_highlight and topk_highlight > 0:
        abs_err = np.abs(y_pred - y_true)
        k = min(topk_highlight, len(abs_err))
        idx = np.argsort(abs_err)[-k:]

        plt.figure(figsize=(7, 7))
        plt.scatter(y_true, y_pred, alpha=0.35, s=20, label="其他样本")
        plt.scatter(y_true[idx], y_pred[idx], alpha=0.9, s=60, label=f"误差最大 Top{k}")
        max_val = float(max(y_true.max(), y_pred.max(), 1.0))
        plt.plot([0, max_val], [0, max_val], "k--", lw=2)

        if extra_labels is not None and len(extra_labels) == len(y_true):
            for i in idx:
                plt.annotate(
                    extra_labels[i],
                    (y_true[i], y_pred[i]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    alpha=0.85
                )

        plt.xlabel("真实 crawled_retweets")
        plt.ylabel("预测 crawled_retweets")
        plt.title(f"高亮误差最大样本{title_suffix}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        p3 = os.path.join(out_dir, "scatter_topk_error_highlight.png")
        plt.savefig(p3, dpi=160)
        plt.show()

    print(f"[Saved figures]\n  {p1}\n  {p2}")
    if topk_highlight and topk_highlight > 0:
        print(f"  {os.path.join(out_dir, 'scatter_topk_error_highlight.png')}")


# =========================
# 8) main
# =========================
def main():
    os.makedirs(CFG.OUT_DIR, exist_ok=True)

    # 1) 读 JSON -> 特征表
    df = load_dataset_from_json_dir(CFG.JSON_DIR, CFG.OBS_HOURS)
    if df.empty:
        raise RuntimeError("No valid samples were loaded. Check JSON format / OBS window / skip rules.")

    # 保存特征表
    if CFG.SAVE_FEATURE_CSV:
        feat_path = os.path.join(CFG.OUT_DIR, "dataset_features.csv")
        df.to_csv(feat_path, index=False, encoding="utf-8-sig")
        print(f"[Saved] {feat_path}")

    # 2) 准备 X / y（目标改为 meta_crawled_retweets）
    id_cols = ["weibo_id", "file"]

    # y：log1p(crawled_retweets)
    y = df[CFG.TARGET_COL_LOG].values.astype(float)

    # X：把 id + 目标列（raw/log）都删掉
    X = df.drop(
        columns=id_cols + [
            CFG.TARGET_COL_RAW, CFG.TARGET_COL_LOG,
            "meta_total_retweets", "meta_crawled_retweets", "coverage",
        ],
        errors="ignore"
    )
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # ========== 再保险：删除泄露特征 ==========
    drop_cols = [c for c in CFG.DROP_LEAK_COLS if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols, errors="ignore")
        print(f"[LeakGuard] Dropped columns: {drop_cols}")
    else:
        print("[LeakGuard] No leak columns were found in X (good).")

    feature_names = X.columns.tolist()

    print("\n" + "=" * 70)
    print("[Info] Dataset summary (Target = meta_crawled_retweets)")
    print("=" * 70)
    print(f"Samples: {len(df)}")
    print(f"OBS window: {CFG.OBS_HOURS} hours")
    print("Target(crawled_retweets) stats:",
          f"min={df[CFG.TARGET_COL_RAW].min():.1f}",
          f"median={df[CFG.TARGET_COL_RAW].median():.1f}",
          f"max={df[CFG.TARGET_COL_RAW].max():.1f}")

    # 3) train/test split
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=CFG.TEST_SIZE, random_state=CFG.RANDOM_SEED, shuffle=True
    )

    # 4) 再切 val（从训练集里切）
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train.values, y_train,
        test_size=0.2, random_state=CFG.RANDOM_SEED, shuffle=True
    )

    # 5) 训练单回归器
    reg = train_xgb_regressor(X_tr, y_tr, X_val, y_val, seed=CFG.RANDOM_SEED)

    # 6) 预测 + 评估（log尺度）
    yhat_log = reg.predict(X_test.values)
    y_true_log = y_test

    print("\n" + "=" * 70)
    print("[Single Regressor] Target = log1p(crawled_retweets)")
    print("=" * 70)
    print(f"RMSE(log): {rmse(y_true_log, yhat_log):.4f}")
    print(f"MAE(log):  {mean_absolute_error(y_true_log, yhat_log):.4f}")
    print(f"R2(log):   {r2_score(y_true_log, yhat_log):.4f}")

    # 7) 反变换到原尺度评估
    yhat_raw = np.expm1(yhat_log)
    ytrue_raw = df_test[CFG.TARGET_COL_RAW].values.astype(float)

    print("\n" + "=" * 70)
    print("[Final scale] expm1 back-transform (crawled_retweets)")
    print("=" * 70)
    print(f"RMSE: {rmse(ytrue_raw, yhat_raw):.4f}")
    print(f"MAE:  {mean_absolute_error(ytrue_raw, yhat_raw):.4f}")
    print(f"R2:   {r2_score(ytrue_raw, yhat_raw):.4f}")

    # 8) 可视化散点图
    labels_for_annot = (df_test["file"].astype(str).tolist()
                        if "file" in df_test.columns else None)

    plot_true_vs_pred(
        y_true=ytrue_raw,
        y_pred=yhat_raw,
        out_dir=CFG.OUT_DIR,
        title_suffix=f"\n(OBS={CFG.OBS_HOURS}h, target=crawled_retweets)",
        topk_highlight=CFG.TOPK_HIGHLIGHT,
        extra_labels=labels_for_annot
    )

    # 9) 输出预测结果表
    pred_df = df_test[[
        "weibo_id", "file",
        "meta_total_retweets", "meta_crawled_retweets", "coverage",
        "n_obs"
    ]].copy()

    pred_df["true_crawled_retweets"] = ytrue_raw
    pred_df["pred_log_crawled"] = yhat_log
    pred_df["pred_crawled_retweets"] = yhat_raw
    pred_df["abs_err"] = np.abs(pred_df["pred_crawled_retweets"] - pred_df["true_crawled_retweets"])
    pred_df["rel_err_pct"] = pred_df["abs_err"] / (pred_df["true_crawled_retweets"] + 1.0) * 100.0
    pred_df = pred_df.sort_values("abs_err", ascending=False)

    pred_path = os.path.join(CFG.OUT_DIR, "predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"\n[Saved] {pred_path}")

    # 10) 保存特征重要性
    fi_path = os.path.join(CFG.OUT_DIR, "feature_importance_single.csv")
    feature_importance_to_csv(reg, feature_names, fi_path)
    print(f"[Saved] {fi_path}")

    # 11) 保存模型包
    bundle = {
        "config": CFG.__dict__,
        "feature_names": feature_names,
        "mode": "single",
        "target_raw": CFG.TARGET_COL_RAW,
        "target_log": CFG.TARGET_COL_LOG,
        "regressor": reg,
    }
    model_path = os.path.join(CFG.OUT_DIR, "model_bundle.pkl")
    joblib.dump(bundle, model_path)
    print(f"[Saved] {model_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
