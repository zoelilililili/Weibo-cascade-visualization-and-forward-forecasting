# -*- coding: utf-8 -*-
"""
直接对 retweet_tree_*.json 绘制传播曲线（不依赖 processed CSV）：
- 累计转发数随时间变化：总体 + 按 depth 分组
- 信息热度曲线：R/U/D/Z/S 五指标线性组合（z-score 标准化）
- 累计传播地区数-时间曲线

"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 路径配置
# =========================
BASE = Path(__file__).parent
tree_dir = BASE / "retweet_trees_for_visualization1"

# 你示例里 out_dir 用于 sunburst，这里我单独建一个目录放曲线图
out_dir = BASE / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# 时间窗口大小
TIME_BIN = "10min"

# 主传播期截断：累计转发达到 q 的时刻 + 缓冲
MAIN_Q = 0.95
RIGHT_PAD_HOURS = 2

# 中文字体（按需保留）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 信息热度五项指标权重（可调）
WEIGHTS = {
    "R": 0.40,  # 转发量
    "U": 0.25,  # 独立用户数
    "D": 0.15,  # 平均深度
    "Z": 0.10,  # 地区扩散
    "S": 0.10   # 深层次转发比例
}


def _extract_region(x: str) -> str:
    """把 '发布于 云南' -> '云南'；空/缺失返回 ''"""
    if not isinstance(x, str):
        return ""
    x = x.strip()
    if not x:
        return ""
    m = re.search(r"发布于\s*(.+)$", x)
    return (m.group(1).strip() if m else x)


def _parse_created_at(series: pd.Series) -> pd.Series:
    """
    解析微博 created_at: 'Tue Nov 04 07:49:15 +0800 2025'
    转为 pandas Timestamp（本地显示为北京时间）。
    """
    # to_datetime 对这种格式通常能解析；加 utc=True 更稳
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    # 微博时间本身带 +0800，解析后是 UTC 时间点；转成上海时区方便看
    try:
        ts = ts.dt.tz_convert("Asia/Shanghai")
    except Exception:
        pass
    # 画图时用 naive 也行，这里统一去掉 tz，避免 matplotlib 某些环境警告
    ts = ts.dt.tz_localize(None)
    return ts


def load_retweets_from_json(json_path: Path) -> pd.DataFrame:
    """从单个 retweet_tree_*.json 提取转发节点表（不含 root）。"""
    with open(json_path, "r", encoding="utf-8") as f:
        tree = json.load(f)

    nodes = tree.get("nodes", [])
    if not isinstance(nodes, list) or len(nodes) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(nodes)

    # 只保留转发节点：is_root=False 或 depth>0
    if "is_root" in df.columns:
        df = df[df["is_root"] == False].copy()
    elif "depth" in df.columns:
        df = df[df["depth"].fillna(0).astype(int) > 0].copy()
    else:
        # 没字段就保守：去掉 parent_id 为 null 的
        if "parent_id" in df.columns:
            df = df[df["parent_id"].notna()].copy()

    if df.empty:
        return df

    # 解析时间
    if "created_at" not in df.columns:
        return pd.DataFrame()

    df["parsed_time"] = _parse_created_at(df["created_at"])
    df = df.dropna(subset=["parsed_time"]).copy()
    df = df.sort_values("parsed_time")

    # 规范字段
    if "depth" in df.columns:
        df["depth"] = pd.to_numeric(df["depth"], errors="coerce").fillna(0).astype(int)
    else:
        df["depth"] = 1  # 没 depth 就当 1

    if "user_name" not in df.columns:
        df["user_name"] = ""

    if "region_name" in df.columns:
        df["region"] = df["region_name"].apply(_extract_region)
    elif "region" not in df.columns:
        df["region"] = ""

    return df


def plot_curves_for_json(json_path: Path):
    print(f"\n处理文件：{json_path.name}")
    df_ret = load_retweets_from_json(json_path)

    if df_ret.empty:
        print("  没有可用转发节点（或时间解析失败），跳过。")
        return

    # 主传播期截断（累计到 MAIN_Q 的时间点）
    t_min = df_ret["parsed_time"].min()
    times_sorted = df_ret["parsed_time"].sort_values().reset_index(drop=True)
    n = len(times_sorted)
    k = int(np.ceil(MAIN_Q * n)) - 1
    k = max(0, min(k, n - 1))
    t_end = times_sorted.iloc[k] + pd.Timedelta(hours=RIGHT_PAD_HOURS)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax_cum, ax_heat, ax_region = axes

    # =========================
    # 1) 累计转发曲线（总体 + 按 depth）
    # =========================
    times_all = df_ret["parsed_time"].values
    counts_all = range(1, len(times_all) + 1)
    ax_cum.step(times_all, counts_all, where="post", label="All depths")

    depths = sorted(df_ret["depth"].dropna().unique())
    for d in depths:
        df_d = df_ret[df_ret["depth"] == d].copy()
        if df_d.empty:
            continue
        times_d = df_d["parsed_time"].values
        counts_d = range(1, len(times_d) + 1)
        ax_cum.step(times_d, counts_d, where="post", label=f"depth = {d}")

    ax_cum.set_ylabel("累计转发数")
    ax_cum.set_title(f"转发数量随时间变化（累计）\n{json_path.stem}")
    ax_cum.legend()
    ax_cum.grid(True)

    # =========================
    # 2) 信息热度曲线 H(t)
    # =========================
    df_idx = df_ret.set_index("parsed_time").sort_index()

    # R(t)
    if "id" in df_idx.columns:
        R = df_idx["id"].resample(TIME_BIN).count()
    else:
        R = df_idx.resample(TIME_BIN).size()

    # U(t)
    U = df_idx["user_name"].resample(TIME_BIN).nunique() if "user_name" in df_idx.columns else R * 0.0

    # D(t)
    D = df_idx["depth"].resample(TIME_BIN).mean() if "depth" in df_idx.columns else R * 0.0

    # Z(t)
    if "region" in df_idx.columns:
        region_series = df_idx["region"].astype(str).str.strip().replace("", pd.NA)
        Z = region_series.resample(TIME_BIN).nunique()
    else:
        Z = R * 0.0

    # S(t)：depth>1 比例
    if "depth" in df_idx.columns:
        deep_cnt = (df_idx["depth"] > 1).resample(TIME_BIN).sum()
        S = deep_cnt / R.replace(0, np.nan)
    else:
        S = R * 0.0

    heat_df = pd.DataFrame({"R": R, "U": U, "D": D, "Z": Z, "S": S}).astype(float)
    heat_df = heat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # z-score 标准化（std=0 时置 0）
    for col in heat_df.columns:
        s = heat_df[col]
        std = s.std()
        heat_df[col] = (s - s.mean()) / std if (std and std > 0) else 0.0

    # 归一化权重
    w_sum = sum(WEIGHTS.values())
    w = {k: (v / w_sum if w_sum > 0 else 1.0 / len(WEIGHTS)) for k, v in WEIGHTS.items()}

    heat_series = (
        heat_df["R"] * w["R"]
        + heat_df["U"] * w["U"]
        + heat_df["D"] * w["D"]
        + heat_df["Z"] * w["Z"]
        + heat_df["S"] * w["S"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 只画 R>0 的时间窗
    mask = (R > 0).reindex(heat_series.index, fill_value=False)

    ax_heat.step(
        heat_series.index[mask],
        heat_series.values[mask],
        where="post",
        label=f"H(t)，窗口 = {TIME_BIN}",
    )
    ax_heat.set_ylabel("热度（标准化加权）")
    ax_heat.set_title("信息热度随时间变化（多指标线性组合）")
    ax_heat.grid(True)
    ax_heat.legend()

    # =========================
    # 3) 累计传播地区数-时间曲线
    # =========================
    df_region = df_ret.copy()
    df_region["region"] = df_region["region"].astype(str).str.strip()
    df_region = df_region[df_region["region"] != ""].dropna(subset=["region"])

    if df_region.empty:
        print("  没有有效的地区信息，地区累计曲线为空。")
    else:
        df_first = (
            df_region.sort_values("parsed_time")
            .drop_duplicates(subset="region", keep="first")
            .sort_values("parsed_time")
        )
        times_region = df_first["parsed_time"].values
        counts_region = range(1, len(times_region) + 1)

        ax_region.step(times_region, counts_region, where="post", label="累计传播到的地区数")
        ax_region.set_ylabel("累计地区数")
        ax_region.set_xlabel("时间")
        ax_region.set_title("累计传播地区数随时间变化（不分 depth）")
        ax_region.grid(True)
        ax_region.legend()

    # x 轴锁主传播期
    for ax in axes:
        ax.set_xlim(t_min, t_end)

    plt.tight_layout()
    out_path = out_dir / f"{json_path.stem}_time_depth_heat_region_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  ✓ 图片已保存到: {out_path}")


def main():
    print(f"读取 JSON 目录：{tree_dir}")
    if not tree_dir.exists():
        print("错误：retweet_trees 目录不存在。")
        return

    json_files = sorted([p for p in tree_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
    if not json_files:
        print("未找到任何 .json 文件。")
        return

    print(f"找到 {len(json_files)} 个 JSON 文件。")
    for i, jp in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}]")
        plot_curves_for_json(jp)

    print("\n全部完成！结果保存在：", out_dir)


if __name__ == "__main__":
    main()
