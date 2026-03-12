import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ================= 路径配置 =================
BASE = Path(__file__).parent
TREE_DIR = BASE / "retweet_trees_for_visualization2"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= 遍历所有 JSON =================
json_files = sorted(TREE_DIR.glob("retweet_tree_*.json"))
print(f"发现 {len(json_files)} 个转发树文件")

# ================= 工具函数：人类可读数值 =================
def human_format(v: float) -> str:
    v = float(v)
    if v >= 1_000_000:
        s = f"{v/1_000_000:.1f}".rstrip("0").rstrip(".")
        return s + "M"
    if v >= 1_000:
        s = f"{v/1_000:.1f}".rstrip("0").rstrip(".")
        return s + "k"
    return f"{int(v)}"

for json_path in json_files:
    print(f"处理: {json_path.name}")

    with open(json_path, "r", encoding="utf-8") as f:
        tree = json.load(f)

    nodes = tree.get("nodes", [])
    meta_mid = tree.get("metadata", {}).get("original_mid", None)

    # 输出命名：radial_<mid>.png
    mid = str(meta_mid) if meta_mid else json_path.stem.replace("retweet_tree_", "")

    if not nodes:
        print("  ⚠ nodes 为空，跳过")
        continue

    # ---------- 按 depth 分组 ----------
    nodes_by_depth = defaultdict(list)
    max_depth = 0
    for n in nodes:
        d = int(n.get("depth", 0))
        nodes_by_depth[d].append(n)
        max_depth = max(max_depth, d)

    # ---------- ✅ children_total frontier 定义 ----------
    # d=1: root.children_total
    # d=k: sum of children_total of nodes at depth k-1
    depths = list(range(1, max_depth + 1))
    values = np.zeros(len(depths), dtype=float)

    for i, d in enumerate(depths):
        s = 0
        for n in nodes_by_depth.get(d - 1, []):
            ct = n.get("children_total")
            if ct is not None:
                s += int(ct)
        values[i] = float(s)

    # 若全为 0，跳过
    if values.max() <= 0:
        print("  ⚠ d>=1 的 children_total 全为 0，跳过")
        continue

    # ================= 极坐标环形条形图 =================
    N = len(depths)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    width = 2 * np.pi / N

    # log 半径
    r = np.log10(values + 1)

    fig = plt.figure(figsize=(7.2, 7.2))
    ax = plt.subplot(111, polar=True)

    # 颜色：按 depth 渐变
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.2, 0.9, N))

    bar_width = width * 0.85
    ax.bar(
        theta,
        r,
        width=bar_width,
        bottom=0.0,
        align="edge",
        color=colors,
        edgecolor="white",
        linewidth=1
    )

    # ---------- 极坐标设置 ----------
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # depth 标签（只剩 d>=1）
    ax.set_xticks(theta + bar_width / 2)
    ax.set_xticklabels([f"d={d}" for d in depths], fontsize=11)

    # ---------- 半径刻度 ----------
    vmax = float(values.max())
    radial_ticks = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    radial_ticks = [t for t in radial_ticks if t <= vmax]
    if not radial_ticks:
        radial_ticks = [int(vmax)]

    ax.set_yticks(np.log10(np.array(radial_ticks) + 1))
    ax.set_yticklabels([human_format(t) for t in radial_ticks], fontsize=10)

    # 把 r 轴刻度标签固定到 d=6 方向（若深度不足则用最后一层）
    idx_rlabel = min(6, N - 1)
    theta_rlabel_center = theta[idx_rlabel] + bar_width / 2
    ax.set_rlabel_position(np.degrees(theta_rlabel_center))

    # ---------- 标题 ----------
    ax.set_title(
        f"Retweet Diffusion by Depth (Radial)\nMID: {mid}",
        va="bottom",
        fontsize=14,
        fontweight="bold"
    )

    # ---------- 数值标注 ----------
    inner_pos = 0.72
    for i, v in enumerate(values):
        if v <= 0:
            continue

        theta_c = theta[i] + bar_width / 2
        r_out = np.log10(v + 1)
        r_text = r_out * inner_pos

        txt = ax.text(
            theta_c,
            r_text,
            human_format(v),
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            rotation=0
        )
        txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    plt.tight_layout()

    # ---------- 保存 ----------
    out_path = OUTPUT_DIR / f"radial_{mid}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"  ✅ 已保存: {out_path}")

print("全部处理完成 ✅")
