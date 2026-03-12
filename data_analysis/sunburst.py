import json
from pathlib import Path
from collections import defaultdict

import plotly.express as px

# ===================== 路径 =====================
BASE = Path(__file__).parent
tree_dir = BASE / "retweet_trees_for_visualization2"

out_dir = BASE / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# ===================== 参数 =====================
USE_VALUE = "count"   # "count" or "children_total"

# ===================== 遍历所有 json =====================
json_files = sorted(tree_dir.glob("*.json"))

print(f"📂 共发现 {len(json_files)} 个 retweet tree json")

for json_path in json_files:
    print(f"\n▶ 处理: {json_path.name}")

    # ===================== 读数据 =====================
    with open(json_path, "r", encoding="utf-8") as f:
        tree = json.load(f)

    nodes = tree["nodes"]
    edges = tree["edges"]
    root_id = tree["root"]["id"]
    mid = tree.get("metadata", {}).get("original_mid", json_path.stem)

    node_by_id = {n["id"]: n for n in nodes}

    # parent 映射：child -> parent
    parent_of = {}
    for e in edges:
        parent_of[e["to"]] = e["from"]

    # ===================== 构造 Sunburst 数据 =====================
    ids, parents, labels, values, colors = [], [], [], [], []

    for nid, n in node_by_id.items():
        ids.append(nid)

        if nid == root_id:
            parents.append("")
        else:
            parents.append(parent_of.get(nid, root_id))

        d = int(n.get("depth", 0))
        uname = n.get("user_name", "")
        labels.append(f"{uname} (d={d})" if uname else f"{nid[:6]}… (d={d})")

        if USE_VALUE == "children_total":
            ct = n.get("children_total")
            values.append(max(1, int(ct) if ct is not None else 1))
        else:
            values.append(1)

        colors.append(d)

    # ===================== 画图 =====================
    fig = px.sunburst(
        ids=ids,
        names=labels,
        parents=parents,
        values=values,
        color=colors,
        color_continuous_scale="Viridis",
        title=f"Retweet Sunburst (MID: {mid})",
    )

    # ===================== hover 信息 =====================
    customdata = []
    for nid in ids:
        n = node_by_id[nid]
        customdata.append([
            n.get("id", ""),
            n.get("user_id", ""),
            n.get("user_name", ""),
            int(n.get("depth", 0)),
            n.get("children_total", None),
            n.get("children_kept", None),
            n.get("children_omitted", None),
        ])

    fig.update_traces(
        customdata=customdata,
        hovertemplate=(
            "<b>%{label}</b><br>"
            "id=%{customdata[0]}<br>"
            "user_id=%{customdata[1]}<br>"
            "depth=%{customdata[3]}<br>"
            "children_total=%{customdata[4]}<br>"
            "children_kept=%{customdata[5]}<br>"
            "children_omitted=%{customdata[6]}<br>"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        margin=dict(t=60, l=10, r=10, b=10),
    )

    # ===================== 保存 =====================
    png_path = out_dir / f"sunburst_{mid}.png"
    try:
        fig.write_image(png_path, width=1200, height=900, scale=2)
        print("  ✅ PNG :", png_path.name)
    except Exception as e:
        print("  ⚠ PNG 导出失败（可忽略）:", str(e))

print("\n🎉 所有 retweet tree 处理完成")
