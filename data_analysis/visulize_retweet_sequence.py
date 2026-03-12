import pandas as pd
import plotly.graph_objects as go
import os
import glob
import json

# 获取所有 JSON 文件
base_dir = os.path.dirname(__file__)
folder_path = os.path.join(base_dir, "..",  "data", "macro_data", "train_data") 
json_files = glob.glob(os.path.join(folder_path, "*.json"))

print(f"找到 {len(json_files)} 个 JSON 文件")

# 创建图表
fig = go.Figure()

TIME_LIMIT = 7200  # 2 小时

for file in json_files:
    try:
        # ========== 1. 读取 JSON ==========
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = data.get("nodes", [])
        if len(nodes) == 0:
            print(f"警告: {file} 没有 nodes")
            continue

        # ========== 2. 构建 DataFrame ==========
        df = pd.DataFrame(nodes)

        if "created_at" not in df.columns:
            print(f"警告: {file} 缺少 created_at")
            continue

        # 微博时间 → datetime
        df["parsed_time"] = pd.to_datetime(
            df["created_at"],
            format="%a %b %d %H:%M:%S %z %Y",
            errors="coerce"
        )

        df = df.dropna(subset=["parsed_time"])
        df["parsed_time"] = df["parsed_time"].astype("int64") / 1e9

        # ========== 3. 按时间排序并生成序号 ==========
        df = df.sort_values("parsed_time")
        df["retweet_sequence"] = range(1, len(df) + 1)

        if len(df) == 0:
            print(f"警告: {file} 无有效时间数据")
            continue

        # ========== 4. 时间归一化 ==========
        min_time = df["parsed_time"].min()
        df["relative_time"] = df["parsed_time"] - min_time
        df_filtered = df[df["relative_time"] <= TIME_LIMIT]

        if len(df_filtered) == 0:
            print(f"警告: {file} 前 {TIME_LIMIT/3600:.1f} 小时无数据")
            continue

        file_name = os.path.basename(file)

        # ========== 5. 添加曲线 ==========
        fig.add_trace(go.Scatter(
            x=df_filtered["relative_time"],
            y=df_filtered["retweet_sequence"],
            mode="lines+markers",
            name=file_name,
            hoverinfo="text",
            text=[f"""
<b>文件: {file_name}</b><br>
起始时间戳: {min_time:.2f}<br>
相对时间: {t:.2f}<br>
retweet_sequence: {s}<br>
点: {i+1}/{len(df_filtered)}<br>
(前{TIME_LIMIT/3600:.1f}小时, 共{len(df)}点)
"""
            for i, (t, s) in enumerate(zip(df_filtered["relative_time"], df_filtered["retweet_sequence"]))],
            marker=dict(size=3, opacity=0.1),
            line=dict(width=0.6),
            opacity=0.8,
            hovertemplate="%{text}<extra></extra>"
        ))

        print(f"已处理: {file_name}, 节点数: {len(df)}")

    except Exception as e:
        print(f"处理 {file} 出错: {e}")


# 更新图表布局 - 移除图例
fig.update_layout(
    title={
        'text': f'所有文件的时间归一化曲线（前{TIME_LIMIT/3600:.0f}小时）<br><sup>悬停在曲线上查看文件名和详细信息</sup>',
        'font': dict(size=22, family="Arial, sans-serif"),
        'y': 0.95
    },
    xaxis_title={
        'text': f'相对时间（从0开始，限制{TIME_LIMIT/3600:.0f}小时）',
        'font': dict(size=14)
    },
    yaxis_title={
        'text': 'retweet_sequence',
        'font': dict(size=14)
    },
    hovermode='closest',
    template='plotly_white',
    height=800,
    showlegend=False,  # 这里改为False，不显示图例
    margin=dict(l=80, r=80, t=100, b=80),  # 减少右侧边距，因为不需要为图例留空间
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial",
        bordercolor="black"
    )
)

# 添加网格，设置x轴范围到7200秒
fig.update_xaxes(
    showgrid=True, 
    gridwidth=1, 
    gridcolor='LightGray',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='gray',
    range=[0, TIME_LIMIT]  # 固定x轴范围为0-72000秒
)

fig.update_yaxes(
    showgrid=True, 
    gridwidth=1, 
    gridcolor='LightGray',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='gray'
)

# 显示图表
fig.show()

# 保存为交互式HTML文件
# save_option = input("是否保存为HTML文件？(y/n): ")
# if save_option.lower() == 'y':
#     fig.write_html(f"time_normalized_{TIME_LIMIT//3600}h_no_legend.html")
#     print(f"已保存为 'time_normalized_{TIME_LIMIT//3600}h_no_legend.html'，用浏览器打开即可交互")