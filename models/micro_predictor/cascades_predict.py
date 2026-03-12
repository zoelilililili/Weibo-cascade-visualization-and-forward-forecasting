import os
import re
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import numpy as np

# ================= 1. 数据处理工具 =================

def extract_cascade_id(filename: str):
    name = os.path.splitext(filename)[0]
    m = re.search(r'(\d+)$', name)
    if not m: raise ValueError(f"无法提取ID: {filename}")
    return m.group(1)

def load_edges(edge_dir):
    graphs = {}
    files = [f for f in os.listdir(edge_dir) if f.endswith(".json")]
    print(f"正在加载社交图: {len(files)} 个文件")
    for jf in files:
        path = os.path.join(edge_dir, jf)
        cid = extract_cascade_id(jf)
        try:
            with open(path, "r", encoding="utf8") as f:
                data = json.load(f)
            edges = data.get("edges", [])
            G = nx.DiGraph()
            for e in edges:
                if isinstance(e, dict): u, v = str(e["source"]), str(e["target"])
                else: u, v = str(e[0]), str(e[1])
                G.add_edge(u, v)
            if G.number_of_nodes() > 0:
                graphs[cid] = G
        except Exception as e: print(f"加载失败 {jf}: {e}")
    return graphs

def load_cascades(cascade_dir):
    cascades_map = {}
    files = [f for f in os.listdir(cascade_dir) if f.endswith(".json")]
    for jf in files:
        path = os.path.join(cascade_dir, jf)
        cid = extract_cascade_id(jf)
        try:
            with open(path, "r", encoding="utf8") as f:
                data = json.load(f)
            users = [str(u) for u in data.get("users", [])]
            if len(users) >= 2: cascades_map[cid] = {"users": users}
        except: continue
    return cascades_map

# ================= 2. Dataset 与 Collate =================

class CascadeDataset(Dataset):
    def __init__(self, cascades_map, graphs_map):
        self.samples = []
        for cid, data in cascades_map.items():
            if cid not in graphs_map: continue
            users = data["users"]
            G = graphs_map[cid]
            
            # 局部编码: 将字符串ID映射为 0 ~ num_nodes-1
            users = data["users"]
            root_user = users[0]

            # 保证 root 在第一个
            uniq_nodes = [root_user] + [n for n in G.nodes() if n != root_user]

            node2idx = {node: i for i, node in enumerate(uniq_nodes)}
            num_nodes = len(uniq_nodes)

            num_nodes = len(uniq_nodes)
            
            # 构建归一化邻接矩阵 (D^-1 * A)
            adj = torch.zeros((num_nodes, num_nodes))
            for u, v in G.edges():
                adj[node2idx[u], node2idx[v]] = 1
            # 行归一化
            deg = adj.sum(dim=1, keepdim=True)
            adj = adj / (deg + 1e-9)
            
            # 将级联序列转为索引
            seq_indices = [node2idx[u] for u in users if u in node2idx]
            
            # 构造训练样本 (前t个预测第t+1个)
            for t in range(1, len(seq_indices)):
                self.samples.append({
                    "x": torch.tensor(seq_indices[:t]),
                    "y": torch.tensor(seq_indices[t]),
                    "adj": adj,
                    "num_nodes": num_nodes
                })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn(batch):
    xs = [s["x"] for s in batch]
    ys = torch.stack([s["y"] for s in batch])
    adjs = [s["adj"] for s in batch]
    nums = [s["num_nodes"] for s in batch]
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=0)
    lens = torch.tensor([len(x) for x in xs])
    return xs_pad, lens, ys, adjs, nums

# ================= 3. 向量化模型架构 =================

class GNNLayer(nn.Module):
    """向量化 GNN：使用矩阵乘法代替循环"""
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim * 2, dim)

    def forward(self, x, adj):
        # x: (N, dim), adj: (N, N)
        neighbor_features = torch.mm(adj, x) # (N, dim)
        combined = torch.cat([x, neighbor_features], dim=1) # (N, dim*2)
        return F.relu(self.lin(combined))

class CascadeModel(nn.Module):
    def __init__(self, emb_dim=64, hidden_dim=64, max_nodes=1000):
        super().__init__()

        # index 0：root 专用 embedding
        self.root_emb = nn.Parameter(torch.randn(emb_dim))

        # index 1 ~ max_nodes-1：普通节点
        self.other_emb = nn.Embedding(max_nodes - 1, emb_dim)

        self.gnn = GNNLayer(emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, xs_pad, lengths, adjs, nums):
        batch_size = xs_pad.size(0)
        logits_all = []

        for i in range(batch_size):
            # 1. 提取当前级联对应的子图节点 Embedding
            N = nums[i]
            adj = adjs[i].to(xs_pad.device)
            # root embedding
            h0 = self.root_emb.unsqueeze(0)        # (1, emb_dim)

            # other nodes: local index 1..N-1 → embedding 0..N-2
            if N > 1:
                other_idx = torch.arange(N - 1, device=xs_pad.device)
                h_rest = self.other_emb(other_idx)
                h = torch.cat([h0, h_rest], dim=0)
            else:
                h = h0


            # 2. GNN 演化 (1层或多层)
            h_graph = self.gnn(h, adj) # (N, emb_dim)

            # 3. 提取级联序列的动态特征
            seq_idx = xs_pad[i, :lengths[i]] # (seq_len,)
            seq_emb = h_graph[seq_idx].unsqueeze(0) # (1, seq_len, emb_dim)
            
            _, (h_n, _) = self.lstm(seq_emb)
            user_context = self.dropout(h_n[-1].squeeze(0)) # (hidden_dim,)

            # 4. 预测：计算上下文与图中所有节点的相似度
            # W = h_graph (N, emb_dim), user_context (dim,)
            logits = torch.matmul(h_graph, user_context) # (N,)
            logits_all.append(logits)
            
        return logits_all

# ================= 4. 训练与评估 =================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载数据
    base_dir = os.path.dirname(__file__)
    cascades_dir = os.path.join(base_dir, "..", "..", "data", "micro_data", "cascades")
    cascades = load_cascades(cascades_dir)
    graphs_dir = os.path.join(base_dir, "..", "..", "data", "micro_data", "edges")
    graphs = load_edges(graphs_dir)
    
    # 划分数据集 (简单示例)
    cids = list(cascades.keys())
    random.shuffle(cids)
    split = int(0.8 * len(cids))
    train_map = {cid: cascades[cid] for cid in cids[:split]}
    test_map = {cid: cascades[cid] for cid in cids[split:]}

    train_loader = DataLoader(CascadeDataset(train_map, graphs), batch_size=16, 
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(CascadeDataset(test_map, graphs), batch_size=16, 
                             collate_fn=collate_fn)

    model = CascadeModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"开始训练 (设备: {device})...")
    for epoch in range(20):
        model.train()
        total_loss = 0
        for xs, lens, ys, adjs, nums in train_loader:
            xs, ys = xs.to(device), ys.to(device)
            logits_list = model(xs, lens, adjs, nums)
            
            loss = 0
            for i, logits in enumerate(logits_list):
                loss += F.cross_entropy(logits.unsqueeze(0), ys[i].unsqueeze(0))
            loss /= len(logits_list)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

    # 评估
    model.eval()
    mrr = 0.0
    hits5 = hits10 = hits30 = 0
    count = 0
    with torch.no_grad():
        for xs, lens, ys, adjs, nums in test_loader:
            xs = xs.to(device)
            logits_list = model(xs, lens, adjs, nums)
            for i, logits in enumerate(logits_list):
                true_y = ys[i].item()
                # 更稳健的 rank 计算：统计 score 严格大于真实标签分数的数目
                score_true = logits[true_y].item()
                rank = int((logits > score_true).sum().item()) + 1
                mrr += 1.0 / rank
                if rank <= 5: hits5 += 1
                if rank <= 10: hits10 += 1
                if rank <= 30: hits30 += 1
                count += 1

    print(f"\n测试结果:")
    print(f"MRR: {mrr/count:.4f} | Hits@5: {hits5/count:.4f} | Hits@10: {hits10/count:.4f} | Hits@30: {hits30/count:.4f}")

if __name__ == "__main__":
    main()
