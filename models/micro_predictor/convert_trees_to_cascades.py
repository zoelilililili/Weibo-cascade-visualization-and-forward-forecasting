import os
import json
import glob
from datetime import datetime

# ...existing code...
def _parse_time(s):
    # 尝试解析常见的微博 created_at 格式: "Sun Dec 07 12:46:59 +0800 2025"
    try:
        return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
    except Exception:
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None

def _collect_node_ids_from_edges_obj(obj):
    """
    从 edges JSON 结构中遍历并收集可能的节点 id（字符串形式）。
    支持 dict/list、值对形式、以及常见键名（source/target/from/to/...）。
    """
    ids = set()
    candidate_keys = {
        "source", "target", "from", "to", "src", "dst",
        "source_id", "target_id", "src_id", "dst_id",
        "u", "v", "user_id", "id", "uid"
    }

    def _rec(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in candidate_keys:
                    if isinstance(v, (str, int)):
                        ids.add(str(v))
                    elif isinstance(v, (list, tuple)):
                        for it in v:
                            if isinstance(it, (str, int)):
                                ids.add(str(it))
                else:
                    _rec(v)
        elif isinstance(o, (list, tuple)):
            # 如果是长度为2的值对，且元素为原子类型，采集两端
            if len(o) == 2 and all(isinstance(x, (str, int)) for x in o):
                ids.add(str(o[0])); ids.add(str(o[1]))
            for it in o:
                _rec(it)
        # 其他类型忽略

    _rec(obj)
    return ids

def _load_edges_node_ids_for_tree(tree_fn):
    """
    根据 retweet_tree 文件名构造对应的 edges 文件名（edges_<basename>），
    加载并返回 edges 中出现的所有节点 id（字符串集合）。
    若文件缺失或加载失败返回 None。
    """
    base_dir = os.path.dirname(__file__)
    edges_dir = os.path.normpath(os.path.join(base_dir, "..", "..", "data", "micro_data", "edges"))
    edges_fname = f"edges_{os.path.basename(tree_fn)}"
    edges_path = os.path.join(edges_dir, edges_fname)
    if not os.path.exists(edges_path):
        return None
    try:
        with open(edges_path, "r", encoding="utf8") as fe:
            data = json.load(fe)
    except Exception:
        return None
    return _collect_node_ids_from_edges_obj(data)

def convert_retweet_trees(src_dir=None, dst_dir=None, write_aggregate=True):
    base_dir = os.path.dirname(__file__)
    if src_dir is None:
        src_dir = os.path.join(base_dir, "..", "..", "data", "micro_data", "cascades_retweet_trees")
    if dst_dir is None:
        dst_dir = os.path.join(base_dir, "..", "..", "data", "micro_data", "cascades")
    src_dir = os.path.normpath(src_dir)
    dst_dir = os.path.normpath(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(src_dir, "*.json")))
    all_cascades = []
    for fn in files:
        try:
            with open(fn, "r", encoding="utf8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"skip {fn}: load error {e}")
            continue

        # 尝试加载对应 edges 文件中出现的节点集合
        edge_nodes = _load_edges_node_ids_for_tree(fn)
        if edge_nodes is None:
            print(f"skip {fn}: corresponding edges file not found or can't be read")
            continue

        # cascade id
        cascade_id = None
        if isinstance(data, dict):
            cascade_id = data.get("metadata", {}).get("original_mid") or data.get("root", {}).get("id")
        if not cascade_id:
            cascade_id = os.path.splitext(os.path.basename(fn))[0]

        nodes = data.get("nodes", []) if isinstance(data, dict) else []
        # collect (time, user_id) pairs — 仅加入在 edges 中出现的节点
        seq = []
        for nd in nodes:
            uid = nd.get("user_id") or nd.get("id") or nd.get("user_name")
            if uid is None:
                continue
            uid_str = str(uid)
            if uid_str not in edge_nodes:
                # 该节点在 edges 中未出现，跳过
                continue
            ctime = nd.get("created_at") or nd.get("crawl_time") or None
            parsed = _parse_time(ctime) if ctime else None
            seq.append((parsed, ctime, uid_str))

        # sort by parsed time when possible; fallback keep original order
        if any(tup[0] is not None for tup in seq):
            # nodes with None time will be placed at the end in original order
            seq_sorted = sorted(enumerate(seq), key=lambda x: (x[1][0] is None, x[1][0] or datetime.min, x[0]))
            users = [t[1][2] for t in seq_sorted]
        else:
            users = [t[2] for t in seq]

        # ensure root appears first if present by id matching metadata.root.id
        root_id = data.get("root", {}).get("id") if isinstance(data, dict) else None
        if root_id and root_id in users:
            # move root to front preserving relative order otherwise
            users = [root_id] + [u for u in users if u != root_id]

        # keep cascades length >= 2
        if len(users) < 2:
            continue

        out_obj = {"cascade_id": str(cascade_id), "users": users}
        out_fn = os.path.join(dst_dir, f"{cascade_id}.json")
        try:
            with open(out_fn, "w", encoding="utf8") as fw:
                json.dump(out_obj, fw, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"write error {out_fn}: {e}")
            continue

        all_cascades.append(out_obj)

    print(f"Processed {len(all_cascades)} cascades -> {dst_dir}")
    return all_cascades

if __name__ == "__main__":
    convert_retweet_trees()