# -*- coding: utf-8 -*-
"""
微博转发树爬虫 - 批量爬取多个微博的完整转发树
"""

import requests
import json
import re
import time
import random
from pathlib import Path
from collections import defaultdict, deque

# ============ 配置部分 ============

# 1. 项目路径配置
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "macrodata"
RESULTS_DIR = PROJECT_ROOT / "WeiboSpider" / "results"

# 确保目录存在
for dir_path in [DATA_DIR, OUTPUT_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 2. 输入输出文件路径
COOKIE_FILE = PROJECT_ROOT / "WeiboSpider" /"cookie.txt"
TEST_MIDS_FILE = PROJECT_ROOT / "WeiboSpider" /"mids.txt"  # 微博ID文件


# 3. 爬虫参数配置
CRAWLER_CONFIG = {
    "timeout": 30,
    "retry_times": 3,
    "sleep_time": 2,
    "max_retweets_per_tree": 5000,   # 单棵树最大转发数限制
    "max_depth": 5,                  # 最大转发深度限制
    "max_children": 5000,              # 单节点最大子节点数限制
    "max_trees": 10,                 # 最大爬取树数量
    "branching_factor": 1000,
    "max_nodes_per_tree": 3000
}

# ============ 辅助函数 ============

def read_cookie():
    """从文件读取cookie"""
    possible_files = [COOKIE_FILE, PROJECT_ROOT / "cookies.txt"]
    
    for file_path in possible_files:
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    cookie = f.read().strip()
                    if cookie:
                        print(f"✓ 从文件读取cookie: {file_path.name}")
                        return cookie
            except Exception as e:
                print(f"⚠ 读取cookie文件失败 {file_path}: {e}")
    
    print("❌ 未找到有效的cookie文件")
    return None

def read_test_mids():
    """从testmid.txt读取测试微博ID列表"""
    if TEST_MIDS_FILE.exists():
        try:
            with open(TEST_MIDS_FILE, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                if lines:
                    print(f"✓ 从文件读取微博ID: {len(lines)} 个")
                    return lines
        except Exception as e:
            print(f"⚠ 读取微博ID文件失败: {e}")
    
    print("❌ 未找到testmid.txt文件或文件为空")
    print(f"请在项目根目录创建 {TEST_MIDS_FILE.name} 文件")
    return []

def get_headers(cookie_str, mid):
    """构造请求头"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": f"https://weibo.com/{mid}",
        "Cookie": cookie_str,
        "Accept": "application/json, text/plain, */*",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/json;charset=UTF-8"
    }
    
    # 提取XSRF-Token
    match = re.search(r'XSRF-TOKEN=([\w-]+)', cookie_str)
    if match:
        headers["X-Xsrf-Token"] = match.group(1)
    
    return headers

# ============ 核心爬取函数 ============

def get_weibo_info(mid, headers):
    """获取微博原始信息"""
    url = "https://weibo.com/ajax/statuses/show"
    params = {"id": mid}
    
    for attempt in range(CRAWLER_CONFIG["retry_times"]):
        try:
            resp = requests.get(url, params=params, headers=headers, 
                              timeout=CRAWLER_CONFIG["timeout"])
            
            if resp.status_code != 200:
                time.sleep(3)
                continue
            
            data = resp.json()
            if "ok" in data and data["ok"] == 1:
                return data
                
        except Exception:
            if attempt < CRAWLER_CONFIG["retry_times"] - 1:
                time.sleep(3)
    
    return None

def get_retweet_info(retweet_id, headers):
    """获取单条转发微博的详细信息"""
    url = "https://weibo.com/ajax/statuses/show"
    params = {"id": retweet_id}
    
    try:
        resp = requests.get(url, params=params, headers=headers, 
                          timeout=CRAWLER_CONFIG["timeout"])
        if resp.status_code == 200:
            data = resp.json()
            if "ok" in data and data["ok"] == 1:
                return data
    except:
        pass
    
    return None

def extract_parent_id(weibo_data):
    """从微博数据中提取父微博ID"""
    if "retweeted_status" in weibo_data:
        retweeted = weibo_data["retweeted_status"]
        if retweeted and "id" in retweeted:
            return str(retweeted["id"])
    
    if "retweeted_status_id" in weibo_data:
        return str(weibo_data["retweeted_status_id"])
    
    if "retweeted_mid" in weibo_data:
        return str(weibo_data["retweeted_mid"])
    
    return None

def get_retweet_page(mid, page, headers):
    """获取单页转发列表"""
    url = "https://weibo.com/ajax/statuses/repostTimeline"
    params = {
        "id": mid,
        "page": page,
        "moduleID": "feed",
        "count": 20
    }
    
    for attempt in range(CRAWLER_CONFIG["retry_times"]):
        try:
            resp = requests.get(url, params=params, headers=headers, 
                              timeout=CRAWLER_CONFIG["timeout"])
            
            if resp.status_code != 200:
                time.sleep(3)
                continue
            
            return resp.json()
            
        except Exception:
            if attempt < CRAWLER_CONFIG["retry_times"] - 1:
                time.sleep(3)
    
    return None

def allowed_nodes_for_depth(depth: int) -> int:
    """
    给定深度 d，返回这一层允许的最大节点数 L_d = max(5, 2^d)
    depth 是从 0 开始，0 层是根节点
    """
    return max(5, 2 ** depth)

def get_user_profile(uid, headers):
    """获取用户详细信息 - 借鉴 user_crawler 中的函数"""
    url = "https://weibo.com/ajax/profile/info"
    params = {"uid": uid}
    
    for attempt in range(3):  # 重试3次
        try:
            resp = requests.get(url, params=params, headers=headers, 
                              timeout=30)
            data = resp.json()
            
            # 检查cookie是否失效
            if data.get("ok") == -100:
                print(f"❌ Cookie 失效")
                return None
            
            user = data.get("data", {}).get("user", {})
            if not user:
                return None
            
            # 解析用户信息
            return {
                "uid": str(user.get("id", uid)),
                "screen_name": user.get("screen_name", ""),
                "description": user.get("description", ""),
                "gender": "男" if user.get("gender") == "m" else ("女" if user.get("gender") == "f" else "未知"),
                "location": user.get("location", ""),
                "followers_count": user.get("followers_count", 0),
                "friends_count": user.get("friends_count", 0),
                "statuses_count": user.get("statuses_count", 0),
                "verified": user.get("verified", False),
                "verified_type": user.get("verified_type", -1),
                "verified_reason": user.get("verified_reason", ""),
                "avatar_hd": user.get("avatar_hd", ""),
                "cover_image_phone": user.get("cover_image_phone", ""),
                "urank": user.get("urank", 0),
                "mbrank": user.get("mbrank", 0),
                "created_at": user.get("created_at", ""),
            }
            
        except Exception:
            if attempt < 2:
                time.sleep(2)
    
    return None

def build_retweet_tree_kary(mid, cookie):
    """
    按“k 叉树”思路构建多级转发树（DFS）：
      - 每个节点最多保留 branching_factor 个子节点
      - 深度从 0 开始到 max_depth
      - 整棵树节点数超过 max_retweets_per_tree + 1（含根）就停止
    """
    headers = get_headers(cookie, mid)
    print(f"构建微博 {mid} 的转发树 (k 叉 DFS)...")

    # 1. 获取原始微博信息
    original_weibo = get_weibo_info(mid, headers)
    if not original_weibo:
        print(f"❌ 无法获取微博 {mid} 的信息")
        return None

    root_id = str(original_weibo.get("id", mid))
    original_user_info = get_user_profile(original_weibo.get("user", {}).get("id", ""), headers)
    original_info = {
        "id": root_id,
        "user_id": str(original_weibo.get("user", {}).get("id", "")),
        "user_name": original_weibo.get("user", {}).get("screen_name", ""),
        "user_profile":original_user_info ,
        "text": original_weibo.get("text_raw", ""),
        "created_at": original_weibo.get("created_at", ""),
        "region_name": original_weibo.get("region_name", ""),
        "reposts_count": original_weibo.get("reposts_count", 0),
        "comments_count": original_weibo.get("comments_count", 0),
        "attitudes_count": original_weibo.get("attitudes_count", 0),
        "is_root": True,
        "parent_id": None,
        "depth": 0,
        "original_mid": mid,
    }

    all_nodes = [original_info]
    visited = set([root_id])

    # 统计 / 限制
    max_depth_cfg = CRAWLER_CONFIG["max_depth"]
    max_total_nodes = CRAWLER_CONFIG.get(
    "max_nodes_per_tree",
    CRAWLER_CONFIG["max_retweets_per_tree"] + 1   # 兼容旧配置
    )
    max_children_cfg = CRAWLER_CONFIG["max_children"]
    k = CRAWLER_CONFIG.get("branching_factor", 3)

    total_nodes = 1                 # 含根
    max_depth_reached = 0
    root_direct_retweets = 0
    level_counts = defaultdict(int)
    level_counts[0] = 1

    def parse_items_from_page_data(data):
        if not data or "data" not in data:
            return []
        raw_data = data["data"]
        if isinstance(raw_data, list):
            return raw_data
        if isinstance(raw_data, dict):
            return raw_data.get("data", [])
        return []

    def get_children_for_parent(parent_id: str, depth: int):
        nonlocal total_nodes, root_direct_retweets

        if total_nodes >= max_total_nodes:
            return []

        target_depth = depth + 1
        if target_depth > max_depth_cfg:
            return []

        children = []
        page = 1
        children_tried = 0

        # ★ 这个父节点下面已经出现过的用户（用于去重）
        used_user_ids = set()

        while (
            len(children) < k
            and children_tried < max_children_cfg
            and total_nodes < max_total_nodes
        ):
            print(
                f"[depth {depth}] 获取微博 {parent_id} 的第 {page} 页转发...",
                end=""
            )
            data = get_retweet_page(parent_id, page, headers)
            items = parse_items_from_page_data(data)

            if not items:
                print(" 无更多数据")
                break

            print(f" 找到 {len(items)} 条")

            for item in items:
                if len(children) >= k:
                    break
                if children_tried >= max_children_cfg:
                    break
                if total_nodes >= max_total_nodes:
                    break

                child_id = str(item.get("id", ""))
                if not child_id or child_id in visited:
                    continue

                # ★ 按 user_id 去重：同一用户只保留一个子节点
                child_user_id = str(item.get("user", {}).get("id", ""))
                if child_user_id and child_user_id in used_user_ids:
                    continue
                used_user_ids.add(child_user_id)
                user_profile = get_user_profile(child_user_id, headers)
                node = {
                    "id": child_id,
                    "user_id": child_user_id,
                    "user_name": item.get("user", {}).get("screen_name", ""),
                    "user_profile": user_profile,
                    "text": item.get("text_raw", ""),
                    "created_at": item.get("created_at", ""),
                    "region_name": item.get("region_name", ""),
                    "reposts_count": item.get("reposts_count", 0),
                    "comments_count": item.get("comments_count", 0),
                    "attitudes_count": item.get("attitudes_count", 0),
                    "is_root": False,
                    "parent_id": parent_id,
                    "depth": target_depth,
                    "original_mid": mid,
                }

                children.append(node)
                children_tried += 1

                visited.add(child_id)
                total_nodes += 1

                if total_nodes % 50 == 0 or total_nodes == max_total_nodes:
                    print(f"\n  👉 当前树节点数: {total_nodes} / {max_total_nodes}")

                all_nodes.append(node)

                if parent_id == root_id:
                    root_direct_retweets += 1

            page += 1
            time.sleep(CRAWLER_CONFIG["sleep_time"] + random.uniform(0, 1))

        level_counts[target_depth] += len(children)
        return children


    def dfs(current_id: str, depth: int):
        """从当前节点向下做 DFS，按 k 叉扩展"""
        nonlocal max_depth_reached, total_nodes

        if depth >= max_depth_cfg:
            return
        if total_nodes >= max_total_nodes:
            return

        # 获取当前节点的子节点（最多 k 个）
        children = get_children_for_parent(current_id, depth)
        if not children:
            return

        max_depth_reached = max(max_depth_reached, depth + 1)

        # 对每个子节点继续 DFS
        for child in children:
            if total_nodes >= max_total_nodes:
                break
            dfs(child["id"], depth + 1)

    # 从根开始 DFS
    dfs(root_id, 0)

    tree_structure = {
        "metadata": {
            "original_mid": mid,
            "original_user": original_info["user_name"],
            "crawl_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_retweets": original_weibo.get("reposts_count", 0),
            "crawled_retweets": total_nodes - 1,  # 不含 root
        },
        "root": original_info,
        "nodes": all_nodes,
        "statistics": {
            "total_nodes": total_nodes,
            "direct_retweets": root_direct_retweets,
            "max_depth": max_depth_reached,
            "unique_users": len(
                set([n["user_id"] for n in all_nodes if n["user_id"]])
            ),
            "level_counts": {str(k): v for k, v in sorted(level_counts.items())},
        },
    }

    print(
        f"✓ k 叉 DFS 完成: {total_nodes} 个节点，"
    )
    return tree_structure

def save_retweet_tree(mid, tree_data, format="json"):
    """保存转发树数据"""
    if not tree_data:
        return False
    
    if format == "json":
        filename = f"retweet_tree_{mid}.json"
        output_path = OUTPUT_DIR / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(tree_data, f, ensure_ascii=False, indent=2)
            print(f"  ✓ JSON保存到: {output_path}")
            return True
        except Exception as e:
            print(f"  ❌ 保存失败: {e}")
            return False
    
    return False

def generate_tree_report(tree_data, mid):
    """生成转发树报告"""
    if not tree_data:
        return None
    
    # 分析用户参与度
    user_participation = defaultdict(int)
    for node in tree_data["nodes"]:
        if node["user_id"]:
            user_participation[node["user_id"]] += 1
    
    # 找出参与度最高的用户
    sorted_users = sorted(user_participation.items(), key=lambda x: x[1], reverse=True)[:5]
    
    report = {
        "微博ID": mid,
        "原始用户": tree_data["root"]["user_name"],
        "原始内容预览": tree_data["root"]["text"][:50] + ("..." if len(tree_data["root"]["text"]) > 50 else ""),
        "爬取时间": tree_data["metadata"]["crawl_time"],
        "统计信息": {
            "微博总转发数": tree_data["metadata"]["total_retweets"],
            "实际爬取转发数": tree_data["metadata"]["crawled_retweets"],
            "节点总数": tree_data["statistics"]["total_nodes"],
            "独立用户数": tree_data["statistics"]["unique_users"],
            "最大深度": tree_data["statistics"]["max_depth"]
        },
        "活跃用户Top5": [
            {"user_id": uid, "转发次数": count} for uid, count in sorted_users
        ]
    }
    
    return report

def batch_build_retweet_trees():
    """批量构建多个微博的转发树"""
    # 1. 读取配置
    cookie = read_cookie()
    if not cookie:
        return
    
    # 2. 读取微博ID列表
    all_mids = read_test_mids()
    if not all_mids:
        return
    
    # 3. 限制爬取数量
    max_trees = CRAWLER_CONFIG["max_trees"]
    if len(all_mids) > max_trees:
        print(f"📊 限制爬取前 {max_trees} 个微博（配置: max_trees = {max_trees}）")
        mids = all_mids[:max_trees]
    else:
        mids = all_mids
    
    print(f"\n开始批量构建 {len(mids)} 个微博的转发树...")
    
    # 4. 逐个构建
    results = []
    for i, mid in enumerate(mids, 1):
        print(f"\n{'='*60}")
        print(f"处理第 {i}/{len(mids)} 个微博: {mid}")
        print(f"{'='*60}")
        
        try:
            # 构建转发树
            # 构建转发树（DFS 多级）
            tree_data = build_retweet_tree_kary(mid, cookie)

            if tree_data:
                # 保存数据
                success = save_retweet_tree(mid, tree_data)
                if success:
                    # 生成报告
                    report = generate_tree_report(tree_data, mid)
                    
                    results.append({
                        "mid": mid,
                        "status": "success",
                        "nodes": tree_data["statistics"]["total_nodes"],
                        "file": f"retweet_tree_{mid}.json",
                        "report": report
                    })
                    
                    # 显示简要报告
                    print(f"  节点数: {tree_data['statistics']['total_nodes']}")
                    print(f"  独立用户: {tree_data['statistics']['unique_users']}")
                    print(f"  最大深度: {tree_data['statistics']['max_depth']}")
            else:
                print(f"  微博 {mid} 转发树构建失败")
                results.append({
                    "mid": mid,
                    "status": "failed"
                })
            
            # 微博间的延时
            if i < len(mids):
                wait_time = random.uniform(
                    CRAWLER_CONFIG["sleep_time"] * 1200,
                    CRAWLER_CONFIG["sleep_time"] * 1500
                )
                print(f"等待 {wait_time:.1f} 秒后处理下一个微博...")
                time.sleep(wait_time)
                
        except KeyboardInterrupt:
            print("\n用户中断")
            break
        except Exception as e:
            print(f"处理微博 {mid} 时发生错误: {e}")
            results.append({
                "mid": mid,
                "status": "error",
                "error": str(e)
            })
    
    # 5. 生成总体报告
    print(f"\n{'='*60}")
    print("批量构建完成！统计信息:")
    print('='*60)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    total_nodes = sum(r.get("nodes", 0) for r in results)
    print(f"成功构建: {success_count}/{len(results)} 个转发树")
    print(f"总节点数: {total_nodes}")
    print(f"数据保存到: {OUTPUT_DIR}")
    
    # 保存总体结果
    summary = {
        "total_trees": len(results),
        "successful_trees": success_count,
        "total_nodes": total_nodes,
        "finish_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": CRAWLER_CONFIG,
        "details": results
    }
    
    summary_file = RESULTS_DIR / "retweet_trees_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"详细报告保存到: {summary_file}")
    
    # 显示成功微博的报告摘要
    print(f"\n📊 成功微博报告摘要:")
    print('='*60)
    for result in results:
        if result["status"] == "success" and "report" in result:
            report = result["report"]
            print(f"微博ID: {report['微博ID']}")
            print(f"用户: {report['原始用户']}")
            print(f"节点数: {report['统计信息']['节点总数']}")
            print(f"独立用户: {report['统计信息']['独立用户数']}")
            print("-" * 40)
    
    print("=" * 60)
    
    return results

# ============ 主程序 ============

def main():
    """主程序"""
    print("=" * 60)
    print("微博转发树批量爬虫")
    print("=" * 60)
    print("📋 功能说明:")
    print("  1. 从 testmid.txt 读取多个微博ID")
    print("  2. 为每个微博构建转发树结构")
    print("  3. 保存为JSON格式")
    print("  4. 生成统计报告")
    print("=" * 60)
    
    # 显示配置信息
    print("⚙️ 爬虫参数:")
    for key, value in CRAWLER_CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # 开始批量构建
    batch_build_retweet_trees()

if __name__ == "__main__":
    main()