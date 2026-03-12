#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import json
import re
import time
import random
import sys
from pathlib import Path
import requests
 
# ============ 配置部分 ============

# 1. 项目路径配置
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"

# 确保目录存在
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 2. 输入输出文件路径
COOKIE_FILE = PROJECT_ROOT / "WeiboSpider" / "cookie.txt"
RETWEET_TREES_DIR = DATA_DIR / "micro_data" / "cascades_retweet_trees"
EDGES_DIR = DATA_DIR / "micro_data" / "edges"
EDGES_DIR.mkdir(parents=True, exist_ok=True)

# 3. 爬虫参数配置
CRAWLER_CONFIG = {
    "timeout": 20,
    "retry_times": 3,
    "sleep_time": 2,
    "max_pages_per_user": 50,      # 每个用户最大抓取页数
    "max_users": 1000,              # 最大用户数限制
    "save_follow_relations": True, # 是否保存关注关系
    "save_fans_relations": False,   # 是否保存粉丝关系
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

def parse_verified_type(v_type):
    """解析认证类型"""
    v_map = {
        0: "黄V (名人/KOL)", 
        1: "蓝V (政府)", 
        2: "蓝V (企业)", 
        3: "蓝V (媒体)",
        -1: "普通用户", 
        200: "微博达人", 
        220: "微博达人"
    }
    return v_map.get(v_type, "其他认证")

def get_headers(cookie_str):
    """构造请求头"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://weibo.com",
        "Cookie": cookie_str,
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json;charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest"
    }
    
    # 提取XSRF-Token
    match = re.search(r'XSRF-TOKEN=([\w-]+)', cookie_str)
    if match:
        headers["X-Xsrf-Token"] = match.group(1)
    
    return headers

# ============ 核心爬取函数 ============

def get_profile_info(uid, headers, retry_times=3):
    """获取用户主页基础信息"""
    url = "https://weibo.com/ajax/profile/info"
    params = {"uid": uid}
    
    for attempt in range(retry_times):
        try:
            resp = requests.get(url, params=params, headers=headers, 
                              timeout=CRAWLER_CONFIG["timeout"])
            data = resp.json()
            
            # 检查cookie是否失效
            if data.get("ok") == -100:
                print(f"❌ Cookie 失效，请重新登录！")
                return None
            
            user = data.get("data", {}).get("user", {})
            if not user:
                print(f"⚠ 用户 {uid} 数据为空")
                return None
            
            return {
                "uid": str(user.get("id", uid)),
                "screen_name": user.get("screen_name", "未知用户"),
                "description": user.get("description", ""),
                "gender": "男" if user.get("gender") == "m" else ("女" if user.get("gender") == "f" else "未知"),
                "location": user.get("location", ""),
                "followers_count": user.get("followers_count", 0),
                "friends_count": user.get("friends_count", 0),
                "statuses_count": user.get("statuses_count", 0),
                "verified": user.get("verified", False),
                "verified_type": parse_verified_type(user.get("verified_type", -1)),
                "verified_reason": user.get("verified_reason", ""),
                "crawl_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except requests.exceptions.Timeout:
            print(f"[超时] 获取用户 {uid} 信息超时 ({attempt+1}/{retry_times})")
            if attempt < retry_times - 1:
                time.sleep(3)
        except Exception as e:
            print(f"[错误] 获取用户 {uid} 信息失败: {e}")
            if attempt < retry_times - 1:
                time.sleep(3)
    
    return None

def get_timeline_stats(uid, headers):
    """
    获取用户最近的微博数据，用于提取：
    1. 真实 IP 属地 (ip_location)
    2. 近期互动数据 (赞/评/转)
    3. 阅读量 (部分可见)
    已增强对非 JSON 响应的处理与诊断输出。
    """
    url = "https://weibo.com/ajax/statuses/mymblog"
    params = {
        "uid": uid,
        "page": 1,
        "feature": 0
    }
    print(f"[-] 正在分析最近微博以提取 IP 和互动数据...")

    stats = {
        "ip_location": "未知/未展示",
        "recent_views_sum": 0,
        "recent_likes": 0,
        "recent_comments": 0,
        "recent_reposts": 0,
        "sample_count": 0
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        # 诊断：状态码 / reason / headers / 重定向历史 / 响应片段
        status = resp.status_code
        reason = getattr(resp, "reason", "")
        ctype = resp.headers.get("Content-Type", "")
        text_snip = resp.text[:1000].strip()
        print(f"    [DEBUG] status={status} reason={reason} content-type={ctype} content_len={len(resp.content)}")
        if resp.history:
            print(f"    [DEBUG] 重定向历史: {[ (h.status_code, h.headers.get('Location')) for h in resp.history ]}")
        # 非 200 时输出更多头部和片段，便于判断是否为 HTML 登录页或 WAF 响应
        if status != 200:
            print("    ⚠ 非 200 响应，头部信息：")
            for k, v in list(resp.headers.items())[:20]:
                print(f"      {k}: {v}")
            # 若响应体为空或为 HTML，打印片段供排查
            if not text_snip:
                print("    ⚠ 响应体为空（长度0）")
            else:
                print("    ⚠ 响应片段（前1000字符）:")
                print(text_snip[:1000])
            return stats

        if not text_snip:
            print("    ⚠ 响应体为空")
            return stats

        # 很多时候被重定向到登录页或返回 HTML（以 '<' 开头）
        if text_snip.startswith("<") or "html" in ctype.lower():
            print("    ⚠ 响应为 HTML（可能需要登录或被反爬），片段:")
            print(text_snip[:400])
            return stats

        # 尝试解析 JSON，捕获解析错误并打印片段以便定位
        try:
            data = resp.json()
        except ValueError as je:
            print(f"    ⚠ JSON 解析失败: {je}")
            print("    响应片段:", text_snip[:400])
            return stats

        if "data" not in data or "list" not in data["data"]:
            print("    ⚠ 未能获取微博列表（可能仅半年可见或无权限）")
            return stats

        tweets = data["data"]["list"]
        stats["sample_count"] = len(tweets)

        if not tweets:
            return stats

        for tweet in tweets:
            if "region_name" in tweet and tweet["region_name"]:
                stats["ip_location"] = tweet["region_name"]
                break

        for tweet in tweets:
            stats["recent_reposts"] += tweet.get("reposts_count", 0)
            stats["recent_comments"] += tweet.get("comments_count", 0)
            stats["recent_likes"] += tweet.get("attitudes_count", 0)

        stats["total_interaction"] = stats["recent_likes"] + stats["recent_comments"] + stats["recent_reposts"]

    except requests.exceptions.Timeout:
        print("    ⚠ 请求超时")
    except Exception as e:
        print(f"    ⚠ 分析时间线失败: {e}")

    return stats

def get_relation_uids(uid, headers, relate_type="follow"):
    """获取关注/粉丝 UID 列表"""
    url = "https://weibo.com/ajax/friendships/friends"
    uids_list = []
    max_pages = CRAWLER_CONFIG["max_pages_per_user"]
    sleep_time = CRAWLER_CONFIG["sleep_time"]
    
    relation_name = "粉丝" if relate_type == "fans" else "关注"
    print(f"  正在抓取{relation_name}列表 (最多{max_pages}页)...")
    
    for page in range(1, max_pages + 1):
        params = {
            "relate": "fans" if relate_type == "fans" else "follow",
            "page": page,
            "uid": uid,
            "type": "all"
        }
        
        try:
            resp = requests.get(url, params=params, headers=headers, 
                              timeout=CRAWLER_CONFIG["timeout"])
            data = resp.json()
            
            if "users" not in data or not data["users"]:
                break
                
            # 提取用户信息
            for user in data["users"]:
                user_info = {
                    "uid": str(user.get("id", "")),
                    "screen_name": user.get("screen_name", ""),
                    "verified": user.get("verified", False),
                    "verified_type": parse_verified_type(user.get("verified_type", -1)),
                    "followers_count": user.get("followers_count", 0),
                    "friends_count": user.get("friends_count", 0)
                }
                uids_list.append(user_info)     
            # 页间延时
            time.sleep(sleep_time + random.uniform(0, 1))
        except Exception as e:
            print(f"  抓取{relation_name}第{page}页失败: {e}")
            break
    return uids_list

def batch_crawl_users():
    """批量爬取多个用户
    已重载为：对 retweet_trees 下每个树文件分别提取 user_id 集合，构建关注关系有向边并分别保存到 data/raw/edges/edges_{树id}.json
    """
    # 1. 读取 cookie
    cookie = read_cookie()
    if not cookie:
        return

    # 内部工具：从对象中递归收集 user_id
    def collect_ids_from_obj(obj, keys=("user_id", "uid")):
        ids = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in keys:
                    if v is None:
                        continue
                    ids.add(str(v))
                else:
                    ids.update(collect_ids_from_obj(v, keys))
        elif isinstance(obj, list):
            for item in obj:
                ids.update(collect_ids_from_obj(item, keys))
        return ids

    if not RETWEET_TREES_DIR.exists():
        print(f"❌ 未找到目录: {RETWEET_TREES_DIR}，请先生成 retweet_trees 数据")
        return

    json_files = list(RETWEET_TREES_DIR.glob("*.json"))
    print(f"✓ 发现 retweet_trees 文件: {len(json_files)} 个，开始逐文件提取并生成 edges ...")

    headers = get_headers(cookie)
    sleep_base = CRAWLER_CONFIG.get("sleep_time", 2)

    overall_summary = []

    for jf in json_files:
        print(f"\n--- 处理树文件: {jf.name} ---")
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ⚠ 无法读取或解析 {jf.name}: {e}")
            continue

        # 提取本树的用户集合
        ids = collect_ids_from_obj(data)
        if not ids:
            print(f"  ⚠ 未从 {jf.name} 提取到任何 user_id，跳过")
            continue

        # 标准化 id 集合
        normalized = set()
        for uid in ids:
            s = str(uid).strip()
            if not s:
                continue
            s = re.sub(r"\s+", "", s)
            normalized.add(s)
        user_set = normalized
        total = len(user_set)
        print(f"  ✓ 提取到 {total} 个唯一 user_id")

        # 为本树构建边集合
        edges = set()
        processed = 0

        for uid in list(user_set):
            processed += 1
            print(f"  [{processed}/{total}] 检查用户 {uid} 的关注列表 ...")
            try:
                follow_list = get_relation_uids(uid, headers, relate_type="follow")
                for u in follow_list:
                    tgt = str(u.get("uid", ""))
                    if tgt and tgt in user_set:
                        edges.add((str(uid), tgt))
                time.sleep(sleep_base + random.uniform(0, 1))
            except KeyboardInterrupt:
                print("  用户中断操作")
                break
            except Exception as e:
                print(f"  ⚠ 获取 {uid} 关注失败: {e}")
                continue

        edges_list = [{"source": s, "target": t} for s, t in sorted(edges)]
        tree_id = jf.stem  # 使用原文件名的 stem 作为树 id
        out_file = EDGES_DIR / f"edges_{tree_id}.json"
        summary = {
            "source_tree_file": jf.name,
            "tree_id": tree_id,
            "total_users_scanned": total,
            "edges_count": len(edges_list),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "edges": edges_list
        }

        try:
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"  ✓ 本树边集合已保存: {out_file}，共 {len(edges_list)} 条有向边")
            overall_summary.append({"tree_file": jf.name, "tree_id": tree_id, "users": total, "edges": len(edges_list), "file": str(out_file)})
        except Exception as e:
            print(f"  ❌ 保存 {out_file} 失败: {e}")

    print("\n全部树处理完成。")
    return overall_summary

# ============ 主程序 ============

def main():
    """主程序"""
    print("=" * 60)
    print("微博用户信息爬虫 - 批量爬取版")
    print("=" * 60)
    
    # 显示配置信息
    print("📋 配置信息:")
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  输出目录: {EDGES_DIR}")
    print()
    print("⚙️ 爬虫参数:")
    for key, value in CRAWLER_CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # 开始爬取
    batch_crawl_users()

if __name__ == "__main__":
    main()