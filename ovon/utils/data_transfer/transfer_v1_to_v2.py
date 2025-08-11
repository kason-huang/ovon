import os
import gzip
import json
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import shutil

input_root = "/root/workspace/lab/goat-bench/data/datasets/goat_bench/hm3d/v1"
output_root = "/root/workspace/lab/ovon/data/datasets/ovon/hm3d/v2_new/"  # 新数据集输出目录
os.makedirs(output_root, exist_ok=True)
splits = ["train", "val_seen", "val_seen_synonyms", "val_unseen"]

print_lock = Lock()

def process_one_file(in_path: str, out_path: str) -> int:
    """处理一个 content/ 下的 json.gz 文件"""
    with gzip.open(in_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    new_episodes = []
    ep_counter = 0
    for ep in data.get("episodes", []):
        tasks = ep.get("tasks", [])
        if not tasks:
            continue

        base = dict(ep)
        base.pop("tasks", None)
        base.pop("episode_id", None)

        for task in tasks:
            # 跳过 description 模态
            if len(task) > 1 and task[1] == "description":
                continue

            new_ep = dict(base)
            new_ep["tasks"] = [task]
            new_ep["episode_id"] = ep_counter
            ep_counter += 1
            new_episodes.append(new_ep)

    new_data = {
        "episodes": new_episodes,
        "goals": data.get("goals", {})
    }

    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False)

    return len(new_episodes)

def main():
    tasks_list = []

    for split in splits:
        in_split_dir = os.path.join(input_root, split)
        out_split_dir = os.path.join(output_root, split)
        os.makedirs(out_split_dir, exist_ok=True)

        # 1️⃣ 先拷贝 split 目录下的 *.json.gz（不是 content 里的）
        for file_path in glob(os.path.join(in_split_dir, "*.json.gz")):
            shutil.copy(file_path, os.path.join(out_split_dir, os.path.basename(file_path)))
            with print_lock:
                print(f"Copied {file_path} -> {out_split_dir}")

        # 2️⃣ 收集 content 下的待处理文件
        in_content_dir = os.path.join(in_split_dir, "content")
        out_content_dir = os.path.join(out_split_dir, "content")
        os.makedirs(out_content_dir, exist_ok=True)

        for in_path in glob(os.path.join(in_content_dir, "*.json.gz")):
            out_path = os.path.join(out_content_dir, os.path.basename(in_path))
            tasks_list.append((in_path, out_path))

    if not tasks_list:
        print("未找到任何 content/*.json.gz 文件")
        return
    print(f"total json.gz num: {len(tasks_list)}")

    # 3️⃣ 多线程处理 content 下的文件
    max_workers = min(32, len(tasks_list))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one_file, in_path, out_path): (in_path, out_path) for in_path, out_path in tasks_list}

        done_cnt = 0
        for fut in as_completed(futures):
            done_cnt += 1
            try:
                n = fut.result()
                with print_lock:
                    print(f"[{done_cnt}/{len(futures)}] OK, new episodes: {n}")
            except Exception as e:
                with print_lock:
                    print(f"[{done_cnt}/{len(futures)}] ERROR: {e}")

if __name__ == "__main__":
    main()