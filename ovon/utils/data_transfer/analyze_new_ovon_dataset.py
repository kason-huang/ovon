import os
import gzip
import json
import pandas as pd

def find_missing_image_goals(root_dir):
    """
    遍历数据集，找出缺少 image_goals 的目标。
    返回 DataFrame，包含 split、文件路径、goal_key。
    """
    missing_records = []

    # 遍历四个 split 目录
    for split in ["train", "val_seen", "val_seen_synonyms", "val_unseen"]:
        content_dir = os.path.join(root_dir, split, "content")
        if not os.path.exists(content_dir):
            continue
        
        # 遍历 .json.gz 文件
        for filename in os.listdir(content_dir):
            if filename.endswith(".json.gz"):
                file_path = os.path.join(content_dir, filename)
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查 goals
                goals = data.get("goals_by_categories", {})
                for goal_key, goal_list in goals.items():
                    for goal in goal_list:
                        if not goal.get("image_goals"):  # 缺失或为空
                            missing_records.append({
                                "split": split,
                                "file_path": file_path,
                                "goal_key": goal_key
                            })
    
    return pd.DataFrame(missing_records)

# 使用示例
if __name__ == "__main__":
    dataset_path = "/root/workspace/lab/goat-bench/data/datasets/goat_bench/hm3d/v2_new"
    df_missing = find_missing_image_goals(dataset_path)
    
    if df_missing.empty:
        print("所有 goals 都包含 image_goals，没有缺失。")
    else:
        print("缺失 image_goals 的目标：")
        print(df_missing)
