import json

ovon_data_path = "/root/workspace/lab/ovon/data/datasets/ovon/hm3d/v1/train/content/1S7LAXRdDqk.json"
goat_bench_data_path = "/root/workspace/lab/goat-bench/data/datasets/goat_bench/hm3d/v1/train/content/1S7LAXRdDqk.json"

ovon_data_path_with_image = "/root/workspace/lab/ovon/data/datasets/ovon/hm3d/v1/train/content/1S7LAXRdDqk_image.json"

# 加载 source.json
with open(goat_bench_data_path, 'r', encoding='utf-8') as f:
    source_data = json.load(f)

# 加载 target.json
with open(ovon_data_path, 'r+', encoding='utf-8') as f:
    target_data = json.load(f)

source_goals = source_data['goals']
target_goals = target_data['goals_by_category']

# pirnt episodes.children_object_categories
for episode in target_data['episodes']:
    print(len(episode['children_object_categories']))

exit(-1)

# 遍历所有目标键
for goal_key in source_goals:
    if goal_key in target_goals:
        source_list = source_goals[goal_key]
        target_list = target_goals[goal_key]

        if len(source_list) != len(target_list):
            print(f"{goal_key} array size not equal, {source_goals} and {target_list}")
            exit(-1)

        for i in range(len(source_list)):
            if 'image_goals' in source_list[i]:
                target_list[i]['image_goals'] = source_list[i]['image_goals']


# 写入到新文件 merged.json
with open(ovon_data_path_with_image, 'w', encoding='utf-8') as f:
    json.dump(target_data, f, indent=4)