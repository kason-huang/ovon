import json

ovon_data_path = "/root/workspace/lab/ovon/data/datasets/ovon/hm3d/v1/train/content/1S7LAXRdDqk.json"
goat_bench_data_path = "/root/workspace/lab/goat-bench/data/datasets/goat_bench/hm3d/v1/train/content/1S7LAXRdDqk.json"

ovon_data_path_with_image = "/root/workspace/lab/ovon/data/datasets/ovon/hm3d/v2/train/content/1S7LAXRdDqk_new_gen.json"

def transfer_image_goals():
    # 加载 source.json
    with open(goat_bench_data_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    # 加载 target.json
    with open(ovon_data_path, 'r+', encoding='utf-8') as f:
        target_data = json.load(f)


    # 添加image_goals内容
    source_goals = source_data['goals']
    target_goals = target_data['goals_by_category']

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

    # 给episodes添加tasks字段，是一个数组，内容是1个，然后这个东西是从source的tasks字段获取的

    # 写入到新文件 merged.json
    with open(ovon_data_path_with_image, 'w', encoding='utf-8') as f:
        json.dump(target_data, f, indent=4)

def fill_image_goals_field():
    ovon_data_path_with_image = "/root/workspace/lab/ovon/data/datasets/ovon/hm3d/v2/train/content/1S7LAXRdDqk.json"
    output_path = "/root/workspace/lab/ovon/data/datasets/ovon/hm3d/v2/train/content/1S7LAXRdDqk_new_gen.json"

    # 加载 target.json
    with open(ovon_data_path_with_image, 'r+', encoding='utf-8') as f:
        target_data = json.load(f)


    # 添加image_goals内容
    target_goals = target_data['goals_by_category']

    for goal_key in target_goals:
            target_list = target_goals[goal_key]
            for i in range(len(target_list)):
                if 'image_goals' not in target_list[i]:
                    target_list[i]['image_goals'] = []

    # 给episodes添加tasks字段，是一个数组，内容是1个，然后这个东西是从source的tasks字段获取的

    # 写入到新文件 merged.json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(target_data, f, indent=4)



fill_image_goals_field()