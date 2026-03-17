#!/usr/bin/env python3
"""
生成覆盖率元数据 (coverage_meta.pkl)
用于新场景的 Object Navigation 任务生成
"""

import os
import pickle
import argparse
from collections import defaultdict
from tqdm import tqdm

import numpy as np
from ovon.dataset.objectnav_generator import ObjectGoalGenerator
from ovon.dataset.semantic_utils import get_hm3d_semantic_scenes


def generate_coverage_meta(
    split="train",
    scene_datasets_path="data/scene_datasets/hm3d",
    categories=None,
    output_path="data/coverage_meta",
    device_id=0,
    max_scenes=None,
):
    """
    生成覆盖率元数据

    Args:
        split: "train" 或 "val"
        scene_datasets_path: 场景数据集路径
        categories: 允许的物体类别列表
        output_path: 输出路径
        device_id: GPU 设备 ID
        max_scenes: 最大处理场景数（用于测试）
    """

    if categories is None:
        categories = ["chair", "bed", "toilet", "sofa", "plant", "tv_monitor"]

    print(f"=== 生成覆盖率元数据 ({split}) ===")
    print(f"场景路径: {scene_datasets_path}")
    print(f"物体类别: {categories}")
    print(f"GPU 设备: {device_id}")

    # 1. 获取场景列表
    print(f"\n[1/4] 获取场景列表...")
    scenes = get_hm3d_semantic_scenes(scene_datasets_path, [split])
    scenes = sorted(scenes)

    if max_scenes:
        scenes = scenes[:max_scenes]
        print(f"限制场景数量: {len(scenes)}")

    print(f"找到 {len(scenes)} 个场景")

    # 2. 初始化 ObjectGoalGenerator
    print(f"\n[2/4] 初始化生成器...")

    # 注意：这里暂时传入 None，我们会修改代码跳过覆盖率的检查
    objectgoal_maker = ObjectGoalGenerator(
        semantic_spec_filepath=f"{scene_datasets_path}/hm3d_annotated_basis.scene_dataset_config.json",
        img_size=(512, 512),
        hfov=90,
        agent_height=1.41,
        agent_radius=0.17,
        sensor_height=1.31,
        pose_sampler_args={
            "r_min": 0.5,
            "r_max": 2.0,
            "r_step": 0.5,
            "rot_deg_delta": 10.0,
            "h_min": 0.8,
            "h_max": 1.4,
            "sample_lookat_deg_delta": 5.0,
        },
        mapping_file="ovon/dataset/source_data/Mp3d_category_mapping.tsv",
        categories=categories,
        coverage_meta_file=None,  # ← 关键：设为 None
        frame_cov_thresh=0.05,
        goal_vp_cell_size=0.25,
        goal_vp_max_dist=1.0,
        start_poses_per_obj=10,  # 覆盖率生成时不需要太多
        device_id=device_id,
        sample_dense_viewpoints=True,
    )

    # 3. 遍历场景，计算覆盖率
    print(f"\n[3/4] 计算物体覆盖率...")

    coverage_metadata = defaultdict(list)

    for scene in tqdm(scenes, desc="处理场景"):
        try:
            # 配置仿真
            sim = objectgoal_maker._config_sim(scene)

            # 获取所有物体
            all_objects = sim.semantic_scene.objects

            # 过滤出目标类别的物体
            for obj in all_objects:
                object_category = objectgoal_maker.cat_map.get(obj.category.name())
                if object_category is None:
                    continue

                # 生成视点并计算覆盖率
                try:
                    viewpoints = objectgoal_maker._make_object_viewpoints(sim, obj)

                    # 记录所有视点的覆盖率
                    for vp in viewpoints:
                        coverage = vp["iou"]
                        coverage_metadata[object_category].append(
                            (coverage, vp, scene)
                        )

                except Exception as e:
                    print(f"  ⚠️  物体 {obj.id} 处理失败: {e}")
                    continue

            sim.close()

        except Exception as e:
            print(f"  ❌ 场景 {scene} 处理失败: {e}")
            continue

    # 4. 保存覆盖率元数据
    print(f"\n[4/4] 保存覆盖率元数据...")

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{split}.pkl")

    with open(output_file, "wb") as f:
        pickle.dump(dict(coverage_metadata), f)

    print(f"✅ 已保存到: {output_file}")

    # 打印统计信息
    print(f"\n=== 统计信息 ===")
    for category, data in coverage_metadata.items():
        coverages = [c[0] for c in data]
        print(f"{category}: {len(data)} 个视点, "
              f"平均覆盖率: {np.mean(coverages):.3f}, "
              f"最小: {np.min(coverages):.3f}, "
              f"最大: {np.max(coverages):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成覆盖率元数据")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"],
                        help="数据集划分")
    parser.add_argument("--scene-datasets-path", type=str,
                        default="data/scene_datasets/hm3d",
                        help="场景数据集路径")
    parser.add_argument("--output-path", type=str,
                        default="data/coverage_meta",
                        help="输出路径")
    parser.add_argument("--device-id", type=int, default=0,
                        help="GPU 设备 ID")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="最大处理场景数（用于测试）")

    args = parser.parse_args()

    generate_coverage_meta(
        split=args.split,
        scene_datasets_path=args.scene_datasets_path,
        output_path=args.output_path,
        device_id=args.device_id,
        max_scenes=args.max_scenes,
    )
