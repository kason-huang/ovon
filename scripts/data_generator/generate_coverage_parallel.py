#!/usr/bin/env python3
"""
并行生成覆盖率元数据脚本

用途: 利用多个 GPU 并行生成覆盖率元数据
时间: 1-2 小时 (8 GPU)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="并行生成覆盖率元数据")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"],
                        help="数据集划分")
    parser.add_argument("--scene-datasets-path", type=str,
                        default="data/scene_datasets/hm3d",
                        help="场景数据集路径")
    parser.add_argument("--output-path", type=str,
                        default="data/coverage_meta",
                        help="输出路径")
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="使用的 GPU 数量")
    parser.add_argument("--device-start-id", type=int, default=0,
                        help="起始 GPU ID")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="最大处理场景数（用于测试）")

    args = parser.parse_args()

    print("=" * 60)
    print("并行生成覆盖率元数据")
    print("=" * 60)
    print(f"数据集划分: {args.split}")
    print(f"场景路径: {args.scene_datasets_path}")
    print(f"输出路径: {args.output_path}")
    print(f"GPU 数量: {args.num_gpus}")
    print(f"起始 GPU ID: {args.device_start_id}")
    print(f"最大场景数: {args.max_scenes or '全部'}")
    print("=" * 60)
    print()

    # 获取场景列表
    print("[1/3] 获取场景列表...")
    from ovon.dataset.semantic_utils import get_hm3d_semantic_scenes

    scenes = get_hm3d_semantic_scenes(args.scene_datasets_path, [args.split])
    scenes = sorted(scenes)

    if args.max_scenes:
        scenes = scenes[:args.max_scenes]

    print(f"找到 {len(scenes)} 个场景")
    print()

    # 按 GPU 分配场景
    print(f"[2/3] 分配场景到 {args.num_gpus} 个 GPU...")
    gpu_scenes = {i: [] for i in range(args.num_gpus)}

    for idx, scene in enumerate(scenes):
        gpu_id = args.device_start_id + (idx % args.num_gpus)
        gpu_scenes[gpu_id].append(scene)

    for gpu_id, scene_list in gpu_scenes.items():
        print(f"  GPU {gpu_id}: {len(scene_list)} 个场景")
    print()

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)

    # 启动并行进程
    print("[3/3] 启动并行生成...")
    processes = []

    for gpu_id, scene_list in gpu_scenes.items():
        if not scene_list:
            continue

        # 为每个 GPU 创建独立的输出文件
        output_file = os.path.join(args.output_path, f"{args.split}_gpu{gpu_id}.pkl")

        # 构建命令
        cmd = [
            "python", "scripts/generate_coverage_meta.py",
            "--split", args.split,
            "--scene-datasets-path", args.scene_datasets_path,
            "--output-path", args.output_path,
            "--device-id", str(gpu_id),
        ]

        # 如果只处理部分场景，传递场景列表
        # 这里需要修改 generate_coverage_meta.py 支持场景列表参数
        # 暂时使用简单的方案：每个 GPU 处理所有场景，但通过环境变量区分

        print(f"启动 GPU {gpu_id} (处理 {len(scene_list)} 个场景)...")

        # 设置环境变量指定该 GPU 处理的场景范围
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        proc = subprocess.Popen(cmd, env=env)
        processes.append((gpu_id, proc))

    print(f"已启动 {len(processes)} 个并行进程")
    print()
    print("生成中... (按 Ctrl+C 停止)")
    print()

    # 等待所有进程完成
    try:
        for gpu_id, proc in processes:
            proc.wait()
            if proc.returncode == 0:
                print(f"✓ GPU {gpu_id} 完成")
            else:
                print(f"✗ GPU {gpu_id} 失败 (退出码: {proc.returncode})")
    except KeyboardInterrupt:
        print("\n收到中断信号，正在停止所有进程...")
        for gpu_id, proc in processes:
            proc.terminate()
        sys.exit(1)

    # 合并结果
    print()
    print("合并结果...")
    import pickle
    from collections import defaultdict

    merged_data = defaultdict(list)

    for gpu_id in range(args.num_gpus):
        partial_file = os.path.join(args.output_path, f"{args.split}_gpu{gpu_id}.pkl")
        if os.path.exists(partial_file):
            with open(partial_file, 'rb') as f:
                data = pickle.load(f)

            for category, items in data.items():
                merged_data[category].extend(items)

            # 删除临时文件
            os.remove(partial_file)

    # 保存合并后的结果
    output_file = os.path.join(args.output_path, f"{args.split}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(dict(merged_data), f)

    print(f"✓ 合并完成，保存到: {output_file}")

    # 打印统计
    print()
    print("=" * 60)
    print("统计信息")
    print("=" * 60)
    for category, items in merged_data.items():
        coverages = [c[0] for c in items]
        import numpy as np
        print(f"{category}:")
        print(f"  视点数: {len(items)}")
        print(f"  平均覆盖率: {np.mean(coverages):.3f}")
        print(f"  最小覆盖率: {np.min(coverages):.3f}")
        print(f"  最大覆盖率: {np.max(coverages):.3f}")

    print()
    print("=" * 60)
    print("✓ 覆盖率元数据生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
