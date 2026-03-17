# `objectnav_generator.py` 脚本运行机制说明

> **文件**: `ovon/dataset/objectnav_generator.py`
> **目的**: 从 HM3D 场景生成 Object Navigation 数据集

---

## 📋 运行流程概览

```
用户运行脚本
    ↓
1. 解析命令行参数
    ↓
2. 确定要处理的场景列表
    ↓
3. 对每个场景调用 make_episodes_for_scene()
    ├─ 配置仿真环境
    ├─ 过滤物体
    ├─ 生成视点
    ├─ 采样起始位置
    └─ 创建数据集片段
    ↓
4. 保存为 JSON.gz 文件
```

---

## 🚀 详细运行步骤

### 步骤 1: 命令行参数解析 (行 1088-1140)

```python
python ovon/dataset/objectnav_generator.py [参数]
```

**关键参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--scene` | None | 指定单个场景（如 `scene0011`） |
| `--split` | "train" | 数据集划分 (train/val_unseen_easy/val_unseen_hard) |
| `--output-path` | "data/datasets/ovon/hm3d/v1_stretch/" | 输出目录 |
| `--num-scenes` | -1 (全部) | 处理场景数量 |
| `--start-poses-per-object` | 2000 | 每个物体的起始姿态数 |
| `--episodes-per-object` | 0 (全部) | 每个物体保留的片段数 |
| `--multiprocessing` | False | 启用多进程 |
| `--tasks-per-gpu` | 1 | 每个 GPU 的任务数 |
| `--disable-euc-geo-ratio-check` | False | 禁用距离比率检查 |
| `--disable-wordnet-mapping` | False | 禁用 WordNet 映射 |

### 步骤 2: 确定场景列表 (行 1141-1153)

```python
if args.scene is not None:
    # 模式 A: 处理单个场景
    scene_id = args.scene.split(".")[0] + ".basis.glb"
    scenes = [scene_id]
else:
    # 模式 B: 处理整个 split 的所有场景
    split = args.split.split("_")[0]  # "train", "val"
    scenes = get_hm3d_semantic_scenes("data/scene_datasets/hm3d", [split])
    scenes = sorted(scenes)

if args.num_scenes > 0:
    scenes = scenes[:args.num_scenes]  # 限制场景数量
```

**场景文件路径**: `data/scene_datasets/hm3d/{split}/{scene_name}.basis.glb`

### 步骤 3: 批量处理场景 (行 1162-1172)

```python
outpath = os.path.join(args.output_path, f"{args.split}/content/")
make_episodes_for_split(
    scenes,                          # 场景列表
    args.split,                      # 数据集划分
    outpath,                         # 输出路径
    args.tasks_per_gpu,              # 每个 GPU 的任务数
    args.enable_multiprocessing,     # 多进程标志
    args.start_poses_per_object,     # 起始姿态数
    args.episodes_per_object,        # 片段数
    args.disable_euc_to_geo_ratio_check,
    args.disable_wordnet_mapping,
)
```

---

## 🔄 两种运行模式

### 模式 A: 单场景处理

```bash
python ovon/dataset/objectnav_generator.py \
    --scene scene0011 \
    --split train
```

**流程**:
1. 直接处理 `data/scene_datasets/hm3d/train/scene0011.basis.glb`
2. 输出到 `data/datasets/ovon/hm3d/v1_stretch/train/content/scene0011.json.gz`

### 模式 B: 批量处理

```bash
python ovon/dataset/objectnav_generator.py \
    --split train \
    --num-scenes 10 \
    --multiprocessing
```

**流程**:
1. 从 `data/scene_datasets/hm3d/train/` 加载所有场景
2. 限制为前 10 个场景
3. 多进程并行处理
4. 每个场景生成独立的 `.json.gz` 文件

---

## 🖥️ 多进程执行流程

当 `--multiprocessing` 启用时：

```
主进程
    ↓
1. 检测可用 GPU (GPUtil)
    gpus = len(GPUtil.getAvailable(limit=256))
    cpu_threads = gpus * 16
    ↓
2. 创建任务列表
    for i, scene in enumerate(scenes):
        deviceId = i % gpus  # 轮询分配 GPU
        items.append((scene, outpath, deviceId, ...))
    ↓
3. 创建进程池 (forkserver 模式)
    with mp_ctx.Pool(cpu_threads) as pool:
        for _ in pool.imap_unordered(make_episodes_for_scene, items):
            pbar.update()
    ↓
4. 并行处理
    CPU Thread 1 → GPU 0 → Scene 1
    CPU Thread 2 → GPU 0 → Scene 2
    CPU Thread 3 → GPU 1 → Scene 3
    ...
```

**性能**: 8 GPU × 16 threads = 128 并发任务

---

## 📦 单个场景处理流程 (`make_episodes_for_scene`)

```python
def make_episodes_for_scene(args):
    scene, outpath, device_id, split, ... = args

    # 1. 检查是否已处理
    if os.path.exists(f"{outpath}/{scene_name}.json.gz"):
        print(f"Skipping scene: {scene}")
        return

    # 2. 加载类别配置
    categories = load_json("data/hm3d_meta/ovon_categories.json")
    if split in ["val_unseen_easy", "val_unseen_hard"]:
        categories = load_json("data/hm3d_meta/ovon_val_splits.json")

    # 3. 创建 ObjectGoalGenerator
    objectgoal_maker = ObjectGoalGenerator(
        semantic_spec_filepath="data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
        img_size=(512, 512),
        hfov=90,
        agent_height=1.41,
        agent_radius=0.17,
        sensor_height=1.31,
        pose_sampler_args={...},
        mapping_file="ovon/dataset/source_data/Mp3d_category_mapping.tsv",
        categories=categories[split],
        coverage_meta_file=f"data/coverage_meta/{split}.pkl",
        frame_cov_thresh=0.05,
        goal_vp_cell_size=0.25,
        goal_vp_max_dist=1.0,
        start_poses_per_obj=start_poses_per_object,
        device_id=device_id,
        sample_dense_viewpoints=True,
        ...
    )

    # 4. 生成物体目标
    object_goals = objectgoal_maker.make_object_goals(
        scene=scene,
        with_viewpoints=True,
        with_start_poses=True
    )

    # 5. 创建数据集片段
    episode_dataset = objectgoal_maker.make_episodes(
        object_goals,
        scene,
        episodes_per_object=episodes_per_object,
        split=split,
    )

    # 6. 保存到磁盘
    save_to = f"{outpath}/{scene_name}.json.gz"
    objectgoal_maker.save_to_disk(episode_dataset, save_to)
    print(f"Total episodes: {len(episode_dataset.episodes)}")
```

---

## 🔍 核心生成流程 (`make_object_goals`)

```
配置 Habitat 仿真
    ↓
加载场景语义数据
    ↓
过滤物体（根据 OVON 类别白名单）
    ↓
对每个物体:
    ├─ 径向采样姿态 (PoseSampler)
    ├─ 渲染观察
    ├─ 过滤可见物体的观察
    ├─ 计算帧覆盖率
    ├─ 应用阈值过滤
    └─ 生成密集视点网格
        ├─ 创建 2D 网格
        ├─ 验证导航可行性
        ├─ 计算三视角覆盖率
        └─ 过滤低覆盖率视点
    ↓
可选: 导航网格聚类
    ├─ 表面采样 (25 点/m²)
    ├─ 层次聚类 (threshold=1.0m)
    └─ 计算聚类到物体的距离
    ↓
采样起始位置
    ├─ 随机采样 或 基于聚类采样
    ├─ 验证约束:
    │   ├─ 孤立区域检查 (radius ≥ 1.5m)
    │   ├─ 距离检查 (1-30m)
    │   ├─ Geo/Euc 比率 (≥ 1.05)
    │   └─ 单楼层检查 (Δh ≤ 0.25m)
    └─ 生成随机旋转
    ↓
合并 WordNet 子类别
    ↓
返回所有目标
```

---

## 📊 输出格式

### 单个场景文件: `{scene_name}.json.gz`

```json
{
  "episodes": [
    {
      "episode_id": "0",
      "scene_id": "data/scene_datasets/hm3d/train/scene0011.basis.glb",
      "object_category": "chair",
      "children_object_categories": ["armchair", "sofa"],
      "start_position": [x, y, z],
      "start_rotation": [w, x, y, z],
      "info": {
        "geodesic_distance": 5.23,
        "euclidean_distance": 3.87
      }
    },
    ...
  ],
  "goals_by_category": {
    "scene0011_chair": [
      {
        "object_category": "chair",
        "object_id": "obj_123",
        "position": [x, y, z],
        "view_points": [
          {
            "agent_state": {
              "position": [x, y, z],
              "rotation": [w, x, y, z]
            },
            "iou": 0.23
          },
          ...
        ]
      },
      ...
    ]
  }
}
```

### 汇总文件: `{split}.json.gz`

```json
{
  "episodes": [],
  "goals_by_category": {}
}
```

**注意**: 汇总文件在开始时创建为空，后续通过合并各场景文件填充。

---

## 💡 典型使用场景

### 场景 1: 生成完整训练集

```bash
python ovon/dataset/objectnav_generator.py \
    --split train \
    --output-path data/datasets/ovon/hm3d/v1_stretch/ \
    --multiprocessing \
    --tasks-per-gpu 2
```

**预期输出**:
- 81 个场景文件
- 每个 `~10-100MB`
- 总计 `~5-10GB`
- 处理时间: 数小时（取决于硬件）

### 场景 2: 快速测试

```bash
python ovon/dataset/objectnav_generator.py \
    --scene scene0011 \
    --split train \
    --start-poses-per-object 100 \
    --episodes-per-object 50
```

**预期输出**:
- 1 个场景文件
- `~5-10MB`
- 处理时间: ~5-10 分钟

### 场景 3: 生成验证集

```bash
python ovon/dataset/objectnav_generator.py \
    --split val_unseen_easy \
    --episodes-per-object 100 \
    --disable-wordnet-mapping
```

**预期输出**:
- 11 个场景文件
- 每个物体 100 个片段
- 禁用 WordNet（更严格的类别匹配）

---

## 🛠️ 依赖检查

脚本运行前需要确保：

```bash
# 1. HM3D 场景文件
ls data/scene_datasets/hm3d/train/*.basis.glb

# 2. 类别配置
ls data/hm3d_meta/ovon_categories.json
ls data/hm3d_meta/ovon_val_splits.json

# 3. WordNet 映射
ls data/wordnet/wordnet_mapping.json

# 4. 覆盖率元数据
ls data/coverage_meta/train.pkl
ls data/coverage_meta/val.pkl

# 5. 场景配置
ls data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json

# 6. 类别映射
ls ovon/dataset/source_data/Mp3d_category_mapping.tsv
```

---

## ⚙️ 性能调优

### GPU 内存优化

```bash
# 减少 batch size（如果 OOM）
--tasks-per-gpu 1

# 或减少场景数量
--num-scenes 5
```

### 速度优化

```bash
# 启用多进程
--multiprocessing

# 增加每个 GPU 的任务数
--tasks-per-gpu 2

# 减少起始姿态数（快速测试）
--start-poses-per-object 500
```

### 质量优化

```bash
# 增加起始姿态数（更高质量）
--start-poses-per-object 5000

# 保留所有片段
--episodes-per-object -1

# 启用严格的距离比率检查
# (默认启用，不要禁用)
```

---

## 🐛 常见问题

### Q1: 跳过已存在的场景？

**现象**: `Skipping scene: scene0011`

**原因**: 输出文件已存在

**解决**:
```bash
rm data/datasets/ovon/hm3d/v1_stretch/train/content/scene0011.json.gz
```

### Q2: GPU 内存不足？

**现象**: `CUDA out of memory`

**解决**:
```bash
--tasks-per-gpu 1  # 减少 GPU 上的并发任务
```

### Q3: 没有找到有效起始位置？

**现象**: `Start poses none for: chair`

**原因**: 约束太严格或物体位置不合理

**解决**:
```bash
--disable-euc-geo-ratio-check  # 放宽约束
--start-poses-per-object 5000  # 增加尝试次数
```

### Q4: 找不到场景文件？

**现象**: `FileNotFoundError: scene0011.basis.glb`

**原因**: 场景文件路径不正确

**解决**:
```bash
# 检查路径
ls data/scene_datasets/hm3d/train/

# 使用完整场景名称
--scene scene0011.basis.glb
```

---

## 📈 监控进度

### 多进程模式

```bash
# 进度条自动显示
Processing scenes: [████████████████████] 45/81
```

### 单进程模式

```bash
# 手动检查输出文件
ls -lh data/datasets/ovon/hm3d/v1_stretch/train/content/*.json.gz | wc -l
```

### 日志分析

```bash
# 查看生成的片段数量
grep "Total episodes:" log.txt
```

---

## 🎯 总结

**脚本运行的核心逻辑**:

1. **输入**: HM3D 场景文件 (`.basis.glb`)
2. **处理**: 生成视点 + 采样起始位置
3. **约束**: 多重质量检查
4. **输出**: JSON.gz 格式的数据集片段

**关键特点**:

- ✅ **模块化**: 每个场景独立处理
- ✅ **可扩展**: 支持多 GPU 并行
- ✅ **可恢复**: 跳过已处理的场景
- ✅ **可配置**: 丰富的命令行参数

**典型工作流**:

```bash
# 1. 测试单个场景
python ovon/dataset/objectnav_generator.py --scene scene0011

# 2. 批量处理
python ovon/dataset/objectnav_generator.py --split train --multiprocessing

# 3. 验证输出
python -c "import gzip; import json; data = json.loads(gzip.open('output.json.gz').read()); print(f'Episodes: {len(data[\"episodes\"])}')"
```

---

**说明完成！** 这个脚本是 OVON 数据集生成的核心工具，通过 Habitat 仿真平台自动创建高质量的物体导航任务。