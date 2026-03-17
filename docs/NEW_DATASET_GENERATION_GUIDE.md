# 用新场景生成 OVON 任务集 - 快速指南

## 📋 完整流程

```
1. 准备新场景
   ↓
2. 修改代码（支持 coverage_meta=None）
   ↓
3. 生成覆盖率元数据
   ↓
4. 生成任务集
   ↓
5. 验证输出
```

---

## 🔧 步骤 1: 准备新场景

### 1.1 场景文件结构

```
data/scene_datasets/hm3d/
├── train/
│   ├── your_scene_001.basis.glb  ← 新场景文件
│   ├── your_scene_002.basis.glb
│   └── ...
└── val/
    └── ...
```

### 1.2 更新场景配置文件

编辑 `data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json`:

```json
{
  "scene_dataset_config_version": "0.1.0",
  "scenes": {
    "train": {
      "your_scene_001": {
        "scene_path": "data/scene_datasets/hm3d/train/your_scene_001.basis.glb"
      }
    }
  }
}
```

### 1.3 验证场景

```bash
# 检查场景文件是否存在
ls -lh data/scene_datasets/hm3d/train/*.basis.glb

# 验证场景格式（需要 habitat-sim）
python -c "
import habitat_sim
sim = habitat_sim.Simulator(
    habitat_sim.Configuration([
        habitat_sim.SimulatorConfiguration()
    ])
)
print('Habitat-sim 已安装')
"
```

---

## 🛠️ 步骤 2: 修改代码

**必须修改** `ovon/dataset/semantic_utils.py` 以支持生成覆盖率。

### 快速应用补丁

```bash
# 备份原文件
cp ovon/dataset/semantic_utils.py ovon/dataset/semantic_utils.py.bak

# 应用修改（需要手动编辑或使用 patch）
# 参考: docs/COVERAGE_META_PATCH.md
```

**关键修改点** (第 48 行):

```python
@staticmethod
def load_categories(
    mapping_file: str,
    coverage_meta_file: str,
    frame_coverage_threshold: float,
    ...
) -> Dict[str, str]:
    # ← 添加这段代码
    if coverage_meta_file is None:
        print("Warning: No coverage filter, loading all categories")
        # ... (见完整补丁文件)
        return mapping

    # 原有代码...
```

---

## 🎯 步骤 3: 生成覆盖率元数据

### 3.1 快速测试（推荐先运行）

```bash
# 测试 1 个场景，确保代码正常工作
python scripts/generate_coverage_meta.py \
    --split train \
    --max-scenes 1 \
    --device-id 0
```

**预期输出**:
```
=== 生成覆盖率元数据 (train) ===
场景路径: data/scene_datasets/hm3d
物体类别: ['chair', 'bed', 'toilet', 'sofa', 'plant', 'tv_monitor']

[1/4] 获取场景列表...
找到 1 个场景

[2/4] 初始化生成器...

[3/4] 计算物体覆盖率...
处理场景: 100%|████████| 1/1

[4/4] 保存覆盖率元数据...
✅ 已保存到: data/coverage_meta/train.pkl

=== 统计信息 ===
chair: 45 个视点, 平均覆盖率: 0.125, 最小: 0.051, 最大: 0.234
...
```

### 3.2 生成完整覆盖率元数据

```bash
# 生成 train 集
python scripts/generate_coverage_meta.py \
    --split train \
    --device-id 0

# 生成 val 集
python scripts/generate_coverage_meta.py \
    --split val \
    --device-id 0
```

**⏱️ 预计时间**:
- 1 个场景: ~5-10 分钟
- 81 个场景: ~6-12 小时
- 取决于硬件配置

### 3.3 验证覆盖率元数据

```bash
# 检查文件
ls -lh data/coverage_meta/*.pkl

# 验证内容
python -c "
import pickle
import numpy as np

for split in ['train', 'val']:
    print(f'\n=== {split}.pkl ===')
    with open(f'data/coverage_meta/{split}.pkl', 'rb') as f:
        data = pickle.load(f)

    print(f'类别数量: {len(data)}')
    for cat, items in list(data.items())[:3]:
        coverages = [c[0] for c in items]
        print(f'  {cat}: {len(items)} 个视点, '
              f'平均覆盖率: {np.mean(coverages):.3f}')
"
```

---

## 🚀 步骤 4: 生成任务集

### 4.1 单场景测试

```bash
python ovon/dataset/objectnav_generator.py \
    --scene your_scene_001 \
    --split train \
    --start-poses-per-object 100 \
    --episodes-per-object 50
```

**输出位置**: `data/datasets/ovon/hm3d/v1_stretch/train/content/your_scene_001.json.gz`

### 4.2 批量生成所有场景

```bash
# 生成所有场景
python ovon/dataset/objectnav_generator.py \
    --split train \
    --output-path data/datasets/ovon/hm3d/v1_stretch/ \
    --start-poses-per-object 2000 \
    --episodes-per-object 0  # 0 = 保留所有
```

### 4.3 多 GPU 加速

```bash
# 启用多进程
python ovon/dataset/objectnav_generator.py \
    --split train \
    --multiprocessing \
    --tasks-per-gpu 2 \
    --start-poses-per-object 2000
```

### 4.4 验证任务集

```bash
# 检查生成的文件
ls -lh data/datasets/ovon/hm3d/v1_stretch/train/content/*.json.gz

# 统计任务数量
python -c "
import gzip
import json

for scene_file in ['your_scene_001.json.gz']:
    with gzip.open(f'data/datasets/ovon/hm3d/v1_stretch/train/content/{scene_file}', 'rt') as f:
        data = json.load(f)

    print(f'{scene_file}:')
    print(f'  总片段数: {len(data[\"episodes\"])}')
    print(f'  类别: {list(data[\"goals_by_category\"].keys())}')
"
```

---

## 📊 步骤 5: 使用新任务集

### 5.1 训练配置

更新实验配置文件 `config/experiments/your_experiment.yaml`:

```yaml
habitat:
  task:
    lab_sensors:
      clip_objectgoal_sensor:
        categories: ["chair", "bed", "toilet", "sofa", "plant", "tv_monitor"]
    measurements:
      ovon_object_goal_id:
        cache: "data/clip_embeddings/your_cache.pkl"

dataset:
  data_path: "data/datasets/ovon/hm3d"
  split: "train"
  content_scenes_path: "{data_path}/v1_stretch/{split}/content/{scene}.json.gz"
```

### 5.2 开始训练

```bash
python -m ovon.run \
    --run-type train \
    --exp-config config/experiments/your_experiment.yaml
```

---

## ⚠️ 常见问题

### Q1: 场景加载失败

**错误**: `FileNotFoundError: your_scene_001.basis.glb`

**解决**:
```bash
# 检查文件路径
ls data/scene_datasets/hm3d/train/your_scene_001.basis.glb

# 检查配置文件
cat data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json
```

### Q2: 没有找到任何物体

**错误**: `Total objects post filtering: 0`

**解决**:
```bash
# 检查场景是否有语义标注
python -c "
import habitat_sim
cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_dataset_config_file = 'data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json'
cfg.scene_id = 'your_scene_001'
sim = habitat_sim.Simulator(cfg)
print(f'物体数量: {len(sim.semantic_scene.objects)}')
"
```

### Q3: GPU 内存不足

**错误**: `CUDA out of memory`

**解决**:
```bash
# 减少并发任务
--tasks-per-gpu 1

# 或减少场景数量
--num-scenes 5
```

---

## 📈 性能优化

### 快速测试（低质量）

```bash
--start-poses-per-object 100 \
--episodes-per-object 10
```

### 生产环境（高质量）

```bash
--start-poses-per-object 5000 \
--episodes-per-object -1  # 保留所有
```

### 平衡配置（推荐）

```bash
--start-poses-per-object 2000 \
--episodes-per-object 0
```

---

## 📚 相关文档

- [运行机制说明](docs/data_generator/objectnav_generator_running_guide.md)
- [代码分析](docs/data_generator/objectnav_generator_analysis.md)
- [覆盖率补丁说明](docs/COVERAGE_META_PATCH.md)

---

## 🎯 总结

1. ✅ 准备场景文件（HM3D 格式）
2. ✅ 修改代码支持 `coverage_meta=None`
3. ✅ 生成覆盖率元数据（必须）
4. ✅ 生成任务集
5. ✅ 验证和使用

**关键要点**: 覆盖率元数据是**必需的中间文件**，必须先生成它才能生成任务集。
