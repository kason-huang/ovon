# 新场景数据集生成脚本使用说明

> **场景**: 00016-qk9eeNeR4vw
> **配置**: 8+ GPU, 完整生成覆盖率元数据
> **目标**: 快速验证 + 生成训练数据

---

## 📁 脚本文件

```
scripts/data_generator/
├── run_new_scene_pipeline.sh         # 主控脚本（完整流程）
├── quick_verify.sh                    # 快速验证脚本
├── generate_coverage_parallel.py     # 并行生成覆盖率元数据
├── generate_coverage_meta.py          # 单 GPU 生成覆盖率
└── README_NEW_SCENE.md                # 本文档
```

---

## 🚀 快速开始

### **方案 1: 一键运行（推荐）**

```bash
# 运行完整流程
cd /home/kason/workspace/ovon
bash scripts/data_generator/run_new_scene_pipeline.sh
```

**流程**:
1. ✅ 环境检查
2. ✅ 快速验证
3. ✅ 生成覆盖率元数据
4. ✅ 生成任务集
5. ✅ 验证输出

**时间**: 约 2-4 小时

---

### **方案 2: 分步执行**

#### **步骤 1: 快速验证** (30-60 分钟)

```bash
cd /home/kason/workspace/ovon
bash scripts/data_generator/quick_verify.sh
```

**目的**: 确保场景和代码正常工作

**输出**:
- `data/coverage_meta/train.pkl` (测试版)
- `data/datasets/ovon/hm3d/v1_stretch/train/content/00016-qk9eeNeR4vw.json.gz`

---

#### **步骤 2: 完整生成覆盖率元数据** (1-2 小时)

**单 GPU**:
```bash
cd /home/kason/workspace/ovon
python scripts/data_generator/generate_coverage_meta.py \
    --split train \
    --device-id 0
```

**多 GPU 并行** (推荐):
```bash
python scripts/data_generator/generate_coverage_parallel.py \
    --split train \
    --num-gpus 8 \
    --device-start-id 0
```

**输出**: `data/coverage_meta/train.pkl`

---

#### **步骤 3: 生成完整任务集** (30-60 分钟)

```bash
python ovon/dataset/objectnav_generator.py \
    --scene 00016-qk9eeNeR4vw \
    --split train \
    --start-poses-per-object 2000 \
    --episodes-per-object 0 \
    --device-id 0
```

**输出**: `data/datasets/ovon/hm3d/v1_stretch/train/content/00016-qk9eeNeR4vw.json.gz`

---

## 📊 脚本详解

### **1. run_new_scene_pipeline.sh**

**用途**: 一键运行完整流程

**特点**:
- ✅ 自动环境检查
- ✅ 分阶段执行
- ✅ 自动错误处理
- ✅ 彩色日志输出
- ✅ 进度跟踪

**配置**:
```bash
# 可在脚本中修改的参数
SCENE_NAME="00016-qk9eeNeR4vw"
SPLIT="train"
NUM_GPUS=8
START_POSES_PER_OBJECT=2000
```

---

### **2. quick_verify.sh**

**用途**: 快速验证场景和代码

**测试内容**:
1. 场景文件完整性
2. Habitat 能否加载场景
3. 物体数量统计
4. 生成测试覆盖率元数据
5. 生成测试任务集

**预期输出**:
```
✓ 场景文件存在
✓ 场景加载成功
  物体总数: 1523
  目标物体数: 45
✓ 覆盖率元数据生成完成
✓ 任务集生成完成
  片段数: 50
```

---

### **3. generate_coverage_parallel.py**

**用途**: 并行生成覆盖率元数据

**优势**:
- 🚀 多 GPU 并行，速度提升 8 倍
- 📊 实时进度显示
- 🔄 自动合并结果
- ✅ 错误处理

**参数**:
```bash
--split train                    # 数据集划分
--num-gpus 8                     # GPU 数量
--device-start-id 0              # 起始 GPU ID
--max-scenes 10                  # 最大场景数（可选）
```

---

## 🔍 验证输出

### **检查覆盖率元数据**

```bash
python -c "
import pickle
with open('data/coverage_meta/train.pkl', 'rb') as f:
    data = pickle.load(f)

print(f'类别数量: {len(data)}')
for cat, items in list(data.items())[:5]:
    print(f'{cat}: {len(items)} 个视点')
"
```

**预期输出**:
```
类别数量: 6
chair: 2340 个视点
bed: 1890 个视点
sofa: 2100 个视点
...
```

---

### **检查任务集**

```bash
python -c "
import gzip
import json

with gzip.open('data/datasets/ovon/hm3d/v1_stretch/train/content/00016-qk9eeNeR4vw.json.gz', 'rt') as f:
    data = json.load(f)

print(f'总片段数: {len(data[\"episodes\"])}')
print(f'类别: {list(data[\"goals_by_category\"].keys())}')

# 查看第一个片段
print('\\n第一个片段:')
print(data['episodes'][0])
"
```

**预期输出**:
```
总片段数: 1250
类别: ['00016-qk9eeNeR4vw_chair', '00016-qk9eeNeR4vw_bed']

第一个片段:
{
  "episode_id": "0",
  "scene_id": "data/scene_datasets/hm3d/train/00016-qk9eeNeR4vw",
  "object_category": "chair",
  "start_position": [x, y, z],
  ...
}
```

---

## ⚙️ 高级配置

### **调整质量参数**

```bash
# 快速测试（低质量）
--start-poses-per-object 100
--episodes-per-object 50

# 标准质量（推荐）
--start-poses-per-object 2000
--episodes-per-object 0

# 最高质量（论文用）
--start-poses-per-object 5000
--episodes-per-object 0
```

### **多场景批量生成**

```bash
# 修改场景列表
SCENES=(
    "00016-qk9eeNeR4vw"
    "00017-oEPjPNSPmzL"
    "00020-XYyR54sxe6b"
)

for scene in "${SCENES[@]}"; do
    python ovon/dataset/objectnav_generator.py \
        --scene "$scene" \
        --split train \
        --start-poses-per-object 2000
done
```

---

## 🐛 故障排除

### **Q1: 场景加载失败**

**错误**: `FileNotFoundError: 00016-qk9eeNeR4vw.basis.glb`

**解决**:
```bash
# 检查文件路径
ls -lh data/scene_datasets/hm3d/train/00016-qk9eeNeR4vw/

# 检查配置文件
grep "00016-qk9eeNeR4vw" data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json
```

---

### **Q2: GPU 内存不足**

**错误**: `CUDA out of memory`

**解决**:
```bash
# 减少并发任务
--tasks-per-gpu 1

# 或使用单个 GPU
python scripts/generate_coverage_meta.py --device-id 0
```

---

### **Q3: 没有找到目标物体**

**错误**: `Total objects post filtering: 0`

**解决**:
```bash
# 检查场景是否有语义标注
python -c "
import habitat_sim
cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_dataset_config_file = 'data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json'
cfg.scene_id = '00016-qk9eeNeR4vw'
cfg.gpu_device_id = 0

sim = habitat_sim.Simulator(cfg)
objects = sim.semantic_scene.objects
print(f'物体总数: {len(objects)}')

# 打印前 10 个物体的类别
for obj in objects[:10]:
    print(f'  {obj.category.name()}')
"
```

---

## 📈 性能优化

### **单 GPU vs 多 GPU**

| 配置 | 覆盖率生成 | 任务集生成 | 总时间 |
|------|-----------|-----------|--------|
| 1 GPU | 6-12 小时 | 12-24 小时 | 18-36 小时 |
| 8 GPU (串行) | 6-12 小时 | 12-24 小时 | 18-36 小时 |
| 8 GPU (并行) | 1-2 小时 ⚡ | 2-4 小时 ⚡ | 3-6 小时 🚀 |

### **推荐的并行策略**

```bash
# 阶段 2: 覆盖率元数据（并行）
python scripts/data_generator/generate_coverage_parallel.py \
    --num-gpus 8 \
    --device-start-id 0

# 阶段 3: 任务集（并行）
python ovon/dataset/objectnav_generator.py \
    --split train \
    --multiprocessing \
    --tasks-per-gpu 2
```

---

## 📚 相关文档

- [覆盖率元数据分析](../docs/data_generator/coverage_meta_analysis.md)
- [新数据集生成指南](../docs/NEW_DATASET_GENERATION_GUIDE.md)
- [运行机制说明](../docs/data_generator/objectnav_generator_running_guide.md)

---

## 🎯 总结

### **最简单的开始方式**

```bash
# 1. 快速验证（30-60 分钟）
bash scripts/data_generator/quick_verify.sh

# 2. 如果验证通过，运行完整流程（2-4 小时）
bash scripts/data_generator/run_new_scene_pipeline.sh
```

### **关键要点**

1. ✅ **场景已配置**: 00016-qk9eeNeR4vw 已在配置文件中
2. ✅ **无需修改代码**: 完整生成覆盖率元数据，代码无需改动
3. ✅ **并行加速**: 8 GPU 可将时间从 36 小时降到 6 小时
4. ✅ **分阶段执行**: 先验证，再生成，确保质量

---

**准备好开始了吗？** 运行 `bash scripts/data_generator/quick_verify.sh` 开始第一步！
