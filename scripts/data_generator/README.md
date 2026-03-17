# 数据生成脚本

本目录包含用于从新场景生成 OVON 数据集的脚本。

---

## 📁 文件结构

```
scripts/data_generator/
├── README.md                          # 本文档
├── README_NEW_SCENE.md                # 详细使用说明
├── run_new_scene_pipeline.sh         # 🎯 主控脚本（完整流程）
├── quick_verify.sh                    # ⚡ 快速验证脚本
├── generate_coverage_parallel.py     # 🚀 并行生成覆盖率元数据
└── generate_coverage_meta.py          # 📊 单 GPU 生成覆盖率
```

---

## 🚀 快速开始

### **从项目根目录运行**

所有脚本都需要从项目根目录 (`/home/kason/workspace/ovon`) 运行：

```bash
# 1. 进入项目根目录
cd /home/kason/workspace/ovon

# 2. 快速验证（推荐先运行）
bash scripts/data_generator/quick_verify.sh

# 3. 如果验证通过，运行完整流程
bash scripts/data_generator/run_new_scene_pipeline.sh
```

---

## 📖 详细文档

查看完整使用说明：[README_NEW_SCENE.md](./README_NEW_SCENE.md)

---

## 🔧 脚本说明

| 脚本 | 用途 | 时间 |
|------|------|------|
| **quick_verify.sh** | 快速验证场景和代码 | 30-60 分钟 |
| **run_new_scene_pipeline.sh** | 一键运行完整流程 | 2-4 小时 (8 GPU) |
| **generate_coverage_meta.py** | 生成覆盖率元数据（单 GPU） | 6-12 小时 |
| **generate_coverage_parallel.py** | 并行生成覆盖率元数据（多 GPU） | 1-2 小时 (8 GPU) |

---

## 📊 工作流程

```
1. 快速验证 (quick_verify.sh)
   ↓
2. 生成覆盖率元数据 (generate_coverage_meta.py)
   ↓
3. 生成任务集 (objectnav_generator.py)
   ↓
4. 验证输出
```

---

## 🎯 适用场景

- ✅ 使用新 HM3D 场景生成数据集
- ✅ 验证场景文件格式正确性
- ✅ 生成高质量的训练数据
- ✅ 并行加速（支持多 GPU）

---

## 🔗 相关文档

- [覆盖率元数据分析](../../docs/data_generator/coverage_meta_analysis.md)
- [新数据集生成指南](../../docs/NEW_DATASET_GENERATION_GUIDE.md)
- [运行机制说明](../../docs/data_generator/objectnav_generator_running_guide.md)

---

## ⚙️ 配置要求

- **场景格式**: HM3D (`.basis.glb`)
- **依赖**: Habitat-sim, PyTorch
- **GPU**: 推荐 8+ GPU（可并行加速）
- **磁盘**: 约 10-100GB（取决于场景数量）

---

## 📝 注意事项

1. **从项目根目录运行**: 所有脚本都需要从 `/home/kason/workspace/ovon` 运行
2. **场景已配置**: 确保 `data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json` 中包含您的场景
3. **分阶段执行**: 建议先运行 `quick_verify.sh` 验证，再运行完整流程

---

## 🆘 获取帮助

查看详细使用说明：
```bash
cat scripts/data_generator/README_NEW_SCENE.md
```
