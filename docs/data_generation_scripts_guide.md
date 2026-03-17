# 数据生成脚本指南

> **快速链接**: [脚本目录](../../scripts/data_generator/)

---

## 🎯 概述

如果您有新的 HM3D 场景需要生成 OVON 数据集，可以使用自动化脚本完成。

**场景示例**: `00016-qk9eeNeR4vw`

---

## 📁 脚本位置

```
scripts/data_generator/
├── README.md                          # 脚本说明
├── README_NEW_SCENE.md                # 详细使用指南
├── run_new_scene_pipeline.sh         # 主控脚本
├── quick_verify.sh                    # 快速验证
├── generate_coverage_parallel.py     # 并行生成覆盖率
└── generate_coverage_meta.py          # 单 GPU 生成覆盖率
```

---

## 🚀 快速开始

### **从项目根目录运行**

```bash
cd /home/kason/workspace/ovon

# 方案 1: 快速验证（推荐）
bash scripts/data_generator/quick_verify.sh

# 方案 2: 完整流程
bash scripts/data_generator/run_new_scene_pipeline.sh
```

---

## 📖 完整文档

- [脚本 README](../../scripts/data_generator/README.md)
- [详细使用说明](../../scripts/data_generator/README_NEW_SCENE.md)
- [覆盖率元数据分析](./data_generator/coverage_meta_analysis.md)
- [新数据集生成指南](./NEW_DATASET_GENERATION_GUIDE.md)

---

## ⏱️ 时间估算（8 GPU 并行）

| 阶段 | 时间 | 说明 |
|------|------|------|
| 快速验证 | 30-60 分钟 | 单场景测试 |
| 生成覆盖率元数据 | 1-2 小时 | 8 GPU 并行 |
| 生成任务集 | 30-60 分钟 | 单场景 |
| **总计** | **2-4 小时** | ⚡ |

---

## ✅ 前提条件

1. ✅ 场景文件已在 `data/scene_datasets/hm3d/train/` 目录
2. ✅ 场景已添加到 `hm3d_annotated_basis.scene_dataset_config.json`
3. ✅ 有可用的 GPU（推荐 8+ 个）
4. ✅ 已安装所有依赖（habitat-sim, pytorch 等）

---

## 🔗 快速链接

| 需求 | 链接 |
|------|------|
| 查看脚本 | [scripts/data_generator/](../../scripts/data_generator/) |
| 快速验证 | [quick_verify.sh](../../scripts/data_generator/quick_verify.sh) |
| 完整流程 | [run_new_scene_pipeline.sh](../../scripts/data_generator/run_new_scene_pipeline.sh) |
| 使用说明 | [README_NEW_SCENE.md](../../scripts/data_generator/README_NEW_SCENE.md) |

---

## 📞 获取帮助

如果遇到问题，请查看：
1. [覆盖率元数据分析](./data_generator/coverage_meta_analysis.md) - 理解原理
2. [运行机制说明](./data_generator/objectnav_generator_running_guide.md) - 了解参数
3. [详细使用说明](../../scripts/data_generator/README_NEW_SCENE.md) - 查看示例
