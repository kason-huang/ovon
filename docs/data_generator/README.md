# ObjectNav Generator Documentation

> **文件**: `ovon/dataset/objectnav_generator.py`
> **用途**: 从 HM3D 场景生成 Object Navigation 数据集
> **最后更新**: 2026-03-17

---

## 📚 文档导航

本目录包含 `objectnav_generator.py` 脚本的完整文档，涵盖不同层面的使用和分析需求。

### 1. [运行机制说明](./objectnav_generator_running_guide.md) ⭐ 推荐新手阅读

**适合人群**: 用户、开发者、运维人员

**内容概要**:
- 🚀 完整运行流程（命令行 → 输出）
- 🔄 两种运行模式（单场景 / 批量）
- 🖥️ 多进程并行执行详解
- 📦 单个场景处理流程
- 💡 典型使用场景示例
- 🛠️ 依赖检查和性能调优
- 🐛 常见问题和解决方案

**何时阅读**:
- ✅ 第一次使用脚本
- ✅ 需要快速上手
- ✅ 遇到运行问题
- ✅ 需要性能优化

---

### 2. [代码分析](./objectnav_generator_analysis.md)

**适合人群**: 开发者、代码审查人员

**内容概要**:
- 🏗️ 类结构和属性详解
- 🔧 核心方法说明（8个关键方法）
- 📊 数据流程图
- 🧠 算法亮点（视点质量、智能采样、开放词汇）
- 📝 代码级使用示例
- 🔍 调试与可视化技巧

**何时阅读**:
- ✅ 需要理解代码实现
- ✅ 进行代码审查或重构
- ✅ 添加新功能
- ✅ 调试内部逻辑

---

### 3. [关键流程实现分析](./key_implementation_analysis.md)

**适合人群**: 算法工程师、研究人员

**内容概要**:
- 🎯 4个核心算法的深度分析
  - 视点生成（网格搜索 + 三视角）
  - 导航网格聚类（层次聚类）
  - 起始位置采样（拒绝采样）
  - 测地距离计算（A* 搜索）
- 📐 复杂度分析
- 🔬 技术细节（为什么这样设计）
- ⚡ 性能优化建议
- 🎨 聚类 vs 随机采样对比

**何时阅读**:
- ✅ 研究算法实现细节
- ✅ 优化生成质量或速度
- ✅ 理解设计决策
- ✅ 改进现有算法

---

### 4. [覆盖率元数据分析](./coverage_meta_analysis.md) 🆕

**适合人群**: 使用新场景的用户、数据集生成者

**内容概要**:
- 🎯 覆盖率元数据的核心概念
- 📊 数据结构和统计信息
- 🔄 三阶段工作流程详解
- ❓ 为什么需要预生成？
- 🛠️ 4种实用方案对比
- 📈 性能和质量分析

**何时阅读**:
- ✅ 准备使用新场景生成数据集
- ✅ 想理解覆盖率过滤的原理
- ✅ 需要决定是否生成覆盖率元数据
- ✅ 遇到覆盖率相关错误

---

## 🎯 快速开始指南

### 第一次使用？

1. **阅读** [运行机制说明](./objectnav_generator_running_guide.md)
2. **运行** 快速测试命令
3. **检查** 输出文件格式

### 遇到问题？

1. **查看** [运行机制说明](./objectnav_generator_running_guide.md) 的"常见问题"部分
2. **检查** 依赖文件是否齐全
3. **参考** 典型使用场景

### 需要修改代码？

1. **阅读** [代码分析](./objectnav_generator_analysis.md)
2. **深入** [关键流程实现分析](./key_implementation_analysis.md)
3. **理解** 算法和数据结构

### 使用新场景生成数据集？

1. **阅读** [覆盖率元数据分析](./coverage_meta_analysis.md) 🆕
2. **参考** [新数据集生成指南](../NEW_DATASET_GENERATION_GUIDE.md)
3. **生成** 覆盖率元数据 + 任务集

---

## 📋 文档对比

| 文档 | 层次 | 读者 | 重点 |
|------|------|------|------|
| 运行机制说明 | 使用层 | 用户 | 如何运行 |
| 代码分析 | 实现层 | 开发者 | 如何实现 |
| 关键流程分析 | 算法层 | 研究人员 | 为什么这样 |
| 覆盖率元数据分析 | 数据层 | 数据生成者 | 质量控制 |

---

## 🔗 相关资源

### 代码文件
- **主脚本**: `ovon/dataset/objectnav_generator.py` (1172 行)
- **姿态采样**: `ovon/dataset/pose_sampler.py`
- **语义工具**: `ovon/dataset/semantic_utils.py`
- **数据集格式**: `ovon/dataset/ovon_dataset.py`

### 外部依赖
- [Habitat 仿真平台](https://aihabitat.org/)
- [HM3D 数据集](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md)
- [scikit-learn 聚类算法](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)

---

## 💻 快速命令参考

```bash
# 单场景测试
python ovon/dataset/objectnav_generator.py --scene scene0011 --split train

# 批量处理（多GPU）
python ovon/dataset/objectnav_generator.py --split train --multiprocessing --tasks-per-gpu 2

# 快速验证
python -c "import gzip, json; data=json.loads(gzip.open('output.json.gz').read()); print(f'{len(data[\"episodes\"])} episodes')"
```

---

## 📝 文档维护

- **创建日期**: 2026-03-17
- **维护者**: OVON 项目团队
- **反馈**: 请在 GitHub Issues 提出问题或建议

---

**提示**: 建议按照"运行机制说明 → 代码分析 → 关键流程分析"的顺序阅读，逐步深入理解脚本。