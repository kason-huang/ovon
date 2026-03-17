# `objectnav_generator.py` 代码分析

> **生成时间**: 2026-03-17
> **文件路径**: `ovon/dataset/objectnav_generator.py`
> **用途**: Object Navigation 数据集生成器

---

## 📋 概述

这是一个用于生成 **Object Navigation** 数据集的 Python 模块，主要用于具身 AI 机器人导航任务。该代码基于 Habitat 仿真平台，在 HM3D (Habitat-Matterport 3D) 场景中生成物体目标导航任务。

**核心功能**：
1. 在 3D 仿真环境中为物体生成可观察的视点（viewpoints）
2. 采样合理的起始位置和旋转角度
3. 计算测地距离和欧氏距离
4. 创建训练/验证数据集片段

---

## 🏗️ 类结构

### `ObjectGoalGenerator` 类

#### 类属性

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ISLAND_RADIUS_LIMIT` | float | 1.5 | 限制孤立导航区域的最小半径（米） |
| `semantic_spec_filepath` | str | - | 语义场景配置文件路径 |
| `img_size` | Tuple[int, int] | - | 传感器图像尺寸 |
| `agent_height` | float | 1.41 | 智能体高度（米） |
| `agent_radius` | float | 0.17 | 智能体半径（米） |
| `hfov` | float | 90 | 水平视场角（度） |
| `goal_vp_cell_size` | float | 0.1 | 视点网格单元大小（米） |
| `goal_vp_max_dist` | float | 1.0 | 视点最大距离（米） |
| `start_poses_per_obj` | int | 500 | 每个物体的起始姿态数量 |
| `start_distance_limits` | Tuple[float, float] | (1.0, 30.0) | 起始位置距离限制（米） |
| `min_geo_to_euc_ratio` | float | 1.05 | 最小测地/欧氏距离比率 |

#### 初始化参数

```python
def __init__(
    self,
    semantic_spec_filepath: str,          # 语义场景配置文件
    img_size: Tuple[int, int],            # 图像尺寸
    hfov: float,                          # 水平视场角
    agent_height: float,                  # 智能体高度
    agent_radius: float,                  # 智能体半径
    sensor_height: float,                 # 传感器高度
    pose_sampler_args: Dict[str, Any],    # 姿态采样器参数
    mapping_file: str,                    # 类别映射文件
    categories: List[str],                # 允许的物体类别
    coverage_meta_file: str,              # 覆盖率元数据文件
    frame_cov_thresh: Tuple[float, float], # 帧覆盖率阈值
    ...
)
```

---

## 🔧 核心方法

### 1. `_config_sim(scene: str) -> Simulator`

**功能**: 配置 Habitat 仿真环境

**关键操作**:
- 配置 GPU 设备 ID
- 创建三个传感器：RGB、深度、语义
- 设置智能体动作空间（look_up, look_down）
- 重新计算导航网格（navmesh）

**代码位置**: 行 131-188

```python
# 关键配置
navmesh_settings.agent_height = self.agent_height      # 1.41m
navmesh_settings.agent_radius = self.agent_radius      # 0.17m
navmesh_settings.agent_max_climb = 0.10                # 10cm
```

---

### 2. `_make_object_viewpoints(sim, obj) -> List[Dict]`

**功能**: 为物体生成密集的候选观察点

**算法流程**:
1. 在物体 AABB 周围创建 2D 网格
2. 对每个网格点：
   - 检查是否在最大距离内
   - 检查导航可行性（向下搜索 2 米）
   - 计算面向物体的旋转
   - 渲染三个视角（look_down, look_up, look_up）
   - 计算帧覆盖率
3. 过滤并排序视点

**代码位置**: 行 199-287

**关键参数**:
- `goal_vp_cell_size`: 网格密度（默认 0.25m）
- `goal_vp_max_dist`: 最大观察距离（默认 1.0m）
- `frame_cov_thresh`: 覆盖率阈值（默认 0.05）

---

### 3. `_sample_start_poses(sim, goals) -> Tuple[List, List]`

**功能**: 随机采样起始位置（全局方法）

**约束条件**:
1. ✓ 距离限制：1.0m ~ 30.0m
2. ✓ 测地/欧氏距离比率 ≥ 1.05
3. ✓ 路径上所有点在同一楼层（高度差 < 0.25m）
4. ✓ 非孤立区域（island_radius ≥ 1.5m）
5. ✓ 距离比率采样率：`20 * (ratio - 0.98)²`

**代码位置**: 行 434-533

**返回值**:
```python
(start_positions, start_rotations, geodesic_dists, euclidean_dists)
```

---

### 4. `_sample_start_poses_wrt_clusters(sim, goals, cluster_centers, distance_to_clusters)`

**功能**: 基于导航网格聚类采样起始位置

**优势**:
- 更均匀的空间分布
- 避免集中在某些区域
- 支持更灵活的采样策略

**算法流程**:
1. 过滤有效聚类（距离在限制范围内）
2. 将片段分配到各聚类
   - 如果 `NC ≤ start_poses_per_obj`: 均匀分配
   - 如果 `NC > start_poses_per_obj`: 随机选择聚类
3. 在每个聚类中心附近采样

**代码位置**: 行 289-432

---

### 5. `_cluster_navmesh(sim, goals_by_category, scene_name)`

**功能**: 对导航网格进行层次聚类

**算法**:
```python
from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering(
    n_clusters=None,
    affinity="euclidean",
    distance_threshold=1.0,  # 1米聚类阈值
).fit(navmesh_pc)
```

**输出**:
- `cluster_infos`: 每个聚类的中心、位置、标准差
- `goal_category_to_cluster_distances`: 物体到聚类的测地距离

**代码位置**: 行 612-671

---

### 6. `_make_goal(sim, pose_sampler, obj, with_viewpoints, with_start_poses)`

**功能**: 为单个物体生成目标

**流程**:
1. 径向采样姿态
2. 渲染观察
3. 过滤可见物体的观察
4. 计算帧覆盖率
5. 应用阈值过滤
6. 生成视点（密集或稀疏）

**代码位置**: 行 554-600

---

### 7. `make_object_goals(scene, with_viewpoints, with_start_poses)`

**功能**: 为场景中所有物体生成目标

**流程**:
```
配置仿真
  ↓
过滤物体（根据类别映射）
  ↓
为每个物体生成目标
  ↓
可选：聚类导航网格
  ↓
合并 WordNet 子类别
  ↓
采样起始位置
  ↓
返回所有目标
```

**代码位置**: 行 673-765

---

### 8. `make_episodes(object_goals, scene, episodes_per_object, split)`

**功能**: 创建数据集片段

**每个片段包含**:
- `episode_id`: 片段唯一标识
- `scene_id`: 场景 ID
- `object_category`: 物体类别
- `start_position`: 起始位置 [x, y, z]
- `start_rotation`: 起始旋转 [w, x, y, z]
- `info`: 包含测地距离和欧氏距离
- `children_object_categories`: WordNet 子类别

**代码位置**: 行 856-927

---

## 🚀 主要函数

### `make_episodes_for_scene(args)`

**功能**: 处理单个场景并生成片段数据

**参数**:
```python
(
    scene,                           # 场景文件路径
    outpath,                         # 输出路径
    device_id,                       # GPU 设备 ID
    split,                           # 数据集划分
    start_poses_per_object,          # 起始姿态数量
    episodes_per_object,             # 片段数量
    disable_euc_to_geo_ratio_check,  # 禁用比率检查
    disable_wordnet_label,           # 禁用 WordNet
)
```

**输出**: `{scene_name}.json.gz`

**代码位置**: 行 930-1014

---

### `make_episodes_for_split(...)`

**功能**: 批量处理多个场景

**特性**:
- 支持多 GPU 并行处理
- 可选多进程加速（`forkserver` 模式）
- 自动检测可用 GPU

**参数**:
- `tasks_per_gpu`: 每个 GPU 运行的任务数（默认 1）
- `enable_multiprocessing`: 启用多进程
- `start_poses_per_object`: 每个物体的起始姿态数（默认 2000）
- `episodes_per_object`: 每个物体保留的片段数（默认 -1，保留全部）

**代码位置**: 行 1017-1084

---

## 📊 数据流

```
HM3D 场景文件 (.basis.glb)
    ↓
加载语义场景
    ↓
过滤物体类别（根据 OVON whitelist）
    ↓
为每个物体生成视点
  ├─ 创建周围网格
  ├─ 检查导航可行性
  ├─ 计算帧覆盖率
  └─ 过滤低覆盖率视点
    ↓
可选：导航网格聚类
    ↓
采样起始位置
  ├─ 随机采样 或 基于聚类采样
  ├─ 验证距离约束
  ├─ 验证楼层约束
  └─ 验证孤立区域
    ↓
创建片段
  ├─ 组合起始位置 + 物体目标
  └─ 添加元数据（距离、类别等）
    ↓
保存为 JSON.gz
```

---

## 🎯 应用场景

这个代码用于生成 **OVON (Open-Vocabulary Navigation)** 数据集：

**训练任务**:
- 根据语言描述找到物体
- 支持开放词汇（通过 WordNet 层次结构）
- 视觉导航：RGB + 深度 + 语义

**支持的数据集划分**:
- `train`: 训练集（81 个场景）
- `val_unseen_easy`: 验证集 - 简单（11 个场景）
- `val_unseen_hard`: 验证集 - 困难（5 个场景）

---

## 🔧 命令行参数

```bash
python ovon/dataset/objectnav_generator.py \
    --scene <scene_name>              # 指定单个场景
    --split <train|val_unseen_easy|val_unseen_hard>  # 数据集划分
    --output-path <path>              # 输出路径
    --num-scenes <n>                  # 处理场景数量
    --tasks-per-gpu <n>               # 每个 GPU 的任务数
    --multiprocessing                 # 启用多进程
    --start-poses-per-object <n>      # 起始姿态数量 (默认 2000)
    --episodes-per-object <n>         # 片段数量 (默认 0=全部)
    --disable-euc-geo-ratio-check     # 禁用距离比率检查
    --disable-wordnet-mapping         # 禁用 WordNet 映射
```

---

## 📦 依赖项

### 核心依赖
- `habitat-sim`: 仿真平台
- `habitat-lab`: 高层接口
- `numpy`: 数值计算
- `scikit-learn`: 聚类算法

### 可选依赖
- `pandas`, `seaborn`: 可视化
- `trimesh`: 网格处理

### 配置文件
- `data/hm3d_meta/ovon_categories.json`: OVON 类别白名单
- `data/wordnet/wordnet_mapping.json`: WordNet 层次映射
- `data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json`: 场景配置

---

## 🧠 算法亮点

### 1. **视点质量保证**
- 通过 `frame_cov_thresh` 确保物体在视野中足够大
- 三视角渲染（look_down, look_up, look_up）增加覆盖范围

### 2. **智能起始位置采样**
- 测地距离 vs 欧氏距离比率确保路径非平凡
- 楼层一致性检查避免跨楼层任务
- 孤立区域检测避免不可达位置

### 3. **高效空间采样**
- 导航网格聚类实现更均匀的空间覆盖
- 密集视点采样提高目标可达性

### 4. **开放词汇支持**
- WordNet 层次结构合并语义相关类别
- 支持 `children_object_categories` 扩展

---

## 📝 使用示例

### 处理单个场景
```bash
python ovon/dataset/objectnav_generator.py \
    --scene scene0011 \
    --split train \
    --output-path data/datasets/ovon/hm3d/ \
    --start-poses-per-object 2000
```

### 批量处理（多进程）
```bash
python ovon/dataset/objectnav_generator.py \
    --split train \
    --multiprocessing \
    --tasks-per-gpu 2 \
    --num-scenes 10
```

### 生成验证集
```bash
python ovon/dataset/objectnav_generator.py \
    --split val_unseen_easy \
    --episodes-per-object 100 \
    --disable-wordnet-mapping
```

---

## 🔍 调试与可视化

代码中包含注释掉的可视化代码：
```python
# 行 279-285: 显示视点的俯视图
# plot_area(
#     candiatate_poses_ious,
#     [v["agent_state"]["position"] for v in view_locations],
#     [object_position],
#     obj.id,
# )

# 行 656-670: 距离分布直方图
if self.verbose:
    plt.figure(figsize=(8, 8))
    sns.histplot(data=hist_data, x="Geodesic distance")
    plt.savefig(...)
```

启用方式：设置 `verbose=True` 和取消相关注释

---

## ⚠️ 注意事项

1. **内存使用**: 密集视点采样会消耗大量内存，建议使用 GPU
2. **计算时间**: 单个场景可能需要数小时（取决于 `start_poses_per_obj`）
3. **磁盘空间**: 每个场景的输出文件约 10-100MB
4. **导航网格**: 确保场景有有效的导航网格，否则会报错

---

## 📚 相关文件

- `ovon/dataset/pose_sampler.py`: 姿态采样器
- `ovon/dataset/semantic_utils.py`: 语义工具（类别映射）
- `ovon/dataset/ovon_dataset.py`: 数据集格式定义
- `ovon/dataset/visualization.py`: 可视化工具（已注释）

---

## 🔄 版本信息

- **最后更新**: 2026-03-17
- **兼容 Habitat 版本**: 0.1.x+
- **支持的场景格式**: HM3D (.basis.glb)

---

## 📖 参考资料

- [Habitat 仿真平台文档](https://aihabitat.org/)
- [HM3D 数据集](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md)
- [OVON 论文](https://arxiv.org/abs/2107.03649) (如果适用)

---

**分析完成！** 如有疑问或需要进一步的详细分析，请参考具体代码行数。
