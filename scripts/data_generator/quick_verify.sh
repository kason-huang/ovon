#!/bin/bash
###############################################################################
# 快速验证脚本
#
# 用途: 快速验证场景和代码是否正常工作
# 时间: 约 30-60 分钟
###############################################################################

set -e

SCENE_NAME="00016-qk9eeNeR4vw"
SCENE_ID="qk9eeNeR4vw"
SPLIT="train"
DEVICE_ID=0

echo "=========================================="
echo "快速验证: ${SCENE_NAME}"
echo "=========================================="

# 1. 验证场景文件
echo ""
echo "[1/4] 验证场景文件..."
SCENE_FILE="data/scene_datasets/hm3d/${SPLIT}/${SCENE_NAME}/${SCENE_ID}.basis.glb"
if [ -f "$SCENE_FILE" ]; then
    echo "✓ 场景文件存在"
    ls -lh "$SCENE_FILE"
else
    echo "✗ 场景文件不存在: $SCENE_FILE"
    exit 1
fi

# 2. 测试场景加载
echo ""
echo "[2/4] 测试场景加载..."
python -c "
import habitat_sim
import os
import sys

try:
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_dataset_config_file = 'data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json'
    cfg.scene_id = '${SCENE_NAME}'
    cfg.gpu_device_id = ${DEVICE_ID}

    # 创建 agent 配置
    agent_cfg = habitat_sim.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))

    objects = sim.semantic_scene.objects
    print(f'✓ 场景加载成功')
    print(f'  物体总数: {len(objects)}')

    # 测试 ObjectCategoryMapping
    try:
        from ovon.dataset.semantic_utils import ObjectCategoryMapping

        target_categories = ['chair', 'bed', 'toilet', 'sofa', 'plant', 'tv_monitor']

        cat_map = ObjectCategoryMapping(
            mapping_file='ovon/dataset/source_data/Mp3d_category_mapping.tsv',
            allowed_categories=set(target_categories),
            coverage_meta_file=None,  # 快速验证时跳过
        )

        target_objects = [o for o in objects if cat_map.get(o.category.name()) is not None]
        print(f'  目标物体数: {len(target_objects)}')

        # 统计目标类别
        category_count = {}
        for obj in target_objects:
            cat = cat_map.get(obj.category.name())
            if cat:
                category_count[cat] = category_count.get(cat, 0) + 1

        print(f'  目标类别分布:')
        for cat, count in sorted(category_count.items()):
            print(f'    {cat}: {count}')

    except ImportError as e:
        print(f'  ⚠️  无法导入 ObjectCategoryMapping: {e}')
        print(f'  继续验证...')

    sim.close()
    print('✓ 场景验证完成')

except Exception as e:
    print(f'✗ 场景加载失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" || exit 1

# 3. 生成测试覆盖率元数据
echo ""
echo "[3/4] 生成测试覆盖率元数据..."
python scripts/data_generator/generate_coverage_meta.py \
    --split "${SPLIT}" \
    --max-scenes 1 \
    --device-id "${DEVICE_ID}"

# 4. 生成测试任务集
echo ""
echo "[4/4] 生成测试任务集..."
python ovon/dataset/objectnav_generator.py \
    --scene "${SCENE_NAME}" \
    --split "${SPLIT}" \
    --start-poses-per-object 100 \
    --episodes-per-object 50 \
    --device-id "${DEVICE_ID}"

# 5. 验证输出
echo ""
echo "=========================================="
echo "验证输出"
echo "=========================================="

OUTPUT_FILE="data/datasets/ovon/hm3d/v1_stretch/${SPLIT}/content/${SCENE_NAME}.json.gz"
if [ -f "$OUTPUT_FILE" ]; then
    echo "✓ 任务集已生成: $OUTPUT_FILE"
    python -c "
import gzip
import json
with gzip.open('${OUTPUT_FILE}', 'rt') as f:
    data = json.load(f)
print(f'  片段数: {len(data[\"episodes\"])}')
print(f'  类别: {list(data[\"goals_by_category\"].keys())}')
"
else
    echo "✗ 任务集未生成"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ 快速验证完成！"
echo "=========================================="
echo ""
echo "如果测试通过，可以运行完整流程:"
echo "  bash scripts/data_generator/run_new_scene_pipeline.sh"
