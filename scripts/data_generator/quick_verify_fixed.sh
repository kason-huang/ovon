#!/bin/bash
###############################################################################
# 快速验证脚本（修复版）
#
# 用途: 快速验证场景和代码是否正常工作
# 时间: 约 10-20 分钟
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

    # 统计类别分布
    category_count = {}
    for obj in objects:
        cat_name = obj.category.name()
        category_count[cat_name] = category_count.get(cat_name, 0) + 1

    print(f'  类别数: {len(category_count)}')
    print(f'  前 10 个类别:')
    for cat, count in sorted(category_count.items(), key=lambda x: -x[1])[:10]:
        print(f'    {cat}: {count}')

    sim.close()
    print('✓ 场景验证完成')

except Exception as e:
    print(f'✗ 场景加载失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" || exit 1

# 3. 创建临时覆盖率文件
echo ""
echo "[3/4] 创建临时覆盖率元数据..."
python -c "
import pickle
from collections import defaultdict
import os

os.makedirs('data/coverage_meta', exist_ok=True)

# 创建空的覆盖率元数据
empty_coverage = defaultdict(list)
temp_file = 'data/coverage_meta/temp_train.pkl'

with open(temp_file, 'wb') as f:
    pickle.dump(dict(empty_coverage), f)

print(f'✓ 临时覆盖率文件已创建: {temp_file}')
"

# 4. 测试覆盖率元数据生成
echo ""
echo "[4/4] 测试覆盖率元数据生成..."

# 设置 PYTHONPATH
export PYTHONPATH=/home/kason/workspace/ovon:$PYTHONPATH

python scripts/data_generator/generate_coverage_meta.py \
    --split "${SPLIT}" \
    --max-scenes 1 \
    --device-id "${DEVICE_ID}" \
    --scene-datasets-path "data/scene_datasets/hm3d" \
    --output-path "data/coverage_meta" 2>&1 | grep -v "^\[Metadata\]" | tail -30

# 检查是否成功生成
if [ -f "data/coverage_meta/train.pkl" ]; then
    echo ""
    echo "=========================================="
    echo "✓ 快速验证完成！"
    echo "=========================================="
    echo ""
    echo "生成的文件:"
    echo "  - data/coverage_meta/train.pkl"
    echo ""
    echo "下一步:"
    echo "  1. 生成完整的覆盖率元数据（所有场景）"
    echo "  2. 生成任务集"
    echo ""
    echo "运行完整流程:"
    echo "  bash scripts/data_generator/run_new_scene_pipeline.sh"
else
    echo ""
    echo "⚠️  覆盖率元数据生成可能失败"
    echo "请检查错误信息"
fi
