#!/bin/bash
###############################################################################
# 新场景数据集生成 - 完整流程脚本
#
# 用途: 为新场景生成完整的 OVON 数据集
# 场景: 00016-qk9eeNeR4vw
# 配置: 8+ GPU, 完整生成覆盖率元数据
#
# 使用方法:
#   cd /home/kason/workspace/ovon
#   bash scripts/data_generator/run_new_scene_pipeline.sh
#
# 时间估算:
#   - 阶段1 (快速验证): 30-60 分钟
#   - 阶段2 (覆盖率元数据): 1-2 小时 (8 GPU 并行)
#   - 阶段3 (生成任务集): 30-60 分钟
#   - 总计: 2-4 小时
###############################################################################

set -e  # 遇到错误立即退出

# ==================== 配置参数 ====================
SCENE_NAME="00016-qk9eeNeR4vw"
SCENE_ID="qk9eeNeR4vw"
SPLIT="train"
NUM_GPUS=8
DEVICE_START_ID=0

# 路径配置
DATA_DIR="/home/kason/workspace/ovon/data"
SCENE_DATASETS_PATH="${DATA_DIR}/scene_datasets/hm3d"
COVERAGE_META_PATH="${DATA_DIR}/coverage_meta"
OUTPUT_PATH="${DATA_DIR}/datasets/ovon/hm3d/v1_stretch"

# 生成参数
START_POSES_PER_OBJECT=2000  # 高质量
EPISODES_PER_OBJECT=0         # 0 = 保留所有

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==================== 工具函数 ====================
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_stage() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

check_file() {
    if [ -f "$1" ]; then
        log_success "✓ 文件存在: $1"
        return 0
    else
        log_error "✗ 文件不存在: $1"
        return 1
    fi
}

# ==================== 阶段 1: 环境检查 ====================
print_stage "阶段 1: 环境检查"

log_info "检查场景文件..."
SCENE_FILE="${SCENE_DATASETS_PATH}/${SPLIT}/${SCENE_NAME}/${SCENE_ID}.basis.glb"
check_file "${SCENE_FILE}" || exit 1

SEMANTIC_FILE="${SCENE_DATASETS_PATH}/${SPLIT}/${SCENE_NAME}/${SCENE_ID}.semantic.glb"
check_file "${SEMANTIC_FILE}" || exit 1

log_info "检查配置文件..."
CONFIG_FILE="${SCENE_DATASETS_PATH}/hm3d_annotated_basis.scene_dataset_config.json"
check_file "${CONFIG_FILE}" || exit 1

log_info "检查场景是否在配置中..."
if grep -q "${SCENE_NAME}" "${CONFIG_FILE}"; then
    log_success "✓ 场景已在配置文件中"
else
    log_error "✗ 场景不在配置文件中，请先添加"
    exit 1
fi

log_info "创建输出目录..."
mkdir -p "${COVERAGE_META_PATH}"
mkdir -p "${OUTPUT_PATH}/${SPLIT}/content/"
log_success "✓ 目录创建完成"

# ==================== 阶段 2: 快速验证 ====================
print_stage "阶段 2: 快速验证（单场景测试）"

log_info "测试场景是否能正常加载..."
python -c "
import habitat_sim
import sys

try:
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_dataset_config_file = '${CONFIG_FILE}'
    cfg.scene_id = '${SCENE_NAME}'
    cfg.gpu_device_id = ${DEVICE_START_ID}

    sim = habitat_sim.Simulator(cfg)
    objects = sim.semantic_scene.objects
    print(f'场景加载成功! 物体数量: {len(objects)}')
    sim.close()
except Exception as e:
    print(f'场景加载失败: {e}')
    sys.exit(1)
" || exit 1

log_success "✓ 场景验证通过"

# ==================== 阶段 3: 生成覆盖率元数据 ====================
print_stage "阶段 3: 生成覆盖率元数据"

log_info "检查覆盖率元数据是否已存在..."
COVERAGE_FILE="${COVERAGE_META_PATH}/${SPLIT}.pkl"
if [ -f "${COVERAGE_FILE}" ]; then
    log_warning "覆盖率元数据已存在: ${COVERAGE_FILE}"
    read -p "是否重新生成? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "跳过覆盖率元数据生成"
        goto generate_episodes
    fi
fi

log_info "生成覆盖率元数据（使用 GPU ${DEVICE_START_ID}）..."
python scripts/data_generator/generate_coverage_meta.py \
    --split "${SPLIT}" \
    --scene-datasets-path "${SCENE_DATASETS_PATH}" \
    --output-path "${COVERAGE_META_PATH}" \
    --device-id "${DEVICE_START_ID}" \
    --max-scenes 1

log_success "✓ 覆盖率元数据生成完成"

# 验证覆盖率元数据
log_info "验证覆盖率元数据..."
python -c "
import pickle
import os

file = '${COVERAGE_FILE}'
if not os.path.exists(file):
    print(f'错误: 覆盖率元数据不存在: {file}')
    exit(1)

with open(file, 'rb') as f:
    data = pickle.load(f)

print(f'覆盖率元数据验证通过:')
print(f'  类别数量: {len(data)}')
for cat, items in list(data.items())[:5]:
    print(f'  {cat}: {len(items)} 个视点')
"

log_success "✓ 覆盖率元数据验证通过"

# ==================== 阶段 4: 生成任务集 ====================
print_stage "阶段 4: 生成任务集"

generate_episodes:
log_info "生成场景任务集..."
python ovon/dataset/objectnav_generator.py \
    --scene "${SCENE_NAME}" \
    --split "${SPLIT}" \
    --output-path "${OUTPUT_PATH}" \
    --start-poses-per-object "${START_POSES_PER_OBJECT}" \
    --episodes-per-object "${EPISODES_PER_OBJECT}" \
    --device-id "${DEVICE_START_ID}"

log_success "✓ 任务集生成完成"

# ==================== 阶段 5: 验证输出 ====================
print_stage "阶段 5: 验证输出"

log_info "检查输出文件..."
OUTPUT_FILE="${OUTPUT_PATH}/${SPLIT}/content/${SCENE_NAME}.json.gz"

if [ -f "${OUTPUT_FILE}" ]; then
    FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
    log_success "✓ 输出文件已生成: ${OUTPUT_FILE} (${FILE_SIZE})"

    # 验证内容
    python -c "
import gzip
import json

with gzip.open('${OUTPUT_FILE}', 'rt') as f:
    data = json.load(f)

num_episodes = len(data['episodes'])
num_categories = len(data['goals_by_category'])

print(f'数据集验证通过:')
print(f'  总片段数: {num_episodes}')
print(f'  类别数: {num_categories}')
for cat in list(data['goals_by_category'].keys())[:5]:
    print(f'  - {cat}')
"
else
    log_error "✗ 输出文件未生成: ${OUTPUT_FILE}"
    exit 1
fi

# ==================== 完成 ====================
print_stage "完成！"

log_success "数据集生成流程已成功完成！"
echo ""
echo "生成的文件:"
echo "  1. 覆盖率元数据: ${COVERAGE_FILE}"
echo "  2. 任务集数据: ${OUTPUT_FILE}"
echo ""
echo "下一步:"
echo "  1. 查看生成的任务: python -c \"import gzip, json; data=json.loads(gzip.open('${OUTPUT_FILE}').read()); print(data['episodes'][0])\""
echo "  2. 开始训练: python -m ovon.run --run-type train --exp-config config/experiments/your_config.yaml"
echo ""
log_info "祝你使用愉快！"
