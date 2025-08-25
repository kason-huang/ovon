#!/bin/bash
# prepare： 输入要测试的类型 val_type,  比如val_seen, val_unseen, val_seen_image, val_seen_object, val_seen_synonyms, val_unseen（都是v2_new目录的）
# 1. 如果val_type是ovon打头的，就直接修改名字就可以了，因为ovon生成的只有object goal，而object category都是一样的
# 2. 如果val_type不带ovon，就是goal打头的，需要把image的embedding给修改下，具体就是把tmp/iin/{split}_embeddings -> train_embbeding
# 3. 把val_type 对应的目录名修改-> val
# 写入运行记录和时间

LOG_FILE="./rename_dirs.log"

VAL_TYPE="${1:-}"
if [[ -z "${VAL_TYPE}" ]]; then
  echo "用法: $0 <val_type>"
  exit 1
fi

A_ROOT="./data/datasets/ovon/hm3d/v2_new"
B_ROOT="./tmp/iin"

SRC_A="${A_ROOT}/${VAL_TYPE}"
DST_A="${A_ROOT}/val"

SRC_B="${B_ROOT}/${VAL_TYPE}_embeddings"
DST_B="${B_ROOT}/train_embeddings"

ts() { date '+%F %T'; }
log() { echo "[$(ts)] val_type=${VAL_TYPE} - $*" | tee -a "${LOG_FILE}"; }

mv_safe() {
  local src="$1" dst="$2"
  if [[ ! -e "$src" ]]; then
    log "ERROR: 源不存在: $src"
    return 1
  fi
  if [[ -e "$dst" ]]; then
    log "ERROR: 目标已存在: $dst"
    return 1
  fi
  mv "$src" "$dst"
  log "mv: $src -> $dst"
}

# 标记哪些目录被修改过，用于后续回退
did_mv_A=false
did_mv_B=false

# === 修改部分 ===
if [[ "${VAL_TYPE}" == ovon* ]]; then
  # 只改 A
  mv_safe "$SRC_A" "$DST_A"
  did_mv_A=true
else
  # 改 B 和 A
  mv_safe "$SRC_B" "$DST_B"
  did_mv_B=true

  mv_safe "$SRC_A" "$DST_A"
  did_mv_A=true
fi

log "目录修改完成，执行eval后 后开始回退..."
# sleep 2
python -m ovon.run   --run-type eval   --exp-config config/experiments/transformer_rl_goat_eval.yaml   habitat_baselines.eval_ckpt_path_dir=/root/workspace/lab/ovon/data/new_checkpoints_goat_0818/latest.pth

# === 回退部分 ===
if $did_mv_B && [[ -e "$DST_B" && ! -e "$SRC_B" ]]; then
  mv "$DST_B" "$SRC_B"
  log "REVERT: $DST_B -> $SRC_B"
fi

if $did_mv_A && [[ -e "$DST_A" && ! -e "$SRC_A" ]]; then
  mv "$DST_A" "$SRC_A"
  log "REVERT: $DST_A -> $SRC_A"
fi

log "所有需要的目录已回退完成。"
echo "" | tee -a "${LOG_FILE}"

