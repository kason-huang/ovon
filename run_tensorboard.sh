#!/bin/bash

# 检查是否传入参数
if [ $# -lt 1 ]; then
  echo "用法: $0 <logdir>"
  exit 1
fi

LOGDIR=$1

tensorboard --logdir "$LOGDIR" --host 0.0.0.0 --port 1234
