#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-1 2 3 4 5}"

for DATASET in CIFAR10 CIFAR100; do
  for SEED in ${SEEDS}; do
    python experiments/vision/train.py \
      -c "experiments/vision/config/ILNN-${DATASET}.txt" \
      --device "${DEVICE}" \
      --seed "${SEED}"
  done
done

