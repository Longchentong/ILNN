#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-1 123456 234567 345678 456789}"

for TASK in processed_pseudogenes unprocessed_pseudogenes; do
  for SEED in ${SEEDS}; do
    python experiments/genomics/train.py \
      -c experiments/genomics/configs/ILNN_TEB_pseudogenes.txt \
      --device "${DEVICE}" \
      --dataset_name "${TASK}" \
      --seed "${SEED}"
  done
done

