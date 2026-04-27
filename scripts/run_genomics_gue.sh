#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-1 999999 888888 555555 333333}"

for TASK in prom_core_tata prom_core_notata prom_core_all prom_300_tata prom_300_notata prom_300_all; do
  for SEED in ${SEEDS}; do
    python experiments/genomics/train.py \
      -c experiments/genomics/configs/ILNN_GUE_promoter.txt \
      --device "${DEVICE}" \
      --dataset_name "${TASK}" \
      --seed "${SEED}"
  done
done

for SEED in ${SEEDS}; do
  python experiments/genomics/train.py \
    -c experiments/genomics/configs/ILNN_GUE_covid.txt \
    --device "${DEVICE}" \
    --seed "${SEED}"
done

