#!/bin/bash
#
# Unified experiment runner for DyGFormer + multi-feature extension.
# Covers 8 feature combinations x 2 datasets x link prediction.
# Node classification requires pretrained LP models (run LP first).
#
# Usage:
#   bash run_experiments.sh              # run all LP experiments
#   bash run_experiments.sh --nc         # run all NC experiments (after LP)
#   bash run_experiments.sh --tag full   # run only the "full" combination
#   bash run_experiments.sh --dataset wikipedia  # run only wikipedia
#
set -euo pipefail

GPU="${GPU:-0}"
NUM_RUNS="${NUM_RUNS:-5}"
MODE="lp"  # default: link prediction
FILTER_TAG=""
FILTER_DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --nc)       MODE="nc"; shift ;;
        --tag)      FILTER_TAG="$2"; shift 2 ;;
        --dataset)  FILTER_DATASET="$2"; shift 2 ;;
        --gpu)      GPU="$2"; shift 2 ;;
        --num_runs) NUM_RUNS="$2"; shift 2 ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

DATASETS=("wikipedia" "reddit")

# Feature combinations: tag -> CLI flags
declare -A CONFIGS
CONFIGS["base"]=""
CONFIGS["S"]="--use_style"
CONFIGS["P"]="--use_personality"
CONFIGS["T"]="--use_topic"
CONFIGS["So"]="--use_social"
CONFIGS["S_P"]="--use_style --use_personality"
CONFIGS["S_P_T"]="--use_style --use_personality --use_topic"
CONFIGS["full"]="--use_style --use_personality --use_topic --use_social"

# Deterministic order
TAG_ORDER=("base" "S" "P" "T" "So" "S_P" "S_P_T" "full")

echo "============================================"
echo "  DyGFormer Multi-Feature Experiments"
echo "  Mode: ${MODE}  GPU: ${GPU}  Runs: ${NUM_RUNS}"
echo "============================================"

for DATASET in "${DATASETS[@]}"; do
    if [[ -n "$FILTER_DATASET" && "$DATASET" != "$FILTER_DATASET" ]]; then
        continue
    fi

    for TAG in "${TAG_ORDER[@]}"; do
        if [[ -n "$FILTER_TAG" && "$TAG" != "$FILTER_TAG" ]]; then
            continue
        fi

        FLAGS=${CONFIGS[$TAG]}

        if [[ "$MODE" == "lp" ]]; then
            echo ""
            echo ">>> [LP] dataset=${DATASET} tag=${TAG} flags=[${FLAGS}]"
            python train_link_prediction.py \
                --dataset_name "$DATASET" \
                --model_name DyGFormer \
                --load_best_configs \
                --num_runs "$NUM_RUNS" \
                --gpu "$GPU" \
                $FLAGS
        elif [[ "$MODE" == "nc" ]]; then
            if [[ "$DATASET" != "wikipedia" && "$DATASET" != "reddit" ]]; then
                echo ">>> Skipping NC for ${DATASET} (only wikipedia/reddit supported)"
                continue
            fi
            echo ""
            echo ">>> [NC] dataset=${DATASET} tag=${TAG} flags=[${FLAGS}]"
            python train_node_classification.py \
                --dataset_name "$DATASET" \
                --model_name DyGFormer \
                --load_best_configs \
                --num_runs "$NUM_RUNS" \
                --gpu "$GPU" \
                $FLAGS
        fi
    done
done

echo ""
echo "============================================"
echo "  All experiments completed."
echo "  Collect results: python collect_results.py"
echo "============================================"
