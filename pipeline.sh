#!/bin/bash
#
# End-to-end pipeline and status inspector for the DyGFormer Social-feature study.
#
# This repository extends DyGLib/DyGFormer with a multi-feature mechanism. The
# only feature that holds up across datasets and tasks is the time-aware Social
# structural channel; Style/Topic/Personality are LIWC-derived proxies kept as
# ablation references. This script is the single entry point that (a) reports the
# full current state of the project and (b) reproduces the Social v1/v2 protocol
# end to end under strict experiment isolation.
#
# Variants (each writes to its own isolated experiment directory):
#   base : DyGFormer baseline, 4 channels, no extra feature
#   v1   : + Social v1_trainfit (10-dim edge feature, train-edge-only z-score)
#   v2   : + Social v2_trainfit (20-dim: v1 5-dim + 5 windowed/decayed/burstiness)
#
# Subcommands:
#   status              read-only inventory of code, data, features and results (default)
#   features            generate Social v1_trainfit and v2_trainfit feature banks
#   smoke               1-seed / 3-epoch link-prediction sanity run per variant
#   lp                  5-seed link prediction for base / v1 / v2
#   nc                  5-seed node classification for base / v1 / v2 (needs lp first)
#   analyze             aggregate results and run the v2-vs-v1 redundancy diagnostic
#   all                 features -> smoke -> lp -> nc -> analyze
#
# Environment overrides:
#   GPU=0 NUM_RUNS=5 DATASETS="wikipedia reddit" CONDA_SH=... CONDA_ENV=dyg
#
# Examples:
#   bash pipeline.sh status
#   GPU=0 bash pipeline.sh features
#   GPU=0 bash pipeline.sh lp
#   bash pipeline.sh analyze

set -euo pipefail

# --- Configuration -----------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-0}"
NUM_RUNS="${NUM_RUNS:-5}"
MODEL="DyGFormer"
DATASETS="${DATASETS:-wikipedia reddit}"
PROCESSED_DIR="${PROCESSED_DIR:-./processed_data}"

# Conda activation is required for compute phases; status runs without it.
CONDA_SH="${CONDA_SH:-/home/zyh/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-dyg}"

EXPERIMENTS_ROOT="experiments"
VARIANT_ORDER="base v1 v2"

variant_dir() {
    # Isolated experiment directory for a variant (avoids artifact overwrite).
    case "$1" in
        base) echo "${EXPERIMENTS_ROOT}/feature_base_ref" ;;
        v1)   echo "${EXPERIMENTS_ROOT}/feature_social_v1_trainfit" ;;
        v2)   echo "${EXPERIMENTS_ROOT}/feature_social_v2_trainfit" ;;
    esac
}

variant_version() {
    # Social feature version tag for a variant.
    case "$1" in
        v1) echo "v1_trainfit" ;;
        v2) echo "v2_trainfit" ;;
    esac
}

# --- Helpers -----------------------------------------------------------------

log() { echo "[pipeline] $*"; }

activate_env() {
    # Activate the conda environment used for training/evaluation.
    if [[ ! -f "$CONDA_SH" ]]; then
        echo "ERROR: conda profile not found at $CONDA_SH (override with CONDA_SH=...)." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    log "conda env '${CONDA_ENV}' active: $(python --version 2>&1)"
}

variant_flags() {
    # Emit the feature CLI flags for a given variant.
    local variant="$1"
    if [[ "$variant" == "base" ]]; then
        return 0
    fi
    local exp_dir="$(variant_dir "$variant")"
    printf -- "--use_social --feature_bank_dir %s/feature_bank --feature_version %s" \
        "$exp_dir" "$(variant_version "$variant")"
}

# --- status ------------------------------------------------------------------

cmd_status() {
    # Read-only snapshot of the project. Safe to run anywhere; tolerates missing
    # data/experiment artifacts (they live on the GPU server and are gitignored).
    set +e

    echo "============================================================"
    echo " Project status: DyGFormer Social-feature study"
    echo "============================================================"

    echo
    echo "## Repository"
    echo "root: $REPO_ROOT"
    echo "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    echo "recent commits:"
    git log --oneline -5 2>/dev/null | sed 's/^/  /'

    echo
    echo "## Entry points"
    for f in train_link_prediction.py evaluate_link_prediction.py \
             train_node_classification.py evaluate_node_classification.py \
             run_experiments.sh pipeline.sh; do
        [[ -e "$f" ]] && echo "  present: $f" || echo "  MISSING: $f"
    done

    echo
    echo "## Feature modules (features/)"
    ls -1 features/*.py 2>/dev/null | sed 's/^/  /'

    echo
    echo "## Documentation"
    for f in INTRO.md PLAN.md RESULTS.md HANDOFF.md $(ls docs/*.md 2>/dev/null); do
        [[ -f "$f" ]] && echo "  $f -> $(grep -m1 '^#' "$f" 2>/dev/null | sed 's/^#* //')"
    done

    echo
    echo "## Processed datasets ($PROCESSED_DIR)"
    for ds in $DATASETS; do
        local csv="$PROCESSED_DIR/$ds/ml_${ds}.csv"
        if [[ -f "$csv" ]]; then
            local rows
            rows="$(($(wc -l < "$csv") - 1))"
            echo "  $ds: ml_${ds}.csv (${rows} edges)"
            ls -1 "$PROCESSED_DIR/$ds"/*.npy 2>/dev/null | sed 's#.*/#    npy: #'
        else
            echo "  $ds: not present locally"
        fi
    done

    echo
    echo "## Feature banks and experiment artifacts"
    if [[ -d "$EXPERIMENTS_ROOT" ]]; then
        for variant in $VARIANT_ORDER; do
            local exp="$(variant_dir "$variant")"
            echo "  [$variant] $exp"
            [[ -d "$exp" ]] || { echo "    (absent)"; continue; }
            find "$exp/feature_bank" -name '*.npy' 2>/dev/null | sed 's/^/    feature: /'
            for ds in $DATASETS; do
                local lp_dir="$exp/saved_results/$MODEL/$ds"
                local lp_n nc_n
                lp_n=$(ls "$lp_dir"/${MODEL}_*_seed*.json 2>/dev/null | wc -l | tr -d ' ')
                nc_n=$(ls "$lp_dir"/node_classification_${MODEL}_*_seed*.json 2>/dev/null | wc -l | tr -d ' ')
                echo "    $ds: lp_result_files=${lp_n} nc_result_files=${nc_n}"
            done
        done
    else
        echo "  $EXPERIMENTS_ROOT/ not present locally (server-side, gitignored)"
    fi

    echo
    echo "## Aggregated metrics (if results present)"
    for variant in $VARIANT_ORDER; do
        local results_root="$(variant_dir "$variant")/saved_results"
        if [[ -d "$results_root" ]] && command -v python >/dev/null 2>&1; then
            echo "  --- variant=$variant ($results_root) ---"
            python experiment_analysis/collect_results.py \
                --task lp --results_root "$results_root" 2>/dev/null | sed 's/^/    /'
        fi
    done

    set -e
    echo
    echo "status complete."
}

# --- features ----------------------------------------------------------------

cmd_features() {
    # Generate Social v1_trainfit and v2_trainfit feature banks per dataset.
    # Normalization is fit on the real LP training edges only (see PLAN.md).
    activate_env
    for variant in v1 v2; do
        local exp="$(variant_dir "$variant")"
        local version="$(variant_version "$variant")"
        local bank_dir="$exp/feature_bank"
        mkdir -p "$bank_dir"
        for ds in $DATASETS; do
            log "generate Social $version for $ds -> $bank_dir/$ds/social_${version}.npy"
            python -m features.social_extractor \
                --dataset_name "$ds" \
                --processed_dir "$PROCESSED_DIR" \
                --feature_bank_dir "$bank_dir" \
                --version "$version"
        done
    done
    log "feature generation done."
}

# --- training ----------------------------------------------------------------

run_lp() {
    # Train link prediction for one variant. Extra args are forwarded (used by smoke).
    local variant="$1"; shift
    local exp="$(variant_dir "$variant")"
    for ds in $DATASETS; do
        log "[LP] variant=$variant dataset=$ds exp=$exp"
        # shellcheck disable=SC2046
        python train_link_prediction.py \
            --dataset_name "$ds" \
            --model_name "$MODEL" \
            --load_best_configs \
            --gpu "$GPU" \
            --experiment_dir "$exp" \
            $(variant_flags "$variant") \
            "$@"
    done
}

cmd_smoke() {
    # Single-seed, 3-epoch sanity run. Verifies executability and artifact paths,
    # not model quality.
    activate_env
    for variant in $VARIANT_ORDER; do
        run_lp "$variant" --num_runs 1 --num_epochs 3 --test_interval_epochs 1
    done
    log "smoke done."
}

cmd_lp() {
    # Full 5-seed link prediction with identical hyperparameters across variants.
    activate_env
    for variant in $VARIANT_ORDER; do
        run_lp "$variant" --num_runs "$NUM_RUNS"
    done
    log "lp done."
}

cmd_nc() {
    # 5-seed node classification. Loads the LP checkpoint from the same
    # experiment_dir / feature_tag, so LP must be trained first.
    activate_env
    for variant in $VARIANT_ORDER; do
        local exp="$(variant_dir "$variant")"
        for ds in $DATASETS; do
            log "[NC] variant=$variant dataset=$ds exp=$exp"
            # shellcheck disable=SC2046
            python train_node_classification.py \
                --dataset_name "$ds" \
                --model_name "$MODEL" \
                --load_best_configs \
                --num_runs "$NUM_RUNS" \
                --gpu "$GPU" \
                --experiment_dir "$exp" \
                $(variant_flags "$variant")
        done
    done
    log "nc done."
}

# --- analyze -----------------------------------------------------------------

cmd_analyze() {
    # Aggregate per-variant results and run the v2-vs-v1 paired delta plus the
    # per-dimension redundancy diagnostic (correlation to v1, label AUC).
    activate_env
    for variant in $VARIANT_ORDER; do
        local results_root="$(variant_dir "$variant")/saved_results"
        [[ -d "$results_root" ]] || continue
        log "collect results: variant=$variant"
        python experiment_analysis/collect_results.py --task all --results_root "$results_root"
    done

    log "v2 vs v1 paired delta and redundancy diagnostic"
    python experiment_analysis/social_v2_analysis.py \
        --base_results_root "$(variant_dir base)/saved_results" \
        --v1_results_root "$(variant_dir v1)/saved_results" \
        --v2_results_root "$(variant_dir v2)/saved_results" \
        --v1_feature_bank_root "$(variant_dir v1)/feature_bank" \
        --v2_feature_bank_root "$(variant_dir v2)/feature_bank" \
        --processed_dir "$PROCESSED_DIR" \
        --datasets $DATASETS
    log "analyze done."
}

# --- dispatch ----------------------------------------------------------------

main() {
    local cmd="${1:-status}"
    case "$cmd" in
        status)   cmd_status ;;
        features) cmd_features ;;
        smoke)    cmd_smoke ;;
        lp)       cmd_lp ;;
        nc)       cmd_nc ;;
        analyze)  cmd_analyze ;;
        all)
            cmd_features
            cmd_smoke
            cmd_lp
            cmd_nc
            cmd_analyze
            ;;
        *)
            echo "Unknown subcommand: $cmd" >&2
            echo "Usage: bash pipeline.sh {status|features|smoke|lp|nc|analyze|all}" >&2
            exit 1
            ;;
    esac
}

main "$@"
