#!/usr/bin/env bash
set -euo pipefail
notebook="${1:?Notebook name required}"
shift
params=(-p mode full -p device cuda)
case "$notebook" in
    convergence|cyclic_similarity|stages_of_inference|mixing_correlation|isolating_stability|bos_comparison|entropy_lens)
        dataset=gsm8k
        if (( $# > 0 )) && [[ "$1" != -* ]]; then
            dataset="$1"
            shift
            if [[ "$dataset" != gsm8k && "$dataset" != hellaswag ]]; then
                printf 'Dataset must be gsm8k or hellaswag\n' >&2
                exit 2
            fi
        fi
        params+=(-p dataset "$dataset" -p limit 256)
        ;;
    limiting_behaviour)
        dataset=gsm8k
        ;;
    multiple_choice)
        dataset=multiple_datasets
        params+=(-p limit 256)
        ;;
    trajectories|initialisation)
        dataset=illustrative
        ;;
    neuron_detection)
        dataset=weights
        ;;
    *) printf 'Unknown notebook: %s\n' "$notebook" >&2; exit 2 ;;
esac
mkdir -p experiments/runs slurm_out
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/looped-matplotlib-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
printf 'Notebook: %s; default parameters: %s; overrides: %s\n' "$notebook" "${params[*]}" "$*"
# Save the executed notebook at completion, avoiding repeated large writes during inference.
.venv/bin/python -m papermill \
    "experiments/${notebook}.ipynb" \
    "experiments/runs/${notebook}-${dataset}-${SLURM_JOB_ID:-local}.ipynb" \
    --no-request-save-on-cell-execute --autosave-cell-every 0 \
    --kernel looped-internals --cwd . --log-output \
    "${params[@]}" "$@"
