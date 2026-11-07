#!/usr/bin/env bash
# Submit independent cache jobs for both datasets; keep convergence/stages together.
set -euo pipefail
render_figures=false
if (( $# == 1 )) && [[ "$1" == --figures ]]; then
    render_figures=true
elif (( $# )); then
    printf 'Usage: bash slurm_jobs/submit_cache.sh [--figures]\nSet LOOPED_INTERNALS_CACHE_DIR to override the cache location.\n' >&2
    exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."

cache_directory="${LOOPED_INTERNALS_CACHE_DIR:-/data-gauss/hugbla/looped-internals-cache}"
mkdir -p slurm_out experiments/runs
manifest="$(mktemp "experiments/runs/cache-submissions-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.tsv")"
printf 'dataset\tnotebooks\tjob_id\n' > "$manifest"
printf 'Cache: %s\nManifest: %s\n' "$cache_directory" "$manifest"
cache_job_ids=()

submit() {
    local cache_dataset="$1" cache_group="$2" cache_job_id
    shift 2
    local dataset_args=()
    local dependency_args=()
    if [[ "$cache_dataset" != shared ]]; then
        dataset_args=(--datasets "$cache_dataset")
    fi
    if [[ "$cache_group" == multiple_choice && -n "${LOOPED_INTERNALS_MCQ_AFTER_JOB:-}" ]]; then
        dependency_args=(--dependency="afterany:${LOOPED_INTERNALS_MCQ_AFTER_JOB}" --kill-on-invalid-dep=yes)
    fi
    cache_job_id="$(sbatch --parsable --job-name="cache-${cache_group}-${cache_dataset}" \
        "${dependency_args[@]}" \
        slurm_jobs/populate_cache.job \
        --cache-dir "$cache_directory" --cache-mode reuse \
        "${dataset_args[@]}" --notebooks "$@")"
    printf '%s\t%s\t%s\n' "$cache_dataset" "$*" "$cache_job_id" | tee -a "$manifest"
    cache_job_ids+=("${cache_job_id%%;*}")
}

for dataset in gsm8k hellaswag; do
    submit "$dataset" profiles convergence stages_of_inference
    for notebook in isolating_stability entropy_lens cyclic_similarity bos_comparison; do
        submit "$dataset" "$notebook" "$notebook"
    done
done

for notebook in limiting_behaviour multiple_choice initialisation trajectories neuron_detection; do
    submit shared "$notebook" "$notebook"
done

if "$render_figures"; then
    dependencies="$(IFS=:; printf '%s' "${cache_job_ids[*]}")"
    figure_job_id="$(sbatch --parsable --dependency="afterok:${dependencies}" \
        --kill-on-invalid-dep=yes slurm_jobs/render_figures.job \
        --cache-dir "$cache_directory")"
    printf 'both\tfigures\t%s\n' "$figure_job_id" | tee -a "$manifest"
fi
