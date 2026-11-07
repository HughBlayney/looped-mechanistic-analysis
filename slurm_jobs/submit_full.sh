#!/usr/bin/env bash
# Submit from the repository root. An optional first argument reuses a preflight ID.
set -euo pipefail
mkdir -p slurm_out experiments/runs
preflight="${1:-}"
if [[ -z "$preflight" ]]; then
    preflight="$(sbatch --parsable slurm_jobs/preflight.job)"
fi
manifest="experiments/runs/submissions-$(date -u +%Y%m%dT%H%M%SZ).tsv"
printf 'experiment\tjob_id\npreflight\t%s\n' "$preflight" > "$manifest"
# Queue the longer analyses first so the short illustrations can fill later slots.
for notebook in convergence limiting_behaviour stages_of_inference mixing_correlation isolating_stability multiple_choice cyclic_similarity bos_comparison entropy_lens neuron_detection trajectories initialisation; do
    job="$(sbatch --parsable --dependency="afterok:${preflight}" --kill-on-invalid-dep=yes "slurm_jobs/${notebook}.job")"
    printf '%s\t%s\n' "$notebook" "$job" | tee -a "$manifest"
done
printf 'Manifest: %s\n' "$manifest"
