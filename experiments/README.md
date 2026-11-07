# Experiments

Each notebook contains its scientific reductions and plots. `support.py` handles
shared model selections, likelihood-task collection, seeds, and output paths;
`behavior_support.py` handles the GSM8K prompt protocol. These are repository
experiment helpers, not part of the installed library. `weight_support.py` loads
and releases checkpoints for weight-only analyses without dataset or rollout settings.

`paper_models.py` defines the ordered `PAPER_MODEL_GROUPS` used by convergence,
cyclic similarity, trajectories, stages of inference, multiple choice, entropy lens,
and neuron detection:

| Group | Models, in panel order |
| --- | --- |
| `main` | Ouro 1.4B, Huginn-0125, Retrofitted Llama, Parcae 770M |
| `appendix` | Retrofitted OLMo, Retrofitted TinyLlama, Parcae 140M, Parcae 370M |
| `all` | Main row followed by appendix row: eight distinct models |

The main Parcae model is `parcae_770m`, with the trained `parcae_770m_rec1`
control from `blayney/parcae-770m-rec1`. The appendix stays fixed; Parcae 1.3B and
Ouro 2.6B are outside these groups. Collection deduplicates model–recurrence pairs, while figures retain
the eight panel positions. Convergence and cyclic similarity call `paper_model_pairs`
after parameter overrides to resolve collection and figure selections together.
Trajectories uses the `main` group at 32 recurrences, producing separate figures
for each model. Stages of inference reuses the groups at 32 recurrences (four for
Ouro), plus 128-recurrence main-model runs. Multiple choice uses the groups at
32 recurrences (eight for Ouro), with matched one-pass controls. Entropy lens
uses the same groups, recurrence counts, and controls. The general checkpoint
registry remains in `support.py`.

## Persistent experiment cache

Notebook caches default to **`/data-gauss/hugbla/looped-internals-cache`**. Set a
persistent directory shared by your interactive sessions and SLURM jobs with:

```bash
export LOOPED_INTERNALS_CACHE_DIR=/data-gauss/hugbla/looped-internals-cache
export LOOPED_INTERNALS_CACHE_MODE=reuse
```

Put these exports in your shell startup file or export them before `sbatch`;
SLURM inherits the submission environment. The notebook parameters `cache_dir`
and `cache_mode` override the environment. For example:

```bash
sbatch slurm_jobs/convergence.job gsm8k \
  -p cache_dir /data-gauss/hugbla/looped-internals-cache -p cache_mode reuse
```

The cache directory is created on first use. `output_dir` still controls the
separate timestamped run archive and does not relocate the cache.

| `cache_mode` | Behaviour |
| --- | --- |
| `reuse` (default) | Load completed entries; resume partial entries; compute missing entries. |
| `require` | Load completed entries and fail on a missing entry before loading its model. Use for plotting without inference. |
| `off` | Compute afresh without reading or writing the persistent cache. Ordinary run exports still occur. |

A cache hit occurs before model loading. Keys include the checkpoint and tokenizer
revisions (or content hashes for local checkpoint files), precision, seed, actual
input texts and selection/order, recurrence count, interventions, measurement
options, and relevant dependency versions. Local checkpoints must remain unchanged
during a running notebook. Models are pinned to the resolved Hub revisions when
loaded, including Parcae's separate tokenizer repository. Figure grouping, colours,
output paths, and the quick/full label do not invalidate numerical results.
Quick/full still controls sample defaults and paper exports: a quick run never
updates the paper directory, including when it reuses a full run's measurements.

`analysis_version` belongs to each notebook. **Bump it when changing its scientific
reduction or input protocol**, but not when changing only plotting. Bump
`COLLECTION_VERSION` in `experiments/cache.py` when changing shared collection or
tokenization semantics. Versions deliberately replace automatic hashing of entire
notebooks or the repository, so switching `MAIN_PARCAE_MODEL` to `parcae_770m`
collects the new model/control without invalidating the other paper models.

Likelihood and entropy collections checkpoint after each reduced example and
restore the random state after reloading the model, mapping the saved CUDA stream
to the model's current GPU when resuming on another device. They retain the original
seed-once convention; skipped examples do not change later stochastic states.
Convergence also saves existing small layer measurements that stages of inference
can reuse for its matching long runs, without extra forwards or spectral entropy.
Multiple choice checkpoints probe weights, optimizer state, and random state after
each training batch, saves finished probes before evaluation, and checkpoints test
counts after each example. The stability sweep checkpoints completed example/depth
contributions. Limiting behaviour saves each model/prompt/example independently;
initialisation saves each measurement/configuration/seed; trajectories and weight
analysis save each model. An interrupted individual trajectory or weight analysis
restarts that model, while completed models remain reusable.

Each entry has a readable `request.json`, an immutable completed `result.pt`, and,
while incomplete, either per-example files under `parts/` or a replaceable
`progress.pt` for training/sweep state. Long collectors write each completed example
once, instead of rewriting their entire history. Writes use a temporary file and an
atomic rename; an interrupted write cannot be mistaken for a complete result.
Per-entry filesystem locks prevent duplicate writers and are released on process
exit. A successfully finalized entry removes its temporary progress checkpoints.
Each run's `cache.json` links to the exact cache entries and their checkpoint identities.
No old datasets, checkpoints, or run archives are migrated or overwritten.
Limiting-behaviour histories and stability model exports in new run directories
link to completed cache files to avoid duplicating them on every plotting rerun.
Keep the cache while retaining those runs; use `rsync -aL` to make a standalone copy.
There is no automatic cache eviction. Allow roughly 10–20 GiB for compact results,
restart files, and a few versions, excluding downloaded datasets and model weights.

### Reuse an existing completed run

Convergence, cyclic similarity, stages of inference, entropy lens, trajectories,
and neuron detection accept `source_run`, the path to a completed run directory:

```bash
sbatch slurm_jobs/entropy_lens.job gsm8k \
  -p source_run experiments/results/entropy_lens/gsm8k/full-20260921T161148.312864Z
```

The notebook checks the recorded scientific settings and dependency versions,
reuses matching model pairs, and collects any newly requested pairs normally.
This allows an existing seven-model run to supply the old panels after adding
770M. Cyclic similarity can reuse its already averaged matrices. The original
source directory remains unchanged and is recorded in the new run's metadata.

Selecting `source_run` explicitly asserts that its scientific implementation is
suitable for the current figures. Historical runs did not record reduction
versions or immutable checkpoint identities consistently; their summaries are
therefore kept as explicitly attributed historical inputs, **not inserted into
the fresh-result cache under today's checkpoint identity**. Setting `source_run`
does not override mismatching recorded parameters. Do not select results produced
under an older tokenization or measurement convention.

### Populate the cache before making figures

The optional warm-up job executes the notebooks' own collection cells, stopping
at their `collection-end` tags. It does not duplicate scientific code, render
figures, or update the paper directory:

```bash
mkdir -p slurm_out
sbatch slurm_jobs/populate_cache.job
# Or fill just the new/missing work for selected analyses:
sbatch slurm_jobs/populate_cache.job --notebooks convergence stages_of_inference
```

GSM8K is the default for convergence, stages of inference, cyclic similarity,
entropy lens, BOS comparison, and isolating stability. To include HellaSwag, or
populate only its entries for those notebooks:

```bash
sbatch slurm_jobs/populate_cache.job --datasets gsm8k hellaswag
sbatch slurm_jobs/populate_cache.job --datasets hellaswag
```

`--datasets` applies only to those six notebooks; other notebooks run once.
Multiple choice retains its own benchmark list, including HellaSwag. Limiting
behaviour remains a GSM8K analysis. The six likelihood analyses use HellaSwag's
labelled validation split, with the activity/context and the correct ending
processed by the packaged likelihood task. Sampling, cache entries, run archives,
and paper figure directories stay separate for each dataset.

Convergence computes only the six same-layer final/successive curves used by its
figures. It skips cross-layer reference comparisons, unrelated stage metrics and
likelihood scoring. It retains only recurrent-layer activations by default.

When both convergence and stages are selected, the runner reads their tagged
`configuration-end` cells before inference. It finds pairs with matching dataset,
sample selection, seed, recurrence count, precision, device type and tokenization.
For matching stages requests that do not need residual entropy, it passes
`cache_profile_pairs` to convergence. That forward pass also captures prelude/coda
activations and saves the small per-head/per-layer profiles stages needs. Stages
then reduces those profiles without loading the model again. No raw activation
history is saved. Different recurrence counts remain separate forward passes.

With the defaults, this shares the main models' 128-recurrence profiles. A
convergence-only cache job does not collect them. An explicit `cache_profile_pairs`
list in convergence's parameter overrides replaces automatic planning; an empty
list disables sharing. Historical stages selected through `source_run` do not
request extra profiles.

Convergence summaries and layer profiles have separate cache namespaces. The
convergence output format and scientific meaning are unchanged, so existing
complete summaries remain reusable. If convergence is already cached without
profiles, stages collects its missing measurements itself; convergence is not
rerun. A resumed run publishes shared profiles only when it has profiles for
every selected example. `cache_mode="require"` can reduce complete shared profiles
without inference or writing new cache entries, even if a stages summary has not
yet been saved.

The runner orders convergence before stages even when they are listed in reverse.
All entries are incremental; resubmit after a time limit to continue. This serial
job is optional: `submit_full.sh` runs independent notebooks concurrently after
preflight. Populate the cache jointly first to share inference across those jobs.

Use `--parameters settings.json` to pass a JSON mapping from notebook names to
parameter overrides, such as `source_run` or a model subset. A notebook-specific
`dataset` replaces `--datasets` for that notebook. A `source_run` must match the
dataset; when running both datasets, select historical sources in separate
invocations. For example:

```json
{
  "entropy_lens": {
    "source_run": "experiments/results/entropy_lens/gsm8k/full-20260921T161148.312864Z"
  }
}
```

To regenerate figures from a populated cache, execute the notebooks with
`cache_mode="require"`, or use the same runner without `--collect-only`:

```bash
.venv/bin/python -m experiments.run_cached --cache-mode require \
  --notebooks convergence stages_of_inference --device cuda --dtype bfloat16
.venv/bin/python -m experiments.run_cached --status
```

The runner accepts the same `--datasets` selection when regenerating figures.

For CPU-only plotting of CUDA-collected results, retain `device="cuda"` and the
original `dtype` in the parameters: they identify the saved computation, and a
complete hit performs no CUDA inference. `require` reports the first missing
entry, while `--status` lists completed, partial, and pending cache entries.
The runner does not create an executed `.ipynb`; use Papermill when that additional
record is wanted. Carry the same `source_run` override into subsequent plotting
runs when reusing historical results.

### Submit cache jobs in parallel

From the repository root, run:

```bash
bash slurm_jobs/submit_cache.sh
```

This submits **15 one-GPU jobs**: convergence and stages together for each of
GSM8K and HellaSwag; separate jobs for cyclic similarity, entropy lens, BOS
comparison and stability on each dataset; and one job each for multiple choice,
limiting behaviour, initialisation, trajectories and neuron detection. SLURM
queues jobs according to available resources and your account limits. These jobs
populate caches only; they do not render or publish paper figures.

The script explicitly selects `cache_mode=reuse` and defaults to
`/data-gauss/hugbla/looped-internals-cache`. Set `LOOPED_INTERNALS_CACHE_DIR` before
launching to use another shared location. Each invocation prints and retains a
unique `experiments/runs/cache-submissions-*.tsv` manifest with every submitted
job ID. If a submission fails partway through, the manifest keeps the IDs already
submitted. The script does not cancel existing jobs: use this layout instead of
an overlapping all-notebooks population job to avoid GPUs waiting on cache locks.

### Generate the final figures

From the repository root, submit collection and dependent figure rendering with:

```bash
export LOOPED_INTERNALS_CACHE_DIR=/data-gauss/hugbla/looped-internals-cache
bash slurm_jobs/submit_cache.sh --figures
```

This queues the same 15 cache jobs for **both GSM8K and HellaSwag**, followed by one
**CPU-only** `render_figures.job`. The figure job has an `afterok` dependency on all
15 jobs and is cancelled if a prerequisite fails. Its ID is included in the same
submission manifest. Once all required results exist, it executes every notebook
in full mode and refreshes `plots/mechanistic_analysis/` and the new run archives.
The dataset-selectable figures run for both datasets; multiple choice retains
ARC-Easy, ARC-Challenge, HellaSwag, and WinoGrande; other shared notebooks run once.
The existing `submit_full.sh` defaults to GSM8K and is not this combined workflow.

Existing matching entries are reused before model loading. New expensive work is
the 770M runs for convergence, stages, cyclic similarity, entropy lens, multiple
choice, trajectories, and neuron detection, plus the matched control where used.
BOS comparison, isolating stability, limiting behaviour, and initialisation keep
their selections. Cache keys are per model/measurement, so the paper-group change
does not invalidate 370M or other completed work. Missing or partial older work
is also completed. Reuse requires unchanged inputs, seeds, precision, analysis
versions, dependencies, and checkpoint/tokenizer revisions.

If a previous multiple-choice population job is still running, set its job ID for
this invocation (replace `12345` with that ID):

```bash
LOOPED_INTERNALS_MCQ_AFTER_JOB=12345 bash slurm_jobs/submit_cache.sh --figures
```

Only the new multiple-choice job waits for that job to end; other collection jobs
can start immediately. The dependency is `afterany`, allowing the new job to resume
checkpoints even if the previous job fails or times out. Omit this setting when no
previous multiple-choice job is running.

A failed cache job can be resubmitted safely; after all caches are complete, submit
rendering separately (also useful when only plotting code changes):

```bash
sbatch slurm_jobs/render_figures.job
```

Rendering always uses `cache_mode=require`: missing results cause an error instead
of inference. Its `device=cuda` and `dtype=bfloat16` settings identify the existing
GPU-produced results; cached tensors and figures are processed on the CPU. It
requests no GPU.

To generate an initial set while multiple choice is still running, select the
other notebooks once their caches are complete:

```bash
sbatch --job-name=paper-figures-initial slurm_jobs/render_figures.job \
  --notebooks convergence stages_of_inference cyclic_similarity entropy_lens \
  trajectories neuron_detection initialisation isolating_stability \
  bos_comparison limiting_behaviour
```

This renders both datasets where applicable and exports the same full-resolution
paper PDFs. It does not wait for multiple choice. The already queued final render
can still refresh the complete set when all collection jobs finish.

## Run locally

From the repository root:

```bash
uv sync --frozen --dev --group experiments
.venv/bin/python -m ipykernel install --prefix .venv \
    --name looped-internals --display-name 'Looped Internals'
mkdir -p experiments/runs
.venv/bin/python -m papermill experiments/convergence.ipynb \
    experiments/runs/convergence.ipynb --kernel looped-internals --cwd . \
    -p mode quick -p dataset gsm8k
```

Open a notebook from the repository or `experiments/`; its bootstrap locates the
repository root. The tagged parameter cell is the only place to change run
settings. Papermill injects overrides immediately after that cell.

## Results

Runs create fresh directories under:

```text
experiments/results/<experiment>/<dataset>/<mode>-<UTC timestamp>/
```

Each contains resolved settings and numerical results, alongside PDF figures.
Weight-only neuron detection omits the dataset directory:
`experiments/results/neuron_detection/<mode>-<UTC timestamp>/`.
New runs save PDFs only. Executed notebooks live in `experiments/runs/`.
Both directories are ignored by Git. Nothing overwrites previous run archives.

At the end of a successful `mode="full"` run, the notebook also copies every PDF
into the stable paper export directory. This policy applies to both manual and
SLURM runs, including convergence and cyclic similarity. Quick runs save their
results locally without copying any paper figures. For convergence and cyclic
similarity, `run.json` records the resolved mode, sample limit, and complete list
of model–recurrence pairs:

```text
plots/mechanistic_analysis/<notebook>/<dataset>/<plot_name>.pdf
```

`<notebook>` is the notebook filename without `.ipynb`. Dataset-specific notebooks
use `gsm8k` or `hellaswag` subdirectories, keeping both datasets side by side.
Initialisation, fixed-sequence trajectories, weight-only neuron detection, and
multiple-choice figures combining multiple datasets omit the dataset directory.
For example:

```text
plots/mechanistic_analysis/convergence/gsm8k/fixed_point_norms_main.pdf
plots/mechanistic_analysis/convergence/hellaswag/fixed_point_norms_main.pdf
plots/mechanistic_analysis/initialisation/summarised_stability.pdf
```

Nested run figures use filename prefixes in the paper directory:
`subdirectory/<metric>.pdf` becomes `subdirectory-<metric>.pdf`. Quick runs and runs
that fail before the final export leave the paper directory alone. Changing
`output_dir` relocates the run archive; the paper directory remains under the
repository's `plots/` directory and is also ignored by Git. You can rsync
`plots/mechanistic_analysis/` directly into the LaTeX repository.

The latest completed full run replaces matching paper PDFs, including when
`models`, `limit`, or other settings override the full preset. Concurrent runs
for the same notebook and dataset therefore share a destination. Files absent
from a later run are retained, so renamed or removed plots may leave older PDFs
to clean up manually. Existing PDFs, PNGs, and other historical artifacts are
not changed by this update.

Pairwise matrices, PCA coordinates, probe weights and compact
similarity trajectories are saved where relevant; ordinary collection releases
the activation cache after each example. Likelihood experiments use lm-eval's
packaged task prompts and `BridgeLM.loglikelihood`, reducing and releasing each
`ModelRunData` immediately. The seeded sample is evaluated in dataset order;
saved indices follow that order. Random generators are seeded once per evaluation, rather than per
example. Large raw histories are retained only for the trajectory/floorplan
examples that need them.
