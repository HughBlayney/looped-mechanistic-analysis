# Packaged likelihood tasks

These tasks run with the published `lm-eval` package. Load them with
`looped_internals.get_task_manager()` and pass that manager to
`lm_eval.simple_evaluate`. For direct loading, call the manager's
`load([task_name])["tasks"]`. The manager also includes lm-eval's built-in tasks.
For the lm-eval CLI, pass this directory with `--include_path`.

Each example produces one `loglikelihood` request:

| Task | Context | Continuation |
| --- | --- | --- |
| `gsm8k_likelihood` | `Question: {question}\nAnswer:` | The full provided answer, including reasoning; original spacing is preserved. |
| `hellaswag_likelihood` | Preprocessed activity and context | A space followed by the preprocessed correct ending (`choices[gold]`). |
| `c4_likelihood` | Empty string | The detokenized document text. |

All three likelihood tasks explicitly set `num_fewshot: 0`. They score the
provided continuation without demonstrations; lm-eval therefore has no
unspecified few-shot default to override during the zero-shot experiment runs.

The `acc` metric on GSM8K and HellaSwag is an exact greedy match of all target
tokens. It is not standard GSM8K answer accuracy or HellaSwag four-choice
accuracy. These tasks primarily support collecting activations on provided
continuations. C4 aggregates word/byte perplexity and bits per byte, unpacking
the `(log_probability, is_greedy)` response from `loglikelihood`.

## Versions and existing data

- HellaSwag version 2 scores the correct ending. Version 1 scored the label
  digit, so its cached activations are not interchangeable with version 2.
  `create_paper_data_hellaswag.py` writes `model_data_hellaswag_v2_lim*.pt` to
  keep these runs separate. The experiment notebook hashes actual inputs and
  rejects stale cache entries automatically.
- C4 version 1 fixes result unpacking for `loglikelihood` and uses the text
  field for decontamination queries.
- GSM8K retains version 3 and its original context/target formatting. Its Hub
  dataset ID is `openai/gsm8k`, configuration `main`. The obsolete unnamespaced
  ID `gsm8k` can only fall back to an existing local cache. The first load under
  the current ID may download a new cache; existing cached datasets are untouched.

YAMLs and helper code were adapted from EleutherAI/lm-evaluation-harness commit
`5d7dc4c0`, including the local likelihood variants. The upstream MIT license is
in `LICENSE.md`. Helpers live beside their YAMLs because lm-eval resolves
`!function` paths relative to each YAML file.
