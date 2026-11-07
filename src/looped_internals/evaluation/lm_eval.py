"""BridgeLM: lm-eval TemplateLM wrapper around TransformerBridge.

Drives evaluation through ``TransformerBridge.run_with_cache(accumulate=True)``,
capturing activations for all forward passes and computing per-batch
:class:`~looped_internals.results.ModelRunData` objects.

Usage::

    import looped_internals
    from looped_internals.models import load_model
    from looped_internals.evaluation.lm_eval import BridgeLM, get_task_manager
    import lm_eval

    bridge, structure = load_model("ouro_1_4b", num_steps=8)
    lm = BridgeLM(bridge, structure, batch_size=1, compute_metrics=True)
    results = lm_eval.simple_evaluate(
        model=lm, tasks=["gsm8k_likelihood"], task_manager=get_task_manager()
    )
    model_data = lm.all_model_data  # list[ModelRunData]
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from lm_eval.api.model import TemplateLM
from lm_eval.tasks import TaskManager
from tqdm.auto import tqdm

from looped_internals.capture import compute_model_run_data, metrics_names_filter
from looped_internals.models.tokenization import BOSPolicy, encode_with_bos_policy
from looped_internals.results import ModelRunData
from looped_internals.structure import ModelLayerStructure

logger = logging.getLogger(__name__)


def get_task_manager() -> TaskManager:
    """Index built-in lm-eval tasks and this package's likelihood tasks.

    Resolve packaged YAMLs independently of the current working directory.
    Pass the returned manager to ``simple_evaluate``, or call its ``load`` method.
    """
    return TaskManager(include_path=str(Path(__file__).resolve().parent / "tasks"))


class BridgeLM(TemplateLM):
    """lm-eval ``TemplateLM`` subclass backed by a :class:`TransformerBridge`.

    Each call to :meth:`_loglikelihood_tokens` runs a batched forward pass
    through the bridge with ``accumulate=True``, extracts log-probabilities for
    continuation tokens, and optionally computes :class:`ModelRunData` metrics
    for every sample in the batch.

    Args:
        bridge: A ``TransformerBridge`` instance (patches already applied).
        structure: Block layout descriptor for the model.
        batch_size: Number of (context, continuation) pairs per forward pass.
            All pairs in a batch must have equal total token lengths; otherwise
            a ``ValueError`` is raised before inference. Use 1 for variable-length
            inputs, since padding is not safely handled by inference or metrics.
        compute_metrics: If ``True``, run ``compute_model_run_data`` for each
            sample and store in :attr:`all_model_data`.
        compute_differences: If ``True``, also compute difference metrics
            (to_final_state_* and successive_recurrence_*) across the full execution sequence.
        compute_residual_entropy: If ``True``, compute effective-rank entropy
            of the residual stream at each virtual layer and store in
            :attr:`PerVirtualLayerMetrics.residual_entropy`.
        residual_entropy_components: If given, use a randomised truncated SVD
            with this many components when computing residual entropy.
            ``None`` (default) uses the full float32 Gram spectrum.
        forward_kwargs: Extra keyword arguments forwarded to ``bridge.forward``
            (e.g. ``num_steps=8`` to override Raven's loaded default).
        bos_policy: ``native`` follows the attached checkpoint tokenizer.
            ``bos``/``no_bos`` explicitly add/remove its automatic leading BOS
            while preserving all other input tokens; used for paired ablations.
    """

    def __init__(
        self,
        bridge,
        structure: ModelLayerStructure | None = None,
        batch_size: int = 1,
        compute_metrics: bool = True,
        compute_pairwise: bool = False,
        compute_differences: bool = False,
        compute_residual_entropy: bool = False,
        residual_entropy_components: int | None = None,
        forward_kwargs: dict | None = None,
        bos_policy: BOSPolicy = "native",
    ) -> None:
        super().__init__()
        self.bridge = bridge
        self.structure = structure
        self._batch_size = batch_size
        self.compute_metrics = compute_metrics
        self.compute_pairwise = compute_pairwise
        self.compute_differences = compute_differences
        self.compute_residual_entropy = compute_residual_entropy
        self.residual_entropy_components = residual_entropy_components
        self.forward_kwargs = forward_kwargs or {}
        if bos_policy not in {"native", "bos", "no_bos"}:
            raise ValueError("bos_policy must be native, bos, or no_bos")
        self.bos_policy = bos_policy
        self._all_model_data: list[ModelRunData] = []

        # Resolve device from bridge parameters
        try:
            self._device = next(bridge.original_model.parameters()).device
        except StopIteration:
            self._device = torch.device("cpu")

    # ---- TemplateLM abstract interface ------------------------------------

    @property
    def eot_token_id(self) -> int:
        if self.bridge.tokenizer is not None:
            eos = self.bridge.tokenizer.eos_token_id
            if eos is not None:
                return eos
        return 0

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        **kwargs,
    ) -> list[int]:
        if self.bridge.tokenizer is None:
            raise RuntimeError("BridgeLM requires a tokenizer for text input.")
        # True enables the checkpoint's own special-token processing. It does
        # not imply BOS: e.g. Ouro's published tokenizer does not prepend it.
        effective = True if add_special_tokens is None else add_special_tokens
        return encode_with_bos_policy(
            self.bridge.tokenizer,
            string,
            self.bos_policy,
            add_special_tokens=effective,
        )

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        **kwargs,
    ) -> list[tuple[float, bool]]:
        """Score log-likelihood of continuation tokens.

        Args:
            requests: Each element is ``((context_str, cont_str), ctx_tokens,
                cont_tokens)``.

        Returns:
            List of ``(sum_log_prob, is_greedy)`` per request.
        """
        results: list[tuple[float, bool]] = []

        n_batches = (len(requests) + self._batch_size - 1) // self._batch_size
        batch_iter = tqdm(
            range(0, len(requests), self._batch_size),
            desc="Scoring batches",
            unit="batch",
            total=n_batches,
            disable=disable_tqdm,
        )
        for batch_start in batch_iter:
            batch = requests[batch_start : batch_start + self._batch_size]
            batch_results = self._score_batch(batch)
            results.extend(batch_results)

        return results

    def _score_batch(
        self,
        batch: list[tuple[tuple[str, str], list[int], list[int]]],
    ) -> list[tuple[float, bool]]:
        """Score an equal-length batch and optionally collect per-sample metrics.

        Parameters
        ----------
        batch : list of tuple
            Nonempty list of ``((context, continuation), context_tokens,
            continuation_tokens)`` entries. Combined context and continuation
            token counts must be equal across entries.

        Returns
        -------
        list of tuple of (float, bool)
            Summed continuation log-probability and whether every scored token
            is a greedy prediction, in input order. A continuation token without
            a preceding context position is skipped.

        Raises
        ------
        ValueError
            If combined token lengths differ. The check occurs before inference
            because padding is not safely masked by inference or metric collection.

        Notes
        -----
        When metrics are enabled and a layer structure is available, append one
        ``ModelRunData`` per sample to ``all_model_data``.
        """
        ctx_cont_tokens = [
            ctx_tokens + cont_tokens for (_, _), ctx_tokens, cont_tokens in batch
        ]
        lengths = [len(tokens) for tokens in ctx_cont_tokens]
        if len(set(lengths)) > 1:
            raise ValueError(
                "BridgeLM cannot batch sequences with different total token "
                f"lengths (context + continuation): {lengths}. This would require "
                "padding, which is not safely masked during inference or metric "
                "computation and can compromise results. Use batch_size=1 or "
                "batch sequences with equal total token lengths."
            )

        input_ids = torch.tensor(ctx_cont_tokens, dtype=torch.long, device=self._device)

        # Forward pass — only cache the hooks capture/extraction.py actually reads.
        # When compute_metrics is False skip the cache entirely.
        with torch.no_grad():
            if self.compute_metrics:
                logits, cache = self.bridge.run_with_cache(
                    input_ids,
                    accumulate=True,
                    names_filter=metrics_names_filter,
                    device="cpu",
                    **self.forward_kwargs,
                )
            else:
                output = self.bridge.forward(input_ids, **self.forward_kwargs)
                logits = output
                cache = None

        # logits: (B, T, vocab)
        if hasattr(logits, "logits"):
            logits = logits.logits
        log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, T, vocab)

        results: list[tuple[float, bool]] = []
        for i, ((_, _), ctx_tokens, cont_tokens) in enumerate(batch):
            cont_start = len(ctx_tokens)

            # log_probs at position t predicts token at position t+1
            # so to score cont_tokens[j], look at log_probs[..., cont_start+j-1, cont_tokens[j]]
            sum_log_prob = 0.0
            is_greedy = True

            for j, tok in enumerate(cont_tokens):
                pred_pos = cont_start + j - 1
                if pred_pos < 0:
                    continue
                lp = log_probs[i, pred_pos, tok].item()
                sum_log_prob += lp
                # Check greediness: is this token the argmax?
                if log_probs[i, pred_pos].argmax().item() != tok:
                    is_greedy = False

            results.append((sum_log_prob, is_greedy))

        # Optionally collect metrics per sample
        if self.compute_metrics and self.structure is not None:
            run_data_list = compute_model_run_data(
                cache=cache,
                input_ids=input_ids,
                structure=self.structure,
                compute_pairwise=self.compute_pairwise,
                compute_differences=self.compute_differences,
                compute_residual_entropy=self.compute_residual_entropy,
                residual_entropy_components=self.residual_entropy_components,
                verbose=True,
            )
            self._all_model_data.extend(run_data_list)

        return results

    # ---- Not used for loglikelihood tasks ---------------------------------

    def loglikelihood_rolling(
        self, requests, disable_tqdm: bool = False
    ) -> list[float]:
        raise NotImplementedError(
            "BridgeLM does not implement loglikelihood_rolling. "
            "Use loglikelihood for scored continuation tasks."
        )

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        raise NotImplementedError("BridgeLM does not implement generate_until.")

    # ---- Accessors --------------------------------------------------------

    @property
    def all_model_data(self) -> list[ModelRunData]:
        """All :class:`ModelRunData` objects collected during evaluation."""
        return self._all_model_data

    @property
    def batch_size(self) -> int:
        return self._batch_size
