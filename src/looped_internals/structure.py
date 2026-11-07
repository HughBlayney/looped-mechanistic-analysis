"""Layer identities and virtual layers for looped transformer models.

A layer is a distinct set of transformer-block weights. A virtual layer is
one application of a layer in a forward pass. Reusing two layers for three
recurrences therefore gives two layers and six virtual layers. Independent
parameter sets remain separate layers even if their weight values are equal.

These descriptors connect layer identities to TransformerLens hook prefixes
and enumerate virtual layers in forward-pass order. Hook names retain the
model adapters' native ``blocks`` naming.

Confirmed hook naming from integration tests:
- Raven:  ``prelude_blocks.{i}.hook_out``, ``core_blocks.{i}.hook_out`` (+.iter_N),
          ``coda_blocks.{i}.hook_out``
- Ouro:   ``blocks.{i}.hook_out`` (+.iter_N for all blocks)
- GPT-2 / non-recurrent: ``blocks.{i}.hook_out`` (no iterations)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LayerGroup:
    """A contiguous group of layers sharing the same hook prefix.

    Attributes:
        hook_prefix: Cache key prefix, e.g. ``"blocks"`` or ``"core_blocks"``.
        n_layers: Number of distinct layer parameter sets in this group.
        is_recurrent: Whether the group is applied once per recurrence,
            producing multiple virtual layers per layer. Accumulated cache
            entries identify applications with ``.iter_N`` suffixes.
    """

    hook_prefix: str
    n_layers: int
    is_recurrent: bool


@dataclass
class ModelLayerStructure:
    """Ordered layer groups and their recurrence layout.

    Examples::

        # Raven (2 prelude + 4 core + 2 coda, 8 recurrences):
        raven_structure = ModelLayerStructure(groups=[
            LayerGroup("prelude_blocks", 2, False),
            LayerGroup("core_blocks", 4, True),
            LayerGroup("coda_blocks", 2, False),
        ])

        # Ouro (24 layers, all recurrent):
        ouro_structure = ModelLayerStructure(groups=[
            LayerGroup("blocks", 24, True),
        ])

        # GPT-2 (12 layers, non-recurrent):
        gpt2_structure = ModelLayerStructure(groups=[
            LayerGroup("blocks", 12, False),
        ])

    Attributes:
        groups: Ordered list of :class:`LayerGroup` objects.
    """

    groups: list[LayerGroup] = field(default_factory=list)

    def virtual_layers(self, n_recurrences: int = 1):
        """Yield virtual layers in group, recurrence, then layer order.

        Recurrent groups contribute ``n_layers * n_recurrences`` virtual
        layers; other groups contribute one virtual layer per layer.
        """
        layer_offset = 0
        virtual_layer_index = 0
        for group in self.groups:
            for iteration in range(n_recurrences if group.is_recurrent else 1):
                for layer in range(group.n_layers):
                    yield VirtualLayer(
                        layer_offset + layer,
                        virtual_layer_index,
                        f"{group.hook_prefix}.{layer}",
                        iteration if group.is_recurrent else None,
                    )
                    virtual_layer_index += 1
            layer_offset += group.n_layers

    def hook_names(self, suffix: str = "hook_out") -> list[str]:
        """Return all base hook names (without ``.iter_N``) for a given suffix.

        Args:
            suffix: The hook suffix, e.g. ``"hook_out"`` or ``"attn.hook_q"``.

        Returns:
            Ordered list of hook names, one per layer.
        """
        names = []
        for group in self.groups:
            for i in range(group.n_layers):
                names.append(f"{group.hook_prefix}.{i}.{suffix}")
        return names

    def iter_layers(self):
        """Yield each ``(group, local_layer_index)`` once, in model order."""
        for group in self.groups:
            for i in range(group.n_layers):
                yield group, i


@dataclass(frozen=True)
class VirtualLayer:
    """One application of a layer at a position in the forward pass.

    Attributes
    ----------
    layer_index : int
        Zero-based identity of the layer's weights across all layer groups.
        Recurrent applications of the same weights share this index.
    virtual_layer_index : int
        Zero-based position of this application in the virtual layer sequence.
    hook_prefix : str
        Native hook prefix for the layer, such as ``"core_blocks.0"``.
        This is a prefix, not a complete cache key.
    iteration : int or None
        Zero-based recurrence index, or ``None`` for an unsuffixed cache entry.

    Notes
    -----
    Use ``residual_key`` and ``attn_key`` to look up this application's layer
    output and attention pattern. ``q_key``, ``k_key``, and ``v_key`` name
    separate projection outputs; ``qkv_key`` names a fused projection output.
    All keys include the iteration suffix when present. They follow the
    supported bridge hook conventions, but do not guarantee that a tensor
    was cached: attention is optional, and adapters may expose separate or
    fused Q/K/V projections.
    """

    layer_index: int
    virtual_layer_index: int
    hook_prefix: str
    iteration: int | None

    @property
    def cache_suffix(self) -> str:
        """Return ``.iter_N`` for an accumulated entry, or an empty string."""
        return "" if self.iteration is None else f".iter_{self.iteration}"

    @property
    def residual_key(self) -> str:
        """Return the cache key for the residual stream after this application."""
        return f"{self.hook_prefix}.hook_out{self.cache_suffix}"

    @property
    def attn_key(self) -> str:
        """Return the cache key for this application's attention pattern."""
        return f"{self.hook_prefix}.attn.hook_pattern{self.cache_suffix}"

    @property
    def q_key(self) -> str:
        """Return the cache key for the separate query projection output."""
        return f"{self.hook_prefix}.attn.q.hook_out{self.cache_suffix}"

    @property
    def k_key(self) -> str:
        """Return the cache key for the separate key projection output."""
        return f"{self.hook_prefix}.attn.k.hook_out{self.cache_suffix}"

    @property
    def v_key(self) -> str:
        """Return the cache key for the separate value projection output."""
        return f"{self.hook_prefix}.attn.v.hook_out{self.cache_suffix}"

    @property
    def qkv_key(self) -> str:
        """Return the cache key for the fused Q/K/V projection output."""
        return f"{self.hook_prefix}.attn.qkv.hook_out{self.cache_suffix}"
