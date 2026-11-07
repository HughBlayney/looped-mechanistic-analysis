"""Tests for looped_internals.structure."""

from __future__ import annotations

from looped_internals.structure import LayerGroup, ModelLayerStructure


def test_virtual_layers_reuse_layer_identity_across_recurrences():
    structure = ModelLayerStructure(
        [
            LayerGroup("prelude_blocks", 1, False),
            LayerGroup("core_blocks", 2, True),
            LayerGroup("coda_blocks", 1, False),
        ]
    )
    virtual_layers = list(structure.virtual_layers(n_recurrences=3))

    assert sum(group.n_layers for group in structure.groups) == 4
    assert [v.layer_index for v in virtual_layers] == [0, 1, 2, 1, 2, 1, 2, 3]
    assert [v.virtual_layer_index for v in virtual_layers] == list(range(8))
    assert [v.iteration for v in virtual_layers] == [None, 0, 0, 1, 1, 2, 2, None]
    assert virtual_layers[1].hook_prefix == virtual_layers[5].hook_prefix
    assert virtual_layers[1].cache_suffix == ".iter_0"
    assert virtual_layers[5].cache_suffix == ".iter_2"
    assert virtual_layers[-1].cache_suffix == ""
    assert virtual_layers[1].residual_key == "core_blocks.0.hook_out.iter_0"
    assert virtual_layers[5].residual_key == "core_blocks.0.hook_out.iter_2"
    assert virtual_layers[-1].residual_key == "coda_blocks.0.hook_out"


class TestLayerGroup:
    def test_fields(self):
        bg = LayerGroup(hook_prefix="blocks", n_layers=12, is_recurrent=False)
        assert bg.hook_prefix == "blocks"
        assert bg.n_layers == 12
        assert not bg.is_recurrent

    def test_recurrent_flag(self):
        bg = LayerGroup(hook_prefix="core_blocks", n_layers=4, is_recurrent=True)
        assert bg.is_recurrent


class TestModelLayerStructure:
    def _raven_structure(self) -> ModelLayerStructure:
        return ModelLayerStructure(
            groups=[
                LayerGroup("prelude_blocks", 2, False),
                LayerGroup("core_blocks", 4, True),
                LayerGroup("coda_blocks", 2, False),
            ]
        )

    def _ouro_structure(self) -> ModelLayerStructure:
        return ModelLayerStructure(
            groups=[
                LayerGroup("blocks", 24, True),
            ]
        )

    def _gpt2_structure(self) -> ModelLayerStructure:
        return ModelLayerStructure(
            groups=[
                LayerGroup("blocks", 12, False),
            ]
        )

    def test_raven_hook_names(self):
        s = self._raven_structure()
        names = s.hook_names("hook_out")
        assert names[:2] == ["prelude_blocks.0.hook_out", "prelude_blocks.1.hook_out"]
        assert names[2:6] == [f"core_blocks.{i}.hook_out" for i in range(4)]
        assert names[6:] == ["coda_blocks.0.hook_out", "coda_blocks.1.hook_out"]
        assert len(names) == 8

    def test_ouro_hook_names(self):
        s = self._ouro_structure()
        names = s.hook_names("hook_out")
        assert len(names) == 24
        assert names[0] == "blocks.0.hook_out"
        assert names[-1] == "blocks.23.hook_out"

    def test_gpt2_hook_names(self):
        s = self._gpt2_structure()
        names = s.hook_names("hook_out")
        assert len(names) == 12
        assert all(n.startswith("blocks.") for n in names)

    def test_iter_layers_raven(self):
        s = self._raven_structure()
        blocks = list(s.iter_layers())
        assert len(blocks) == 8
        assert blocks[0][0].hook_prefix == "prelude_blocks"
        assert blocks[0][1] == 0
        assert blocks[2][0].hook_prefix == "core_blocks"
        assert blocks[2][0].is_recurrent
        assert not blocks[0][0].is_recurrent
        assert not blocks[6][0].is_recurrent

    def test_recurrent_groups_identified(self):
        s = self._raven_structure()
        recurrent_groups = [g for g in s.groups if g.is_recurrent]
        non_recurrent = [g for g in s.groups if not g.is_recurrent]
        assert len(recurrent_groups) == 1
        assert recurrent_groups[0].hook_prefix == "core_blocks"
        assert len(non_recurrent) == 2

    def test_ouro_all_recurrent(self):
        s = self._ouro_structure()
        assert all(g.is_recurrent for g in s.groups)

    def test_gpt2_none_recurrent(self):
        s = self._gpt2_structure()
        assert not any(g.is_recurrent for g in s.groups)

    def test_attn_hook_names(self):
        s = self._ouro_structure()
        names = s.hook_names("attn.hook_q")
        assert all("attn.hook_q" in n for n in names)
        assert len(names) == 24
