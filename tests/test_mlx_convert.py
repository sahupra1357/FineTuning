"""mlx_lm.convert must receive a local path (see mlx_backend._local_snapshot)."""

import pytest

pytest.importorskip("mlx_lm")

from finetune.backends import mlx_backend  # noqa: E402


def test_local_path_is_passed_through(make_cfg, tiny_model_dir):
    cfg = make_cfg()
    assert mlx_backend._local_snapshot(cfg) == str(tiny_model_dir)


def test_repo_id_is_resolved_to_a_local_snapshot(make_cfg, monkeypatch, tmp_path):
    import huggingface_hub

    calls = {}

    def fake_snapshot_download(repo_id, revision=None, allow_patterns=None, **kw):
        calls.update(repo_id=repo_id, allow_patterns=allow_patterns, kw=kw)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    cfg = make_cfg("model.name_or_path=openai/gpt-oss-20b")
    assert mlx_backend._local_snapshot(cfg) == str(tmp_path)
    assert calls["repo_id"] == "openai/gpt-oss-20b"
    assert "model*.safetensors" in calls["allow_patterns"]
    assert "local_files_only" not in calls["kw"]  # the call that broke with newer huggingface_hub


def test_qlora_quantizes_from_local_snapshot(make_cfg, tmp_path):
    cfg = make_cfg("mac.mlx_cache_dir=" + str(tmp_path / "mlx"), method="qlora", backend="mlx")
    out = mlx_backend.resolve_model_path(cfg)
    assert (tmp_path / "mlx").exists() and out.startswith(str(tmp_path / "mlx"))


@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "4bit"])
def test_gpt_oss_moe_lora_backward(quantized):
    """Regression: gpt-oss top-k routing crashed the backward pass under LoRA
    ("[gather_axis] Cannot calculate VJP with respect to indices")."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models import gpt_oss
    from mlx_lm.tuner.utils import linear_to_lora_layers

    mlx_backend._patch_moe_topk()
    args = gpt_oss.ModelArgs(num_hidden_layers=2, num_local_experts=4, num_experts_per_tok=2, vocab_size=64,
                             hidden_size=64, intermediate_size=64, head_dim=16, num_attention_heads=4,
                             num_key_value_heads=2, layer_types=["sliding_attention", "full_attention"])
    model = gpt_oss.Model(args)
    if quantized:
        nn.quantize(model, group_size=32, bits=4)
    model.freeze()
    linear_to_lora_layers(model, 2, {"rank": 4, "scale": 2.0, "dropout": 0.0})
    x, y = mx.array([[1, 2, 3, 4, 5]]), mx.array([[2, 3, 4, 5, 6]])
    loss, grads = nn.value_and_grad(model, lambda m: nn.losses.cross_entropy(m(x), y).mean())(model)
    mx.eval(loss, grads)
    names = [k for k, _ in tree_flatten(grads)]
    assert mx.isfinite(loss).item()
    assert any("router" in k for k in names) and any("experts" in k for k in names)
