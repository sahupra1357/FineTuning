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
