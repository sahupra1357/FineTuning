"""End-to-end runs on a tiny offline model (CPU). Each backend is skipped if not installed."""

import json

import pytest

from finetune.pipeline import run_training


def _check_run(run_dir, adapter_file):
    state = json.loads((run_dir / "run_state.json").read_text())
    assert state["status"] == "completed"
    assert state["baseline_eval_loss"] is not None and state["best_eval_loss"] is not None
    assert (run_dir / "best" / adapter_file).exists()
    assert (run_dir / "final" / adapter_file).exists()
    assert (run_dir / "report.html").exists() and (run_dir / "plots" / "loss.png").exists()
    tags = {json.loads(line)["tag"] for line in (run_dir / "samples.jsonl").read_text().splitlines()}
    assert {"baseline", "best"} <= tags
    return state


def test_hf_backend_end_to_end_and_resume(make_cfg):
    pytest.importorskip("peft")
    cfg = make_cfg(run_name="hf", backend="hf")
    run_dir = run_training(cfg)
    state = _check_run(run_dir, "adapter_model.safetensors")
    assert state["last_step"] == 6

    state["status"] = "stopped"  # pretend it was interrupted, then extend the run
    (run_dir / "run_state.json").write_text(json.dumps(state))
    run_training(make_cfg("training.max_steps=9", run_name="hf", backend="hf"))
    assert json.loads((run_dir / "run_state.json").read_text())["last_step"] == 9


def test_mlx_backend_end_to_end(make_cfg):
    pytest.importorskip("mlx_lm")
    cfg = make_cfg("lora.target_modules=[q_proj,v_proj]", run_name="mlx", backend="mlx")
    _check_run(run_training(cfg), "adapters.safetensors")

