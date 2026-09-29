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



@pytest.mark.parametrize("backend,module", [("hf", "peft"), ("mlx", "mlx_lm")])
def test_dry_run_evaluates_and_logs_often(make_cfg, backend, module):
    pytest.importorskip(module)
    cfg = make_cfg("training.max_steps=-1", "training.eval_every_steps=50", "logging.log_every_steps=10",
                   run_name=f"dry-{backend}", backend=backend)
    run_dir = run_training(cfg, dry_run_steps=8)
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    eval_steps = sorted({r["step"] for r in rows if "eval/loss" in r})
    train_steps = sorted({r["step"] for r in rows if "train/loss" in r})
    assert eval_steps == [0, 2, 4, 6, 8]  # baseline + every dry_run_steps // 4
    assert len(train_steps) == 8          # every dry_run_steps // 10 (min 1)
