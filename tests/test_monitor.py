import math

import pytest

from finetune.config import TrainingConfig
from finetune.monitor import RunMonitor, RunState, StopTraining, TopKAdapters
from finetune.report import BAD, OK, diagnose


def test_nan_and_spike_abort():
    m = RunMonitor(TrainingConfig())
    with pytest.raises(StopTraining) as e:
        m.check_train_loss(float("nan"), 3)
    assert e.value.status == "aborted"
    m = RunMonitor(TrainingConfig(loss_spike_factor=3))
    for i in range(25):
        m.check_train_loss(1.0, i)
    with pytest.raises(StopTraining):
        m.check_train_loss(10.0, 26)


def test_early_stopping_and_state_roundtrip():
    m = RunMonitor(TrainingConfig(early_stopping_patience=2))
    assert m.on_eval(2.0, 10) and m.should_early_stop() is None
    assert not m.on_eval(2.1, 20)
    assert not m.on_eval(2.2, 30)
    assert "early stopping" in m.should_early_stop()
    m2 = RunMonitor(TrainingConfig(early_stopping_patience=2))
    m2.load_state_dict(m.state_dict())
    assert m2.best_step == 10 and m2.evals_without_improvement == 2


def test_topk_keeps_best_and_prunes(tmp_path):
    keep = TopKAdapters(tmp_path, k=2)

    def save(p):
        p.mkdir(parents=True)
        (p / "w").write_text(p.name)

    for step, loss in [(10, 3.0), (20, 2.0), (30, 2.5), (40, 2.8), (50, 1.5)]:
        keep.consider(step, loss, save)
    assert [e["step"] for e in keep.entries] == [50, 20]
    assert sorted(p.name for p in (tmp_path / "adapters").glob("step-*")) == ["step-000020", "step-000050"]
    assert (tmp_path / "best" / "w").read_text() == "step-000050"
    assert TopKAdapters(tmp_path, k=2).best["step"] == 50  # reloads index after restart


def test_diagnose_flags_overfitting():
    rows = [{"step": s, "train/loss": 3 - s / 50} for s in range(1, 101)]
    rows += [{"step": s, "eval/loss": v} for s, v in [(0, 3.0), (25, 2.2), (50, 2.0), (75, 2.3), (100, 2.6)]]
    state = RunState(run_name="r", backend="hf", method="lora", model="m", baseline_eval_loss=3.0)
    levels = dict((msg.split(":")[0].split(" ")[0], lvl) for lvl, msg in diagnose(rows, state))
    assert levels["Training"] == OK and levels["Validation"] == OK
    assert levels["Overfitting"] == BAD
    assert math.isclose(state.baseline_eval_loss, 3.0)
