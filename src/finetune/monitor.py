"""Training guardrails and best-checkpoint bookkeeping (backend-agnostic).

``RunMonitor`` decides when a run should stop (NaN / loss spike / early stopping /
time budget). ``TopKAdapters`` keeps the k best adapters by validation loss and
mirrors the best one to ``<run_dir>/best``. ``RunState`` is the small JSON file the
report reads (status, stop reason, timings, best step).
"""

from __future__ import annotations

import json
import math
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import TrainingConfig


class StopTraining(Exception):
    """Raised inside a training loop to stop cleanly (state is saved first)."""

    def __init__(self, reason: str, status: str = "stopped"):
        super().__init__(reason)
        self.reason = reason
        self.status = status


class RunMonitor:
    def __init__(self, tcfg: TrainingConfig, start_time: float | None = None):
        self.tcfg = tcfg
        self.start_time = start_time or time.time()
        self.recent_losses: deque[float] = deque(maxlen=50)
        self.best_eval_loss = math.inf
        self.best_step = -1
        self.evals_without_improvement = 0
        self.n_evals = 0

    # ------------------------------------------------------------------ checks
    def check_train_loss(self, loss: float, step: int) -> None:
        if not math.isfinite(loss):
            if self.tcfg.abort_on_nan:
                raise StopTraining(f"training loss became {loss} at step {step} "
                                   "(try a lower learning_rate or check the data)", "aborted")
            return
        f = self.tcfg.loss_spike_factor
        if f and len(self.recent_losses) >= 20:
            mean = sum(self.recent_losses) / len(self.recent_losses)
            if loss > f * mean:
                raise StopTraining(f"loss spike at step {step}: {loss:.3f} > {f}x running mean "
                                   f"{mean:.3f} (diverging; lower learning_rate)", "aborted")
        self.recent_losses.append(loss)

    def on_eval(self, eval_loss: float, step: int) -> bool:
        """Record an evaluation. Returns True if it is a new best (see ``should_early_stop``)."""
        self.n_evals += 1
        if not math.isfinite(eval_loss):
            raise StopTraining(f"validation loss became {eval_loss} at step {step}", "aborted")
        improved = eval_loss < self.best_eval_loss - self.tcfg.early_stopping_min_delta
        if improved:
            self.best_eval_loss, self.best_step = eval_loss, step
            self.evals_without_improvement = 0
        else:
            self.evals_without_improvement += 1
        return improved

    def should_early_stop(self) -> str | None:
        p = self.tcfg.early_stopping_patience
        if p and self.evals_without_improvement >= p:
            return (f"early stopping: validation loss did not improve for {p} evaluations "
                    f"(best {self.best_eval_loss:.4f} at step {self.best_step})")
        return None

    def time_exceeded(self) -> str | None:
        if self.tcfg.max_hours and (time.time() - self.start_time) / 3600 >= self.tcfg.max_hours:
            return f"time budget of {self.tcfg.max_hours}h reached"
        return None

    # ------------------------------------------------------------------ state
    def state_dict(self) -> dict[str, Any]:
        return {
            "recent_losses": list(self.recent_losses),
            "best_eval_loss": self.best_eval_loss if math.isfinite(self.best_eval_loss) else None,
            "best_step": self.best_step,
            "evals_without_improvement": self.evals_without_improvement,
            "n_evals": self.n_evals,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.recent_losses.extend(d.get("recent_losses", []))
        b = d.get("best_eval_loss")
        self.best_eval_loss = math.inf if b is None else float(b)
        self.best_step = d.get("best_step", -1)
        self.evals_without_improvement = d.get("evals_without_improvement", 0)
        self.n_evals = d.get("n_evals", 0)


class TopKAdapters:
    """Keep the k best adapters (by eval loss) under ``adapters/`` and the best under ``best/``."""

    def __init__(self, run_dir: Path, k: int):
        self.run_dir = Path(run_dir)
        self.k = max(1, k)
        self.root = self.run_dir / "adapters"
        self.index_path = self.root / "index.json"
        self.entries: list[dict[str, Any]] = []
        if self.index_path.exists():
            self.entries = json.loads(self.index_path.read_text())

    def consider(self, step: int, eval_loss: float, save_fn: Callable[[Path], None]) -> bool:
        """Save the adapter if it ranks in the top-k. Returns True if it is the new best."""
        self.entries = [e for e in self.entries if e["step"] != step]
        worst = max((e["eval_loss"] for e in self.entries), default=math.inf)
        if len(self.entries) >= self.k and eval_loss >= worst:
            return False

        path = self.root / f"step-{step:06d}"
        if path.exists():
            shutil.rmtree(path)
        save_fn(path)
        self.entries.append({"step": step, "eval_loss": eval_loss, "path": str(path)})
        self.entries.sort(key=lambda e: e["eval_loss"])
        for e in self.entries[self.k :]:
            shutil.rmtree(e["path"], ignore_errors=True)
        self.entries = self.entries[: self.k]
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(self.entries, indent=2))

        is_best = self.entries[0]["step"] == step
        if is_best:
            best = self.run_dir / "best"
            if best.exists():
                shutil.rmtree(best)
            shutil.copytree(path, best)
            (best / "best.json").write_text(json.dumps({"step": step, "eval_loss": eval_loss}, indent=2))
        return is_best

    @property
    def best(self) -> dict[str, Any] | None:
        return self.entries[0] if self.entries else None


@dataclass
class RunState:
    run_name: str
    backend: str
    method: str
    model: str
    status: str = "running"            # running | completed | stopped | aborted | failed
    stop_reason: str | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    total_steps: int = 0
    last_step: int = 0
    best_step: int | None = None
    best_eval_loss: float | None = None
    baseline_eval_loss: float | None = None
    final_eval_loss: float | None = None
    trained_tokens: int = 0
    active_seconds: float = 0.0        # compute time summed across resumed sessions (for cost)
    hardware: str | None = None
    gpu_count: int = 1
    cost_per_gpu_hour: float | None = None
    dataset_stats: dict[str, Any] = field(default_factory=dict)
    memory_estimate: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def active_hours(self) -> float:
        return self.active_seconds / 3600

    @property
    def estimated_cost(self) -> float | None:
        if self.cost_per_gpu_hour is None:
            return None
        return round(self.active_hours * self.cost_per_gpu_hour * self.gpu_count, 2)

    def save(self, run_dir: Path) -> None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "run_state.json").write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def load(cls, run_dir: Path) -> "RunState | None":
        p = Path(run_dir) / "run_state.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)
