"""Backend-agnostic run orchestration.

``run_training`` prepares everything that does not depend on the framework
(run directory, data, pre-flight checks, logging, guardrails), then hands a
``RunContext`` to the selected backend. Backends only compute losses, save
adapters and generate text; every decision (log, keep best, stop early, write
report) goes through the context, so Mac and CUDA runs behave identically.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .data import load_tokenizer, prepare_datasets, render_mask_preview
from .memory import estimate_memory, memory_advice
from .monitor import RunMonitor, RunState, StopTraining, TopKAdapters
from .plots import make_plots
from .report import build_report, print_diagnostics
from .tracking import MetricsLogger

log = logging.getLogger(__name__)

SaveFn = Callable[[Path], None]
GenerateFn = Callable[[list[str]], list[str]]


# --------------------------------------------------------------------------- preflight
def preflight(cfg: Config, tok=None, train=None, val=None, stats=None, write_to: Path | None = None) -> dict:
    """Dataset stats + mask preview + memory estimate. Printed and optionally saved."""
    if tok is None:
        tok = load_tokenizer(cfg)
    if train is None:
        train, val, stats = prepare_datasets(cfg, tok)
    est = estimate_memory(cfg)
    lines = [
        f"Run: {cfg.run_name}   backend={cfg.backend}   method={cfg.method}",
        f"Model: {cfg.model.name_or_path}",
        "",
        "Dataset:",
        *[f"  {k}: {json.dumps(v)}" for k, v in stats.items()],
        "",
        "Memory estimate (approx.):",
        f"  {json.dumps(est)}",
    ]
    tips = memory_advice(est, cfg)
    if est.get("fits") == "no":
        lines.append("  ❌ Likely OUT OF MEMORY. Try: " + "; ".join(tips))
    elif est.get("fits") == "tight":
        lines.append("  ⚠️  Tight fit. If it OOMs: " + "; ".join(tips))
    tr = stats["train"]
    if tr["examples"] and tr["truncated"] / tr["examples"] > 0.05:
        lines.append(f"  ⚠️  {tr['truncated']}/{tr['examples']} examples truncated to max_seq_length="
                     f"{cfg.model.max_seq_length}")
    lines += ["", "Loss-mask preview of the first training example ([[...]] = trained tokens):",
              render_mask_preview(tok, train[0])]
    text = "\n".join(lines)
    print(text)
    if write_to:
        write_to.mkdir(parents=True, exist_ok=True)
        (write_to / "preflight.txt").write_text(text)
    return {"memory": est, "stats": stats}


# --------------------------------------------------------------------------- context
@dataclass
class RunContext:
    cfg: Config
    run_dir: Path
    tokenizer: Any
    train: list[dict]
    val: list[dict]
    logger: MetricsLogger
    monitor: RunMonitor
    keeper: TopKAdapters
    state: RunState
    resuming: bool = False
    persist: Callable[[], None] = lambda: None  # e.g. Modal volume.commit()
    dry_run_steps: int | None = None
    session_start: float = field(default_factory=time.time)
    _prior_active: float = 0.0
    _last_state_save: float = 0.0

    # ----------------------------------------------------------- derived values
    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    def total_steps(self) -> int:
        t = self.cfg.training
        if self.dry_run_steps:
            return self.dry_run_steps
        if t.max_steps and t.max_steps > 0:
            return t.max_steps
        steps_per_epoch = math.ceil(len(self.train) / (t.batch_size * t.grad_accum_steps))
        return max(1, math.ceil(steps_per_epoch * t.epochs))

    # ----------------------------------------------------------- persistence
    def save_state(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_state_save < 30:
            return
        self._last_state_save = now
        self.state.active_seconds = self._prior_active + (now - self.session_start)
        self.state.save(self.run_dir)
        (self.run_dir / "monitor_state.json").write_text(json.dumps(self.monitor.state_dict()))

    # ----------------------------------------------------------- callbacks from backends
    def log_train(self, step: int, metrics: dict[str, float], tokens: int = 0) -> None:
        """Log training metrics; raises StopTraining on NaN / divergence / time budget."""
        self.state.last_step = step
        self.state.trained_tokens += tokens
        self.logger.log(metrics, step)
        loss = metrics.get("train/loss")
        if loss is not None:
            self.monitor.check_train_loss(loss, step)
        reason = self.monitor.time_exceeded()
        if reason:
            raise StopTraining(reason, "stopped")
        self.save_state()

    def generate_samples(self, step: int, tag: str, generate: GenerateFn | None) -> None:
        prompts = self.cfg.logging.sample_prompts
        if not prompts or generate is None:
            return
        try:
            outputs = generate(prompts)
        except Exception as e:  # sampling must never kill training
            log.warning("sample generation failed: %s", e)
            return
        self.logger.log_samples([{"prompt": p, "output": o} for p, o in zip(prompts, outputs)], step, tag)

    def on_eval(self, step: int, eval_loss: float, save_adapter: SaveFn, generate: GenerateFn | None) -> None:
        """Record a validation result, keep top-k adapters, sample, and maybe stop early."""
        if step == 0:
            self.state.baseline_eval_loss = eval_loss
            self.logger.log({"eval/loss": eval_loss}, 0)
            print(f"[baseline] val loss {eval_loss:.4f}  ppl {math.exp(min(eval_loss, 50)):.2f}")
            self.generate_samples(0, "baseline", generate)
            self.save_state(force=True)
            return

        improved = self.monitor.on_eval(eval_loss, step)
        self.keeper.consider(step, eval_loss, save_adapter)
        self.state.best_step = self.monitor.best_step
        self.state.best_eval_loss = self.monitor.best_eval_loss
        self.logger.log({"eval/loss": eval_loss, "eval/best_loss": self.monitor.best_eval_loss}, step)
        mark = "★ new best" if improved else f"(no improvement x{self.monitor.evals_without_improvement})"
        print(f"[eval] step {step}  val loss {eval_loss:.4f}  ppl {math.exp(min(eval_loss, 50)):.2f}  {mark}")

        every = self.cfg.logging.sample_every_evals
        if every and self.monitor.n_evals % every == 0:
            self.generate_samples(step, "step", generate)
        if self.cfg.logging.live_plots:
            try:
                make_plots(self.run_dir)
            except Exception as e:
                log.debug("plotting failed: %s", e)
        self.save_state(force=True)
        self.persist()
        reason = self.monitor.should_early_stop()
        if reason:
            raise StopTraining(reason, "stopped")

    def on_checkpoint(self) -> None:
        self.save_state(force=True)
        self.persist()

    def finalize(self, status: str, reason: str | None, generate: GenerateFn | None = None,
                 restore_best: Callable[[Path], None] | None = None) -> None:
        best = self.run_dir / "best"
        if status in ("completed", "stopped") and restore_best and best.exists():
            try:
                restore_best(best)
                self.generate_samples(self.state.best_step or self.state.last_step, "best", generate)
            except Exception as e:
                log.warning("could not restore best adapter for final samples: %s", e)
        elif status in ("completed", "stopped"):
            self.generate_samples(self.state.last_step, "final", generate)

        self.state.status = status
        self.state.stop_reason = reason
        self.state.ended_at = time.time()
        self.state.final_eval_loss = self.state.best_eval_loss
        self.save_state(force=True)
        self.logger.log_summary({
            "status": status, "best_eval_loss": self.state.best_eval_loss,
            "best_step": self.state.best_step, "baseline_eval_loss": self.state.baseline_eval_loss,
        })
        report = build_report(self.run_dir)
        if best.exists():
            self.logger.log_artifact_dir(best, f"{self.cfg.run_name}-best")
        self.logger.close()
        self.persist()
        print(f"\n=== Run {self.cfg.run_name}: {status}" + (f" ({reason})" if reason else "") + " ===")
        print_diagnostics(self.run_dir)
        print(f"Report:      {report}")
        print(f"Best adapter: {best if best.exists() else '(none saved)'}")
        print(f"TensorBoard: tensorboard --logdir {self.run_dir / 'tensorboard'}")


# --------------------------------------------------------------------------- entry point
def run_training(cfg: Config, persist: Callable[[], None] | None = None,
                 dry_run_steps: int | None = None, force: bool = False) -> Path:
    """Train with the configured backend; returns the run directory."""
    # Our own messages at INFO; third-party libraries (httpx, matplotlib, ...) only at WARNING.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("finetune").setLevel(logging.INFO)
    run_dir = cfg.run_dir
    prior = RunState.load(run_dir)
    resuming = bool(prior and cfg.training.resume and prior.status != "completed")
    if prior and prior.status == "completed" and not dry_run_steps:
        raise SystemExit(f"run {cfg.run_name} already completed; choose a new run_name")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")

    tok = load_tokenizer(cfg)
    train, val, stats = prepare_datasets(cfg, tok)
    pf = preflight(cfg, tok, train, val, stats, write_to=run_dir)
    if pf["memory"].get("fits") == "no" and not force:
        raise SystemExit("Pre-flight memory estimate says this will not fit (use --force to try anyway).")

    state = RunState(
        run_name=str(cfg.run_name), backend=cfg.backend, method=cfg.method, model=cfg.model.name_or_path,
        hardware=pf["memory"].get("hardware"), gpu_count=cfg.modal.gpu_count if cfg.backend == "hf" else 1,
        cost_per_gpu_hour=cfg.modal.cost_per_gpu_hour if cfg.backend == "hf" else None,
        dataset_stats=stats, memory_estimate=pf["memory"],
    )
    monitor = RunMonitor(cfg.training)
    prior_active = 0.0
    if resuming and prior:
        print(f"Resuming run {cfg.run_name} from step {prior.last_step}")
        for k in ("started_at", "baseline_eval_loss", "best_step", "best_eval_loss", "trained_tokens"):
            setattr(state, k, getattr(prior, k))
        prior_active = prior.active_seconds
        ms = run_dir / "monitor_state.json"
        if ms.exists():
            monitor.load_state_dict(json.loads(ms.read_text()))

    ctx = RunContext(
        cfg=cfg, run_dir=run_dir, tokenizer=tok, train=train, val=val,
        logger=MetricsLogger(cfg, run_dir), monitor=monitor,
        keeper=TopKAdapters(run_dir, cfg.training.keep_top_k), state=state, resuming=resuming,
        persist=persist or (lambda: None), dry_run_steps=dry_run_steps, _prior_active=prior_active,
    )
    ctx.state.total_steps = ctx.total_steps()
    ctx.save_state(force=True)

    if cfg.backend == "mlx":
        from .backends import mlx_backend as backend
    else:
        from .backends import hf_backend as backend

    try:
        backend.train(ctx)
    except KeyboardInterrupt:
        # Backends save a resumable checkpoint before re-raising.
        ctx.finalize("stopped", "interrupted by user (re-run the same command to resume)")
        raise
    except Exception as e:
        ctx.finalize("failed", f"{type(e).__name__}: {e}")
        raise
    return run_dir
