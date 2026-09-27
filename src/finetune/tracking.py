"""Metrics fan-out: JSONL (always) + TensorBoard + Weights & Biases (both optional).

Both backends call the same ``MetricsLogger`` with the same metric names, so Mac
and Modal runs are directly comparable in TensorBoard / W&B:

  train/loss  train/learning_rate  train/grad_norm  train/epoch
  eval/loss   eval/perplexity      eval/best_loss
  perf/tokens_per_sec  perf/step_time_sec  perf/memory_gb
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from .config import Config

log = logging.getLogger(__name__)


def _wandb_has_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path.home() / ".netrc"
    return netrc.exists() and "api.wandb.ai" in netrc.read_text(errors="ignore")


def _make_tb_writer(logdir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter  # PyTorch's writer (CUDA backend)

        return SummaryWriter(log_dir=str(logdir))
    except Exception:
        pass
    try:
        from tensorboardX import SummaryWriter  # framework-free writer (MLX backend)

        return SummaryWriter(logdir=str(logdir))
    except Exception:
        log.warning("TensorBoard enabled but neither torch.utils.tensorboard nor tensorboardX "
                    "is installed (pip install 'finetune[logging]'); skipping TensorBoard.")
        return None


class MetricsLogger:
    def __init__(self, cfg: Config, run_dir: Path):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.samples_path = self.run_dir / "samples.jsonl"
        self.tb = _make_tb_writer(self.run_dir / "tensorboard") if cfg.logging.tensorboard else None
        self.wandb = self._init_wandb()

    # ----------------------------------------------------------------- wandb
    def _init_wandb(self):
        wcfg = self.cfg.logging.wandb
        if not wcfg.enabled or wcfg.mode == "disabled":
            return None
        try:
            import wandb
        except ImportError:
            log.warning("W&B enabled but `wandb` is not installed; skipping W&B.")
            return None
        mode = wcfg.mode
        if mode == "online" and not _wandb_has_credentials():
            log.warning("W&B: no API key found (run `wandb login` or set WANDB_API_KEY); "
                        "logging offline — sync later with `wandb sync %s/wandb`.", self.run_dir)
            mode = "offline"
        run_id = re.sub(r"[^A-Za-z0-9_-]", "-", str(self.cfg.run_name))[:64]
        try:
            return wandb.init(
                project=wcfg.project,
                entity=wcfg.entity,
                name=self.cfg.run_name,
                id=run_id,
                resume="allow",  # re-running the same run_name continues the same W&B run
                config=self.cfg.to_dict(),
                tags=wcfg.tags or None,
                dir=str(self.run_dir),
                mode=mode,
            )
        except Exception as e:  # never let observability kill a training run
            log.warning("W&B init failed (%s); continuing without W&B.", e)
            return None

    # ----------------------------------------------------------------- logging
    def log(self, metrics: dict[str, Any], step: int) -> None:
        clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None}
        if "eval/loss" in clean and "eval/perplexity" not in clean:
            clean["eval/perplexity"] = math.exp(min(clean["eval/loss"], 50.0))
        record = {"step": int(step), "time": time.time(), **clean}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.tb is not None:
            for k, v in clean.items():
                if math.isfinite(v):
                    self.tb.add_scalar(k, v, step)
            self.tb.flush()
        if self.wandb is not None:
            try:
                self.wandb.log(clean, step=int(step))
            except Exception as e:
                log.debug("wandb.log failed: %s", e)

    def log_samples(self, samples: list[dict[str, str]], step: int, tag: str) -> None:
        """samples: [{"prompt": ..., "output": ...}]; tag: "baseline" | "step" | "final"."""
        with open(self.samples_path, "a") as f:
            for s in samples:
                f.write(json.dumps({"step": int(step), "tag": tag, **s}, ensure_ascii=False) + "\n")
        if self.tb is not None:
            md = "\n\n".join(f"**Prompt:** {s['prompt']}\n\n**Output:** {s['output']}" for s in samples)
            self.tb.add_text(f"samples/{tag}", md, step)
            self.tb.flush()
        if self.wandb is not None:
            try:
                import wandb

                table = wandb.Table(columns=["step", "prompt", "output"],
                                    data=[[step, s["prompt"], s["output"]] for s in samples])
                self.wandb.log({f"samples/{tag}": table}, step=int(step))
            except Exception as e:
                log.debug("wandb sample table failed: %s", e)

    def log_summary(self, summary: dict[str, Any]) -> None:
        if self.wandb is not None:
            try:
                for k, v in summary.items():
                    if isinstance(v, (int, float, str, bool)):
                        self.wandb.summary[k] = v
            except Exception:
                pass

    def log_artifact_dir(self, path: Path, name: str, type_: str = "model") -> None:
        if self.wandb is None:
            return
        try:
            import wandb

            art = wandb.Artifact(re.sub(r"[^A-Za-z0-9_.-]", "-", name)[:120], type=type_)
            art.add_dir(str(path))
            self.wandb.log_artifact(art)
        except Exception as e:
            log.warning("W&B artifact upload failed: %s", e)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
        if self.wandb is not None:
            try:
                self.wandb.finish()
            except Exception:
                pass


def read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # partially written line after a crash
    return rows


def read_samples(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "samples.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
