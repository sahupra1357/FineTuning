"""Typed configuration shared by every backend.

One YAML file describes a whole experiment (model, method, data, training,
logging, infra). Both the Mac (MLX) and the CUDA (HF/PyTorch, used on Modal)
backends read the same object, so switching infra is a CLI flag, not a rewrite.
"""

from __future__ import annotations

import copy
import datetime as _dt
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ModelConfig(_Base):
    # Hugging Face repo id or local path. Used by the CUDA backend and for the tokenizer.
    name_or_path: str
    # Optional MLX-format repo/path for the Mac backend (e.g. an mlx-community 4-bit repo).
    # When null the Mac backend converts `name_or_path` itself (quantizing for QLoRA).
    mlx_name_or_path: str | None = None
    # Approximate parameter count (billions); used by the memory estimator when the
    # model config cannot be fetched.
    params_b: float | None = None
    max_seq_length: int = 2048
    trust_remote_code: bool = False
    # Override the tokenizer chat template (Jinja string) if the model has none.
    chat_template: str | None = None
    # CUDA backend attention implementation: "sdpa", "flash_attention_2", "eager".
    attn_implementation: str = "sdpa"
    revision: str | None = None


class LoRAConfig(_Base):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # "all-linear" or an explicit list, e.g. ["q_proj", "k_proj", "v_proj", "o_proj"].
    target_modules: str | list[str] = "all-linear"
    # PEFT >= 0.17 only: parameter names to adapt directly (e.g. MoE expert weights).
    target_parameters: list[str] | None = None
    use_dora: bool = False
    # MLX only: number of transformer blocks (counted from the top) that get adapters; -1 = all.
    num_layers: int = -1


class QuantConfig(_Base):
    bits: Literal[4, 8] = 4
    # CUDA (bitsandbytes)
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    # MLX
    group_size: int = 64


class DataConfig(_Base):
    # Local JSONL/JSON file or a Hugging Face dataset id (set `hf_dataset` instead).
    train_path: str | None = None
    val_path: str | None = None
    hf_dataset: str | None = None
    hf_config: str | None = None
    hf_train_split: str = "train"
    hf_val_split: str | None = None
    # Held-out fraction used when no explicit validation set is given.
    val_ratio: float = 0.05
    # "auto" detects chat (`messages`), instruction (`instruction`/`input`/`output`),
    # prompt/completion or plain `text` records.
    format: Literal["auto", "chat", "instruction", "completion", "text"] = "auto"
    system_prompt: str | None = None
    # Only assistant tokens contribute to the loss (recommended for instruction tuning).
    train_on_completions_only: bool = True
    max_train_samples: int | None = None
    max_val_samples: int | None = 500
    # Drop (instead of truncate) examples longer than max_seq_length.
    drop_long_examples: bool = False
    seed: int = 42

    @model_validator(mode="after")
    def _has_source(self) -> "DataConfig":
        if not self.train_path and not self.hf_dataset:
            raise ValueError("data: set either `train_path` or `hf_dataset`")
        return self


class TrainingConfig(_Base):
    epochs: float = 1.0
    # Hard cap on optimizer steps; overrides epochs when > 0.
    max_steps: int = -1
    batch_size: int = 1
    grad_accum_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    optimizer: Literal["adamw", "paged_adamw_8bit", "adamw_8bit"] = "adamw"
    seed: int = 42

    # Evaluation / checkpointing
    eval_every_steps: int = 100
    save_every_steps: int = 100
    keep_top_k: int = 2               # best adapters kept by val loss (plus "best/")
    keep_last_checkpoints: int = 1    # full (resumable) checkpoints kept
    eval_at_start: bool = True        # baseline val loss / samples before any training

    # Guardrails
    early_stopping_patience: int = 5  # evals without improvement; 0 disables
    early_stopping_min_delta: float = 0.0
    abort_on_nan: bool = True
    loss_spike_factor: float = 5.0    # abort if loss > factor * running mean (0 disables)
    max_hours: float | None = None    # wall-clock budget; stops cleanly and saves

    resume: bool = True               # auto-resume from the latest checkpoint of the run


class WandbConfig(_Base):
    enabled: bool = False
    project: str = "finetune"
    entity: str | None = None
    mode: Literal["online", "offline", "disabled"] = "online"
    tags: list[str] = Field(default_factory=list)


class LoggingConfig(_Base):
    tensorboard: bool = True
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    log_every_steps: int = 10
    # Fixed prompts whose generations are recorded at start, during training and at the end.
    sample_prompts: list[str] = Field(default_factory=list)
    # Generate samples every N evaluations (generation is slow); 0 = only start/end.
    sample_every_evals: int = 2
    sample_max_new_tokens: int = 200
    # Refresh PNG plots on every evaluation.
    live_plots: bool = True


class ModalConfig(_Base):
    gpu: str = "A100-80GB"            # e.g. "A10G", "L4", "L40S", "A100-40GB", "A100-80GB", "H100"
    gpu_count: int = 1
    timeout_hours: float = 12.0       # Modal max is 24h; training auto-resumes on re-run
    runs_volume: str = "finetune-runs"
    cache_volume: str = "finetune-hf-cache"
    data_volume: str = "finetune-data"
    secrets: list[str] = Field(default_factory=lambda: ["huggingface"])
    # Approx. USD per GPU-hour for the cost line in the report (check modal.com/pricing).
    cost_per_gpu_hour: float | None = None


class MacConfig(_Base):
    # Directory used to cache converted/quantized MLX models.
    mlx_cache_dir: str = "mlx_models"


class Config(_Base):
    run_name: str | None = None
    output_dir: str = "runs"
    method: Literal["lora", "qlora"] = "qlora"
    backend: Literal["mlx", "hf"] = "hf"

    model: ModelConfig
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    quant: QuantConfig = Field(default_factory=QuantConfig)
    data: DataConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    modal: ModalConfig = Field(default_factory=ModalConfig)
    mac: MacConfig = Field(default_factory=MacConfig)

    @field_validator("backend", mode="before")
    @classmethod
    def _alias_backend(cls, v: Any) -> Any:
        aliases = {"mac": "mlx", "cuda": "hf", "modal": "hf", "torch": "hf"}
        return aliases.get(v, v) if isinstance(v, str) else v

    @model_validator(mode="after")
    def _fill_run_name(self) -> "Config":
        if not self.run_name:
            stem = self.model.name_or_path.rstrip("/").split("/")[-1].lower()
            ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_name = f"{stem}-{self.method}-{ts}"
        return self

    # ------------------------------------------------------------------ helpers
    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / str(self.run_name)

    @property
    def effective_batch_size(self) -> int:
        return self.training.batch_size * self.training.grad_accum_steps

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _deep_update(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _parse_override(item: str) -> dict:
    """Turn ``training.learning_rate=1e-4`` into ``{"training": {"learning_rate": 1e-4}}``."""
    if "=" not in item:
        raise ValueError(f"override must look like key.sub=value, got {item!r}")
    key, raw = item.split("=", 1)
    value = yaml.safe_load(raw)
    node: dict = {}
    cur = node
    parts = key.strip().split(".")
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
    return node


def load_config(path: str | Path, overrides: list[str] | None = None, **top_level: Any) -> Config:
    """Load a YAML config, supporting an optional ``base:`` key for inheritance.

    ``overrides`` are dotted ``key=value`` strings (values parsed as YAML).
    ``top_level`` keyword arguments that are not None are applied last.
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    base = raw.pop("base", None)
    if base:
        base_path = (path.parent / base).resolve()
        with open(base_path) as f:
            raw = _deep_update(yaml.safe_load(f) or {}, raw)

    for item in overrides or []:
        raw = _deep_update(raw, _parse_override(item))
    raw = _deep_update(raw, {k: v for k, v in top_level.items() if v is not None})
    return Config.model_validate(raw)
