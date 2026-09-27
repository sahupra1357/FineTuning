"""Rough pre-flight memory estimate: will this config fit the target hardware?

The numbers are deliberately approximate (±20%). They exist to catch configs
that clearly will not fit *before* paying for a GPU or waiting on a download.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from typing import Any

from .config import Config

log = logging.getLogger(__name__)

GPU_MEMORY_GB = {
    "T4": 16, "L4": 24, "A10G": 24, "A10": 24, "L40S": 48, "A100": 40, "A100-40GB": 40,
    "A100-80GB": 80, "H100": 80, "H200": 141, "B200": 180,
}


def _model_dims(cfg: Config) -> dict[str, float]:
    """hidden size / layers / vocab / params; from the HF config when reachable, else heuristics."""
    dims: dict[str, float] = {}
    try:
        from transformers import AutoConfig

        hf = AutoConfig.from_pretrained(cfg.model.name_or_path, trust_remote_code=cfg.model.trust_remote_code)
        hf = getattr(hf, "text_config", hf)
        dims = {
            "hidden": hf.hidden_size,
            "layers": hf.num_hidden_layers,
            "vocab": hf.vocab_size,
            "intermediate": getattr(hf, "intermediate_size", 4 * hf.hidden_size),
        }
    except Exception as e:  # offline or gated: fall back to params_b heuristics
        log.debug("could not fetch model config (%s); using heuristics", e)

    params_b = cfg.model.params_b
    if not dims:
        pb = params_b or 7.0
        # Typical shapes: 7B ~ 4096x32, 14B ~ 5120x48, 24B ~ 5120x40 (wide MLP)
        hidden = 4096 if pb < 10 else 5120 if pb < 30 else 8192
        layers = max(12, round(pb * 1e9 / (12 * hidden * hidden)))
        dims = {"hidden": hidden, "layers": layers, "vocab": 128_000, "intermediate": 3.5 * hidden}
    if params_b is None:
        h, L = dims["hidden"], dims["layers"]
        params_b = (L * (4 * h * h + 3 * h * dims["intermediate"]) + 2 * dims["vocab"] * h) / 1e9
    dims["params_b"] = params_b
    return dims


def available_memory_gb(cfg: Config) -> tuple[float | None, str]:
    """Usable accelerator memory for the configured backend, and a label."""
    if cfg.backend == "mlx":
        total = None
        if platform.system() == "Darwin":
            try:
                total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 1e9
                try:
                    wired = int(subprocess.check_output(["sysctl", "-n", "iogpu.wired_limit_mb"]))
                    if wired > 0:
                        return wired / 1024, f"Mac GPU wired limit ({wired} MB)"
                except Exception:
                    pass
            except Exception:
                total = None
        total = total or 32.0
        # macOS lets the GPU wire ~75% of unified memory by default (raise with mac_wired_limit.sh).
        return total * 0.75, f"Mac unified memory {total:.0f} GB (≈75% usable by GPU)"
    gpu = cfg.modal.gpu.split(":")[0].upper()
    mem = GPU_MEMORY_GB.get(gpu) or GPU_MEMORY_GB.get(cfg.modal.gpu)
    try:
        import torch

        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            return mem, torch.cuda.get_device_name(0)
    except Exception:
        pass
    return (mem * cfg.modal.gpu_count if mem else None), f"{cfg.modal.gpu} x{cfg.modal.gpu_count}"


def estimate_memory(cfg: Config) -> dict[str, Any]:
    d = _model_dims(cfg)
    P = d["params_b"] * 1e9
    h, L, V, inter = d["hidden"], d["layers"], d["vocab"], d["intermediate"]
    b, s = cfg.training.batch_size, cfg.model.max_seq_length

    if cfg.method == "qlora":
        bytes_per_param = 0.5 * 1.1 if cfg.quant.bits == 4 else 1.06  # + quant constants
        weights = P * bytes_per_param + V * h * 2  # lm_head / embeddings usually kept in 16-bit
    else:
        weights = P * 2  # bf16

    r = cfg.lora.r
    tm = cfg.lora.target_modules
    per_layer = r * (8 * h + 3 * (h + inter)) if tm == "all-linear" else r * 2 * h * max(1, len(tm))
    n_layers = L if cfg.lora.num_layers in (-1, None) or cfg.backend != "mlx" else min(L, cfg.lora.num_layers)
    lora_params = per_layer * n_layers
    # adapter weights (fp32) + grads + Adam m/v
    lora_state = lora_params * 16

    per_layer_act = b * s * h * 34  # bytes/layer, bf16, flash/sdpa attention (Korthikanti et al.)
    if cfg.training.gradient_checkpointing:
        activations = b * s * h * 2 * L + per_layer_act
    else:
        activations = per_layer_act * L
    logits = b * s * V * 10  # bf16 logits + fp32 upcast for the loss + grad
    overhead = 1.5e9

    total = (weights + lora_state + activations + logits + overhead) * 1.1
    avail, label = available_memory_gb(cfg)
    gb = lambda x: round(x / 1e9, 2)  # noqa: E731
    est = {
        "params_b": round(d["params_b"], 2),
        "weights_gb": gb(weights),
        "lora_params_m": round(lora_params / 1e6, 1),
        "lora_state_gb": gb(lora_state),
        "activations_gb": gb(activations),
        "logits_gb": gb(logits),
        "total_gb": gb(total),
        "available_gb": round(avail, 1) if avail else None,
        "hardware": label,
    }
    if avail:
        ratio = gb(total) / avail
        est["fits"] = "yes" if ratio < 0.85 else "tight" if ratio < 1.0 else "no"
    return est


def memory_advice(est: dict[str, Any], cfg: Config) -> list[str]:
    tips = []
    if est.get("fits") in ("tight", "no"):
        if cfg.training.batch_size > 1:
            tips.append("lower training.batch_size (raise grad_accum_steps to keep the effective batch)")
        if cfg.model.max_seq_length > 1024:
            tips.append("lower model.max_seq_length")
        if not cfg.training.gradient_checkpointing:
            tips.append("enable training.gradient_checkpointing")
        if cfg.method == "lora":
            tips.append("switch method to qlora")
        if cfg.backend == "mlx":
            tips.append("raise the GPU wired limit: sudo ./scripts/mac_wired_limit.sh <MB>")
            tips.append("adapt fewer blocks via lora.num_layers")
        else:
            tips.append("pick a larger modal.gpu (e.g. A100-80GB / H100)")
    return tips
