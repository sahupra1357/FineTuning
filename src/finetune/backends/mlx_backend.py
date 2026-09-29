"""Apple Silicon backend: MLX + mlx-lm LoRA layers (LoRA on fp16/bf16 weights, QLoRA on 4-bit weights).

A compact training loop (instead of ``mlx_lm.tuner.train``) so that the Mac
backend has the same guardrails, top-k/best adapters, resumable checkpoints and
metrics as the CUDA backend. Adapters are written in mlx-lm's format
(``adapters.safetensors`` + ``adapter_config.json``), so they work directly with
``mlx_lm.generate --adapter-path`` and ``mlx_lm.fuse``.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from ..data import IGNORE_INDEX, epoch_batches, render_prompt
from ..monitor import StopTraining
from ..pipeline import RunContext

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- model
def resolve_model_path(cfg) -> str:
    """MLX weights to load: explicit mlx repo, a locally quantized copy (QLoRA), or the HF repo."""
    if cfg.model.mlx_name_or_path:
        return cfg.model.mlx_name_or_path
    if cfg.method != "qlora":
        return cfg.model.name_or_path  # mlx_lm.load reads HF safetensors directly
    slug = cfg.model.name_or_path.rstrip("/").split("/")[-1]
    out = Path(cfg.mac.mlx_cache_dir) / f"{slug}-{cfg.quant.bits}bit-g{cfg.quant.group_size}"
    if not (out / "config.json").exists():
        from mlx_lm import convert

        print(f"Quantizing {cfg.model.name_or_path} to {cfg.quant.bits}-bit MLX weights -> {out} (one-time)")
        if out.exists():
            shutil.rmtree(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        convert(cfg.model.name_or_path, str(out), quantize=True, q_bits=cfg.quant.bits,
                q_group_size=cfg.quant.group_size, trust_remote_code=cfg.model.trust_remote_code)
    return str(out)


def _is_quantized(model) -> bool:
    import mlx.nn as nn

    return any(isinstance(m, nn.QuantizedLinear) or "Quantized" in type(m).__name__
               for _, m in model.named_modules())


def _lora_keys(model, target_modules) -> list[str] | None:
    """Module paths inside a transformer block to adapt; None = every linear layer (all-linear)."""
    if target_modules == "all-linear":
        return None
    wanted = set(target_modules)
    keys = set()
    for name, _ in model.layers[0].named_modules():
        if name and name.split(".")[-1] in wanted:
            keys.add(name)
    if not keys:
        raise ValueError(f"none of lora.target_modules={target_modules} found in the model's blocks")
    return sorted(keys)


def build_lora_model(cfg, model_path: str):
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

    model, mlx_tok = load(model_path)
    quantized = _is_quantized(model)
    if cfg.method == "qlora" and not quantized:
        log.warning("method=qlora but %s is not quantized; training LoRA on full-precision weights", model_path)
    if cfg.method == "lora" and quantized:
        log.warning("method=lora but %s is quantized; this is effectively QLoRA", model_path)

    model.freeze()
    n_blocks = len(model.layers)
    num_layers = n_blocks if cfg.lora.num_layers in (-1, None) else min(cfg.lora.num_layers, n_blocks)
    lora_parameters: dict[str, Any] = {
        "rank": cfg.lora.r,
        "scale": cfg.lora.alpha / cfg.lora.r,  # same effective scaling as PEFT's alpha / r
        "dropout": cfg.lora.dropout,
    }
    keys = _lora_keys(model, cfg.lora.target_modules)
    if keys is not None:
        lora_parameters["keys"] = keys
    linear_to_lora_layers(model, num_layers, lora_parameters, use_dora=cfg.lora.use_dora)
    print_trainable_parameters(model)

    adapter_config = {
        "model": model_path,
        "fine_tune_type": "dora" if cfg.lora.use_dora else "lora",
        "num_layers": num_layers,
        "lora_parameters": lora_parameters,
    }
    return model, mlx_tok, adapter_config


# --------------------------------------------------------------------------- helpers
def _set_wired_limit() -> None:
    import mlx.core as mx

    try:
        info = (getattr(mx, "device_info", None) or mx.metal.device_info)()
        limit = info["max_recommended_working_set_size"]
        setter = getattr(mx, "set_wired_limit", None) or mx.metal.set_wired_limit
        setter(limit)
    except Exception:
        pass  # not on Metal (e.g. Linux CPU build) or older MLX


def _peak_memory_gb() -> float | None:
    import mlx.core as mx

    for fn in (getattr(mx, "get_peak_memory", None), getattr(getattr(mx, "metal", None), "get_peak_memory", None)):
        if fn:
            try:
                return fn() / 1e9
            except Exception:
                continue
    return None


def _make_batch(rows: list[dict]):
    """Pad a list of tokenized rows -> (inputs, targets, mask) shifted for next-token prediction."""
    import mlx.core as mx
    import numpy as np

    n = max(len(r["input_ids"]) for r in rows)
    ids = np.zeros((len(rows), n), dtype=np.int32)
    labels = np.full((len(rows), n), IGNORE_INDEX, dtype=np.int32)
    for i, r in enumerate(rows):
        L = len(r["input_ids"])
        ids[i, :L] = r["input_ids"]
        labels[i, :L] = r["labels"]
    targets = labels[:, 1:]
    mask = targets != IGNORE_INDEX
    targets = np.where(mask, targets, 0)
    ntoks = int(sum(len(r["input_ids"]) for r in rows))
    return mx.array(ids[:, :-1]), mx.array(targets), mx.array(mask.astype(np.float32)), ntoks


def _build_schedule(cfg, total_steps: int):
    import mlx.optimizers as optim

    t = cfg.training
    warmup = int(t.warmup_ratio * total_steps)
    main_steps = max(1, total_steps - warmup)
    if t.lr_scheduler == "cosine":
        main = optim.cosine_decay(t.learning_rate, main_steps, 0.0)
    elif t.lr_scheduler == "linear":
        main = optim.linear_schedule(t.learning_rate, 0.0, main_steps)
    else:
        main = lambda _: t.learning_rate  # noqa: E731
    if warmup > 0:
        return optim.join_schedules([optim.linear_schedule(0.0, t.learning_rate, warmup), main], [warmup])
    return main


# --------------------------------------------------------------------------- training
def train(ctx: RunContext) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten

    cfg, t = ctx.cfg, ctx.cfg.training
    mx.random.seed(t.seed)
    _set_wired_limit()

    model_path = resolve_model_path(cfg)
    model, mlx_tok, adapter_config = build_lora_model(cfg, model_path)
    if t.gradient_checkpointing:
        from mlx_lm.tuner.trainer import grad_checkpoint

        grad_checkpoint(model.layers[0])

    total_steps = ctx.total_steps()
    optimizer = optim.AdamW(learning_rate=_build_schedule(cfg, total_steps), weight_decay=t.weight_decay)
    tok = ctx.tokenizer

    # ------------------------------------------------------------ closures
    def loss_fn(model, inputs, targets, mask):
        logits = model(inputs)
        ce = nn.losses.cross_entropy(logits, targets) * mask
        ntoks = mask.sum()
        return ce.astype(mx.float32).sum() / mx.maximum(ntoks, 1), ntoks

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def save_adapter(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(path / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
        (path / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))

    def restore_best(path: Path) -> None:
        model.load_weights(str(path / "adapters.safetensors"), strict=False)

    def generate(prompts: list[str]) -> list[str]:
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        model.eval()
        outs = []
        try:
            for p in prompts:
                ids = tok(render_prompt(tok, [{"role": "user", "content": p}]), add_special_tokens=False)["input_ids"]
                outs.append(mlx_generate(model, mlx_tok, prompt=ids, max_tokens=cfg.logging.sample_max_new_tokens,
                                         sampler=make_sampler(0.0), verbose=False).strip())
        finally:
            model.train()
        return outs

    def evaluate() -> float:
        model.eval()
        total, count = 0.0, 0.0
        for i in range(0, len(ctx.val), t.batch_size):
            inputs, targets, mask, _ = _make_batch(ctx.val[i : i + t.batch_size])
            loss, n = loss_fn(model, inputs, targets, mask)
            mx.eval(loss, n)
            total += loss.item() * n.item()
            count += n.item()
        model.train()
        return total / max(count, 1.0)

    n_train = len(ctx.train)
    batches_per_epoch = max(1, -(-n_train // t.batch_size))

    def micro_batch(idx: int) -> list[dict]:
        epoch, pos = divmod(idx, batches_per_epoch)
        order = epoch_batches(n_train, t.batch_size, t.seed, epoch)
        return [ctx.train[i] for i in order[pos]]

    # ------------------------------------------------------------ checkpoints
    def save_checkpoint(step: int) -> None:
        d = ctx.checkpoints_dir / f"step-{step:06d}"
        save_adapter(d)
        mx.save_safetensors(str(d / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
        (d / "trainer_state.json").write_text(json.dumps({"step": step}))
        ckpts = sorted(ctx.checkpoints_dir.glob("step-*"))
        for old in ckpts[: -max(1, t.keep_last_checkpoints)]:
            shutil.rmtree(old, ignore_errors=True)
        ctx.on_checkpoint()

    start_step = 0
    if ctx.resuming:
        ckpts = sorted(ctx.checkpoints_dir.glob("step-*"))
        if ckpts:
            last = ckpts[-1]
            model.load_weights(str(last / "adapters.safetensors"), strict=False)
            optimizer.init(model.trainable_parameters())
            optimizer.state = tree_unflatten(list(mx.load(str(last / "optimizer.safetensors")).items()))
            start_step = json.loads((last / "trainer_state.json").read_text())["step"]
            print(f"Resumed MLX weights + optimizer from {last}")

    # ------------------------------------------------------------ loop
    has_val = len(ctx.val) > 0
    status, reason = "completed", None
    step = start_step
    model.train()
    try:
        if has_val and t.eval_at_start and start_step == 0:
            ctx.on_eval(0, evaluate(), save_adapter, generate)

        win_loss, win_toks, win_trained, win_start = 0.0, 0, 0, time.time()
        win_gnorm, win_n = 0.0, 0
        while step < total_steps:
            step += 1
            acc = None
            for k in range(t.grad_accum_steps):
                inputs, targets, mask, ntoks = _make_batch(micro_batch((step - 1) * t.grad_accum_steps + k))
                (loss, n), grads = loss_and_grad(model, inputs, targets, mask)
                acc = grads if acc is None else tree_map(lambda a, b: a + b, acc, grads)
                mx.eval(acc, loss, n)
                win_loss += loss.item() * n.item()
                win_trained += int(n.item())
                win_toks += ntoks
            acc = tree_map(lambda g: g / t.grad_accum_steps, acc)
            if t.max_grad_norm and t.max_grad_norm > 0:
                acc, gnorm = optim.clip_grad_norm(acc, t.max_grad_norm)
            else:
                gnorm = mx.sqrt(sum((g * g).sum() for _, g in tree_flatten(acc)))
            optimizer.update(model, acc)
            mx.eval(model.parameters(), optimizer.state, gnorm)
            win_gnorm += gnorm.item()
            win_n += 1

            if step % cfg.logging.log_every_steps == 0 or step == total_steps:
                now = time.time()
                metrics = {
                    "train/loss": win_loss / max(win_trained, 1),
                    "train/learning_rate": optimizer.learning_rate.item(),
                    "train/grad_norm": win_gnorm / max(win_n, 1),
                    "train/epoch": step * t.grad_accum_steps / batches_per_epoch,
                    "perf/tokens_per_sec": win_toks / max(now - win_start, 1e-6),
                    "perf/step_time_sec": (now - win_start) / max(win_n, 1),
                }
                mem = _peak_memory_gb()
                if mem:
                    metrics["perf/memory_gb"] = mem
                print(f"step {step}/{total_steps}  loss {metrics['train/loss']:.4f}  "
                      f"lr {metrics['train/learning_rate']:.2e}  {metrics['perf/tokens_per_sec']:.0f} tok/s"
                      + (f"  mem {mem:.1f} GB" if mem else ""))
                ctx.log_train(step, metrics, tokens=win_toks)
                win_loss, win_toks, win_trained, win_start, win_gnorm, win_n = 0.0, 0, 0, now, 0.0, 0

            if has_val and step % t.eval_every_steps == 0:
                ctx.on_eval(step, evaluate(), save_adapter, generate)
                win_start = time.time()  # keep eval/sampling time out of throughput
            if step % t.save_every_steps == 0:
                save_checkpoint(step)

        if has_val and step % t.eval_every_steps != 0:
            ctx.on_eval(step, evaluate(), save_adapter, generate)
    except StopTraining as e:
        status, reason = e.status, e.reason
        save_checkpoint(step)
    except KeyboardInterrupt:
        save_checkpoint(step)
        raise

    save_adapter(ctx.run_dir / "final")
    ctx.finalize(status, reason, generate, restore_best)
