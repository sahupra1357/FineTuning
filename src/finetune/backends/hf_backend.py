"""CUDA backend: transformers + PEFT (LoRA) + bitsandbytes (QLoRA).

Runs on Modal (see ``modal_app.py``) or any machine with an NVIDIA GPU; it also
runs on CPU for tiny smoke-test models. Uses ``transformers.Trainer`` for the
loop (mixed precision, paged optimizers, gradient checkpointing, resumable
checkpoints) and routes every metric/decision through the shared RunContext.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

from ..monitor import StopTraining
from ..pipeline import RunContext

log = logging.getLogger(__name__)

OPTIMIZERS = {"adamw": "adamw_torch", "paged_adamw_8bit": "paged_adamw_8bit", "adamw_8bit": "adamw_bnb_8bit"}


def _is_gpt_oss(cfg, model_type: str | None = None) -> bool:
    return "gpt-oss" in cfg.model.name_or_path.lower() or model_type == "gpt_oss"


# --------------------------------------------------------------------------- model
def load_model(cfg, for_training: bool = True):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cuda = torch.cuda.is_available()
    if cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    hf_cfg = AutoConfig.from_pretrained(cfg.model.name_or_path, trust_remote_code=cfg.model.trust_remote_code)
    import transformers

    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    kwargs: dict[str, Any] = dict(
        **{dtype_key: dtype},
        trust_remote_code=cfg.model.trust_remote_code,
        revision=cfg.model.revision,
        attn_implementation=cfg.model.attn_implementation,
    )
    method = cfg.method
    if _is_gpt_oss(cfg, getattr(hf_cfg, "model_type", None)):
        from transformers import Mxfp4Config

        # MXFP4 weights -> bf16 for training (bitsandbytes cannot quantize the fused expert tensors).
        kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        if method == "qlora":
            log.warning("gpt-oss: bitsandbytes QLoRA is not supported for its MoE experts; "
                        "training LoRA on bf16 weights instead (use an 80 GB GPU).")
            method = "lora"
    elif method == "qlora":
        if not cuda:
            raise RuntimeError("QLoRA (bitsandbytes) needs an NVIDIA GPU; use method: lora on CPU "
                               "or the MLX backend on a Mac.")
        from transformers import BitsAndBytesConfig

        q = cfg.quant
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=q.bits == 4,
            load_in_8bit=q.bits == 8,
            bnb_4bit_quant_type=q.quant_type,
            bnb_4bit_use_double_quant=q.double_quant,
            bnb_4bit_compute_dtype=getattr(torch, q.compute_dtype),
        )

    if cuda:
        kwargs["device_map"] = "auto" if cfg.modal.gpu_count > 1 else {"": 0}
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **kwargs)
    if not for_training:
        return model, method

    gc = cfg.training.gradient_checkpointing
    if method == "qlora":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gc, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    elif gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    return model, method


def apply_lora(model, cfg):
    from peft import LoraConfig, get_peft_model

    lc = cfg.lora
    kwargs: dict[str, Any] = dict(
        r=lc.r, lora_alpha=lc.alpha, lora_dropout=lc.dropout, target_modules=lc.target_modules,
        bias="none", task_type="CAUSAL_LM", use_dora=lc.use_dora,
    )
    if lc.target_parameters:
        kwargs["target_parameters"] = lc.target_parameters  # peft >= 0.17
    model = get_peft_model(model, LoraConfig(**kwargs))
    model.print_trainable_parameters()
    return model


# --------------------------------------------------------------------------- data
class _ListDataset:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


class Collator:
    """Right-pads a batch; labels padded with -100. Counts trained tokens for throughput."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id
        self.tokens_seen = 0

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            labels[i, :L] = torch.tensor(b["labels"])
            attn[i, :L] = 1
        self.tokens_seen += int(attn.sum())
        return {"input_ids": ids, "labels": labels, "attention_mask": attn}


# --------------------------------------------------------------------------- generation
def make_generate_fn(model, tok, max_new_tokens: int):
    import torch

    from ..data import render_prompt

    def generate(prompts: list[str]) -> list[str]:
        was_training = model.training
        model.eval()
        old_side = tok.padding_side
        tok.padding_side = "left"
        outs = []
        try:
            with torch.no_grad():
                for p in prompts:  # one at a time: simple and memory-safe
                    text = render_prompt(tok, [{"role": "user", "content": p}])
                    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
                    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=tok.pad_token_id, use_cache=True)
                    outs.append(tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        finally:
            tok.padding_side = old_side
            if was_training:
                model.train()
        return outs

    return generate


# --------------------------------------------------------------------------- training
def _latest_checkpoint(d: Path) -> str | None:
    cks = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])) if d.exists() else []
    return str(cks[-1]) if cks else None


def _training_args(**kw):
    """Build TrainingArguments across transformers 4.x / 5.x (renamed / removed arguments)."""
    import inspect

    from transformers import TrainingArguments

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "evaluation_strategy" in params:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    dropped = [k for k in kw if k not in params]
    if dropped:
        log.debug("TrainingArguments: ignoring unsupported %s", dropped)
    return TrainingArguments(**{k: v for k, v in kw.items() if k in params})


def train(ctx: RunContext) -> None:
    import torch
    from transformers import Trainer, TrainerCallback

    cfg, t = ctx.cfg, ctx.cfg.training
    tok = ctx.tokenizer
    torch.manual_seed(t.seed)

    model, _ = load_model(cfg)
    model = apply_lora(model, cfg)
    generate = make_generate_fn(model, tok, cfg.logging.sample_max_new_tokens)
    collator = Collator(tok.pad_token_id)
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()

    def save_adapter(path: Path) -> None:
        model.save_pretrained(str(path))
        tok.save_pretrained(str(path))

    def restore_best(path: Path) -> None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(str(path / "adapter_model.safetensors")))

    class Bridge(TrainerCallback):
        """Forwards Trainer events to the RunContext and converts StopTraining into a clean stop."""

        def __init__(self):
            self.stop: StopTraining | None = None
            self.t_last = time.time()
            self.tok_last = 0

        def _halt(self, control, e: StopTraining):
            self.stop = e
            control.should_training_stop = True
            control.should_save = True  # keep a resumable checkpoint

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs or "loss" not in logs:
                return
            now = time.time()
            toks = collator.tokens_seen - self.tok_last
            m = {
                "train/loss": logs["loss"],
                "train/learning_rate": logs.get("learning_rate"),
                "train/grad_norm": logs.get("grad_norm"),
                "train/epoch": logs.get("epoch"),
                "perf/tokens_per_sec": toks / max(now - self.t_last, 1e-6),
            }
            if cuda:
                m["perf/memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
            self.t_last, self.tok_last = now, collator.tokens_seen
            try:
                ctx.log_train(state.global_step, m, tokens=toks)
            except StopTraining as e:
                self._halt(control, e)

        def on_evaluate(self, args, state, control, metrics=None, **kw):
            if not metrics or "eval_loss" not in metrics:
                return
            try:
                ctx.on_eval(state.global_step, metrics["eval_loss"], save_adapter, generate)
            except StopTraining as e:
                self._halt(control, e)

        def on_save(self, args, state, control, **kw):
            ctx.on_checkpoint()

    has_val = len(ctx.val) > 0
    args = _training_args(
        output_dir=str(ctx.checkpoints_dir),
        per_device_train_batch_size=t.batch_size,
        per_device_eval_batch_size=t.batch_size,
        gradient_accumulation_steps=t.grad_accum_steps,
        num_train_epochs=t.epochs,
        max_steps=ctx.total_steps() if (ctx.dry_run_steps or t.max_steps > 0) else -1,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler,
        warmup_steps=int(t.warmup_ratio * ctx.total_steps()),
        weight_decay=t.weight_decay,
        max_grad_norm=t.max_grad_norm,
        optim=OPTIMIZERS[t.optimizer] if cuda else "adamw_torch",
        bf16=bf16,
        fp16=cuda and not bf16,
        logging_steps=ctx.log_every_steps,
        eval_strategy="steps" if has_val else "no",
        eval_steps=ctx.eval_every_steps,
        save_strategy="steps",
        save_steps=t.save_every_steps,
        save_total_limit=max(1, t.keep_last_checkpoints),
        seed=t.seed,
        report_to="none",  # our MetricsLogger handles TensorBoard / W&B with shared metric names
        remove_unused_columns=False,
        dataloader_pin_memory=cuda,
        gradient_checkpointing=False,  # already configured on the model
        disable_tqdm=False,
    )
    bridge = Bridge()
    trainer = Trainer(
        model=model, args=args, data_collator=collator,
        train_dataset=_ListDataset(ctx.train), eval_dataset=_ListDataset(ctx.val) if has_val else None,
        callbacks=[bridge],
    )

    resume_from = _latest_checkpoint(ctx.checkpoints_dir) if ctx.resuming else None
    status, reason = "completed", None
    try:
        if has_val and t.eval_at_start and not resume_from:
            trainer.evaluate()  # step-0 baseline (loss + samples)
        trainer.train(resume_from_checkpoint=resume_from)
        if bridge.stop:
            status, reason = bridge.stop.status, bridge.stop.reason
        elif has_val and trainer.state.global_step % ctx.eval_every_steps != 0:
            trainer.evaluate()  # make sure the last steps are evaluated
    except KeyboardInterrupt:
        save_adapter(ctx.run_dir / "interrupted")  # last weights; resume uses checkpoints/
        raise
    save_adapter(ctx.run_dir / "final")  # weights at the last step (best/ holds the best by val loss)
    ctx.finalize(status, reason, generate, restore_best)
    if math.isfinite(ctx.monitor.best_eval_loss):
        log.info("best val loss %.4f at step %d", ctx.monitor.best_eval_loss, ctx.monitor.best_step)
