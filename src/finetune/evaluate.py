"""Before/after comparison: base model vs. fine-tuned adapter.

Reports validation loss / perplexity for both, and side-by-side generations for
the configured sample prompts plus a few held-out validation prompts (with the
reference answer), written to ``<run>/eval_<adapter>.md`` and ``.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .config import Config, load_config
from .data import load_records, load_tokenizer, normalize_record, prepare_datasets, render_prompt


def _eval_prompts(cfg: Config, n_val: int) -> list[dict[str, str]]:
    items = [{"prompt": p, "reference": ""} for p in cfg.logging.sample_prompts]
    _, val_raw = load_records(cfg.data)
    for raw in val_raw[:n_val]:
        ex = normalize_record(raw, cfg.data.format, cfg.data.system_prompt)
        msgs = ex.get("messages")
        if not msgs:
            continue
        # first assistant turn; prompt = the user message right before it
        k = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        user = next((m["content"] for m in reversed(msgs[:k]) if m["role"] == "user"), None)
        if user:
            items.append({"prompt": user, "reference": msgs[k]["content"]})
    return items


# --------------------------------------------------------------------------- HF
def _hf_run(cfg: Config, adapter_dir: Path, val: list[dict], prompts: list[str], max_new: int):
    import torch
    from peft import PeftModel

    from .backends.hf_backend import Collator, load_model, make_generate_fn

    tok = load_tokenizer(cfg)
    model, _ = load_model(cfg, for_training=False)
    model.eval()
    collate = Collator(tok.pad_token_id)

    def val_loss(m) -> float | None:
        if not val:
            return None
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val), cfg.training.batch_size):
                b = collate(val[i : i + cfg.training.batch_size])
                b = {k: v.to(m.device) for k, v in b.items()}
                n = int((b["labels"][:, 1:] != -100).sum())
                tot += float(m(**b).loss) * n
                cnt += n
        return tot / max(cnt, 1)

    base = {"val_loss": val_loss(model), "outputs": make_generate_fn(model, tok, max_new)(prompts)}
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    tuned = {"val_loss": val_loss(model), "outputs": make_generate_fn(model, tok, max_new)(prompts)}
    return base, tuned


# --------------------------------------------------------------------------- MLX
def _mlx_run(cfg: Config, adapter_dir: Path, val: list[dict], prompts: list[str], max_new: int):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import load_adapters

    from .backends.mlx_backend import _make_batch, resolve_model_path

    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    model, mlx_tok = load(adapter_cfg.get("model") or resolve_model_path(cfg))
    tok = load_tokenizer(cfg)

    def val_loss(m) -> float | None:
        if not val:
            return None
        m.eval()
        tot, cnt = 0.0, 0.0
        for i in range(0, len(val), cfg.training.batch_size):
            inputs, targets, mask, _ = _make_batch(val[i : i + cfg.training.batch_size])
            ce = (nn.losses.cross_entropy(m(inputs), targets) * mask).astype(mx.float32).sum()
            n = mask.sum()
            mx.eval(ce, n)
            tot += ce.item()
            cnt += n.item()
        return tot / max(cnt, 1.0)

    def gen(m) -> list[str]:
        m.eval()
        outs = []
        for p in prompts:
            ids = tok(render_prompt(tok, [{"role": "user", "content": p}]), add_special_tokens=False)["input_ids"]
            outs.append(generate(m, mlx_tok, prompt=ids, max_tokens=max_new, sampler=make_sampler(0.0)).strip())
        return outs

    base = {"val_loss": val_loss(model), "outputs": gen(model)}
    load_adapters(model, str(adapter_dir))
    tuned = {"val_loss": val_loss(model), "outputs": gen(model)}
    return base, tuned


# --------------------------------------------------------------------------- entry
def compare(run_dir: Path, adapter: str = "best", n_val_prompts: int = 5,
            max_new_tokens: int | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    adapter_dir = run_dir / adapter
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter not found: {adapter_dir}")
    tok = load_tokenizer(cfg)
    _, val, _ = prepare_datasets(cfg, tok)
    items = _eval_prompts(cfg, n_val_prompts)
    prompts = [it["prompt"] for it in items]
    max_new = max_new_tokens or cfg.logging.sample_max_new_tokens

    runner = _mlx_run if cfg.backend == "mlx" else _hf_run
    base, tuned = runner(cfg, adapter_dir, val, prompts, max_new)

    def ppl(x):
        return None if x is None else round(math.exp(min(x, 50)), 3)

    def rnd(x):
        return None if x is None else round(x, 4)

    result = {
        "adapter": str(adapter_dir),
        "base": {"val_loss": rnd(base["val_loss"]), "val_ppl": ppl(base["val_loss"])},
        "tuned": {"val_loss": rnd(tuned["val_loss"]), "val_ppl": ppl(tuned["val_loss"])},
        "samples": [
            {**it, "base": b, "tuned": t} for it, b, t in zip(items, base["outputs"], tuned["outputs"])
        ],
    }
    (run_dir / f"eval_{adapter}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    lines = [f"# Evaluation: base vs `{adapter}`", "",
             "| | val loss | val perplexity |", "|---|---|---|",
             f"| base | {result['base']['val_loss']} | {result['base']['val_ppl']} |",
             f"| fine-tuned | {result['tuned']['val_loss']} | {result['tuned']['val_ppl']} |", ""]
    for s in result["samples"]:
        lines += [f"## Prompt\n{s['prompt']}", ""]
        if s["reference"]:
            lines += [f"**Reference:** {s['reference']}", ""]
        lines += [f"**Base:** {s['base']}", "", f"**Fine-tuned:** {s['tuned']}", "", "---", ""]
    md = run_dir / f"eval_{adapter}.md"
    md.write_text("\n".join(lines))
    print("\n".join(lines[:6]))
    print(f"Full comparison: {md}")
    return result
