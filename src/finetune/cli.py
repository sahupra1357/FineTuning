"""Command line interface.

  finetune prepare --config configs/7b_qlora.yaml --backend mac     # data + memory pre-flight only
  finetune train   --config configs/7b_qlora.yaml --backend mac     # train locally (MLX on Mac, HF on CUDA)
  finetune report  --run runs/<name>                                 # rebuild plots + report.html
  finetune eval    --run runs/<name> [--adapter best]                # base vs fine-tuned comparison
  finetune merge   --run runs/<name> [--dequantize] [--gguf --llama-cpp ~/llama.cpp]

Training on Modal uses modal_app.py (same configs):  modal run modal_app.py --config ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="YAML config (see configs/)")
    p.add_argument("--backend", choices=["mac", "mlx", "hf", "cuda"], help="override config backend")
    p.add_argument("--method", choices=["lora", "qlora"], help="override config method")
    p.add_argument("--run-name", help="run directory name (re-use it to resume)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config value, e.g. --set training.learning_rate=1e-4 (repeatable)")


def _load(args):
    from .config import load_config

    ov = list(args.overrides)
    if getattr(args, "wandb", None) is not None:
        ov.append(f"logging.wandb.enabled={'true' if args.wandb else 'false'}")
    if getattr(args, "no_tensorboard", False):
        ov.append("logging.tensorboard=false")
    return load_config(args.config, ov, backend=args.backend, method=args.method, run_name=args.run_name)


def cmd_prepare(args) -> None:
    from .pipeline import preflight

    cfg = _load(args)
    preflight(cfg)


def cmd_train(args) -> None:
    from .pipeline import run_training

    cfg = _load(args)
    run_training(cfg, dry_run_steps=args.dry_run_steps, force=args.force)


def cmd_report(args) -> None:
    from .report import build_report, print_diagnostics

    path = build_report(Path(args.run))
    print_diagnostics(Path(args.run))
    print(f"Report: {path}")


def cmd_eval(args) -> None:
    from .evaluate import compare

    compare(Path(args.run), adapter=args.adapter, n_val_prompts=args.n_val_prompts,
            max_new_tokens=args.max_new_tokens)


def cmd_merge(args) -> None:
    from .config import load_config
    from .merge import export_gguf, merge_hf_adapter, merge_mlx_adapter

    run = Path(args.run)
    cfg = load_config(run / "config.yaml")
    out = Path(args.out) if args.out else run / "merged"
    adapter = run / args.adapter
    if cfg.backend == "mlx":
        merged = merge_mlx_adapter(cfg, adapter, out, dequantize=args.dequantize or args.gguf)
    else:
        merged = merge_hf_adapter(cfg, adapter, out, push_to_hub=args.push_to_hub)
    if args.gguf:
        if not args.llama_cpp:
            sys.exit("--gguf needs --llama-cpp /path/to/llama.cpp")
        export_gguf(merged, Path(args.llama_cpp), args.gguf_type)


def main(argv: list[str] | None = None) -> None:
    import os

    # Hides "PyTorch was not found" on the Mac (MLX) path, where torch is intentionally absent.
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    parser = argparse.ArgumentParser(prog="finetune", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="dataset stats, loss-mask preview and memory estimate")
    _add_config_args(p)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="train locally (MLX on Mac, HF/PyTorch on CUDA)")
    _add_config_args(p)
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None,
                   help="force Weights & Biases on/off (default: from config)")
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--dry-run-steps", type=int, default=None, help="train only N steps as a sanity check")
    p.add_argument("--force", action="store_true", help="start even if the memory estimate says it won't fit")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("report", help="rebuild plots and report.html for a run (works mid-training)")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("eval", help="compare base vs fine-tuned (val loss + generations)")
    p.add_argument("--run", required=True)
    p.add_argument("--adapter", default="best", help="best | final | adapters/step-000100")
    p.add_argument("--n-val-prompts", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("merge", help="merge adapter into the base model (+ optional GGUF)")
    p.add_argument("--run", required=True)
    p.add_argument("--adapter", default="best")
    p.add_argument("--out", default=None)
    p.add_argument("--dequantize", action="store_true", help="MLX: write fp16 weights instead of 4-bit")
    p.add_argument("--push-to-hub", default=None, help="HF: repo id to upload the merged model (private)")
    p.add_argument("--gguf", action="store_true", help="also export GGUF via llama.cpp")
    p.add_argument("--llama-cpp", default=None, help="path to a llama.cpp checkout")
    p.add_argument("--gguf-type", default="q8_0", choices=["f16", "bf16", "q8_0"])
    p.set_defaults(func=cmd_merge)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
