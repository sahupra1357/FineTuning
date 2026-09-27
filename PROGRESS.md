# Build progress

Tracks where the pipeline build stands so work can resume from any session.

## Plan
| # | Step | Status |
|---|------|--------|
| 1 | Skeleton, typed config (`src/finetune/config.py`), presets (`configs/`) | done |
| 2 | Data pipeline (load, normalize, chat template, assistant-only masking, stats) | todo |
| 3 | Monitoring: metrics logger (JSONL + TensorBoard + W&B), run monitor, plots, HTML report | todo |
| 4 | CUDA backend (HF + PEFT, LoRA / bitsandbytes QLoRA) | todo |
| 5 | Modal app (GPU from config, volumes, secrets, resume, TensorBoard endpoint) | todo |
| 6 | Mac backend (MLX LoRA / QLoRA) | todo |
| 7 | Merge / eval / export + CLI | todo |
| 8 | Tests + README | todo |

## Decisions
- Mac trains with MLX (`mlx-lm`); CUDA trains with transformers + PEFT (+ bitsandbytes for QLoRA).
- Default models: Qwen2.5-7B-Instruct (7B), Mistral-Small-24B (dense ~20B QLoRA), gpt-oss-20b
  (LoRA on H100 via Modal; QLoRA on Mac via MLX 4-bit).
- Target Mac: M4 Max, 32 GB unified memory.
- TensorBoard and W&B both optional via `logging.tensorboard` / `logging.wandb.enabled`.

## Notes / known issues
- None yet.
