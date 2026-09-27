# Build progress

Tracks where the pipeline build stands so work can resume from any session.

## Plan
| # | Step | Status |
|---|------|--------|
| 1 | Skeleton, typed config (`src/finetune/config.py`), presets (`configs/`) | done |
| 2 | Data pipeline (load, normalize, chat template, assistant-only masking, stats) | done |
| 3 | Monitoring: metrics logger (JSONL + TensorBoard + W&B), run monitor, plots, HTML report, memory estimator | done |
| 4 | CUDA backend (HF + PEFT, LoRA / bitsandbytes QLoRA) + shared orchestration (`pipeline.py`) | done — CPU smoke-tested incl. resume |
| 5 | Modal app (GPU from config, volumes, secrets, resume, TensorBoard endpoint) | done (merge fn needs step 7 `merge.py`) |
| 6 | Mac backend (MLX LoRA / QLoRA) | done — tested on MLX Linux CPU build (LoRA, QLoRA, resume, mlx_lm adapter compat) |
| 7 | Merge / eval / export + CLI | todo |
| 8 | Tests + README | todo |

## Decisions
- Mac trains with MLX (`mlx-lm`); CUDA trains with transformers + PEFT (+ bitsandbytes for QLoRA).
- Default models: Qwen2.5-7B-Instruct (7B), Mistral-Small-24B (dense ~20B QLoRA), gpt-oss-20b
  (LoRA on H100 via Modal; QLoRA on Mac via MLX 4-bit).
- Target Mac: M4 Max, 32 GB unified memory.
- TensorBoard and W&B both optional via `logging.tensorboard` / `logging.wandb.enabled`.

## Notes / known issues
- HF Hub is blocked from the build container; tests use an offline tiny Llama (`tests/tiny_model.py`).
- CUDA/QLoRA paths (bitsandbytes, gpt-oss MXFP4 dequant) not executed here — need a GPU run on Modal.
- MLX Metal-specific paths (wired limit, peak memory) only run on a real Mac.
