# LoRA / QLoRA fine-tuning for 7B–20B LLMs on a Mac and on Modal GPUs

One pipeline and one YAML config per experiment, with two ways to run it:

| | Local Mac (Apple Silicon) | Modal (NVIDIA GPUs) |
|---|---|---|
| Engine | **MLX** (`mlx-lm`) | **PyTorch**: transformers + PEFT (+ bitsandbytes) |
| LoRA | full-precision (bf16/fp16) base model | bf16 base model |
| QLoRA | 4-bit MLX-quantized base model | 4-bit NF4 base model (bitsandbytes) |
| Command | `finetune train --config ... --backend mac` | `modal run modal_app.py --config ...` |

Both backends use the same data processing, the same loss masking (only assistant tokens are
trained), the same metric names, the same guardrails, and produce the same report. That means
runs from the Mac and from Modal can be compared directly.

---

## 1. Setup

**Mac (M-series):**
```bash
./scripts/setup_mac.sh && source .venv/bin/activate
huggingface-cli login          # gated models (Llama, Mistral, ...)
wandb login                    # optional
```

**Modal (cloud GPUs):**
```bash
pip install -e ".[modal]" && modal setup
modal secret create huggingface HF_TOKEN=hf_xxx
modal secret create wandb WANDB_API_KEY=xxx        # only if you enable W&B
```

**A local NVIDIA box:** `./scripts/setup_cuda.sh`, then use `--backend hf`.

## 2. Quick sanity check (minutes)

```bash
finetune train --config configs/smoke_test.yaml --backend mac     # Mac
modal run modal_app.py --config configs/smoke_test.yaml           # Modal (T4)
```

## 3. Presets

| Config | Model | Method | Where | Est. memory |
|---|---|---|---|---|
| `7b_qlora.yaml` | Qwen2.5-7B-Instruct | QLoRA | Mac 32 GB ✅ / Modal A10G | ~16 GB |
| `7b_lora.yaml` | Qwen2.5-7B-Instruct | LoRA | Mac 32 GB (short sequences) / Modal L40S | ~27 GB |
| `20b_qlora.yaml` | Mistral-Small-24B (dense) | QLoRA | Modal L40S; Mac only with `max_seq_length≤1024` + wired limit | ~25 GB |
| `gpt_oss_20b_lora.yaml` | gpt-oss-20b (MoE) | LoRA (bf16) | Modal H100 | ~58 GB |
| `gpt_oss_20b_qlora_mac.yaml` | gpt-oss-20b | QLoRA (MLX 4-bit) | Mac 32 GB + wired limit | ~18 GB |

**M4 Max with 32 GB:**
- 7B QLoRA is easy and 7B LoRA works.
- For ~20B QLoRA, first run `sudo ./scripts/mac_wired_limit.sh 26624` (by default macOS only
  lets the GPU use about 75% of RAM). Use `batch_size: 1` and `lora.num_layers` to adapt only
  the top blocks.
- 20B LoRA on bf16 weights (about 42 GB of weights alone) does not fit, so run it on Modal.

**gpt-oss-20b on CUDA:** its released weights are MXFP4. The CUDA backend dequantizes them to
bf16 and trains LoRA; bitsandbytes cannot quantize its fused MoE expert tensors, so "qlora" falls
back to LoRA with a warning. On a Mac, MLX quantizes it to 4 bits, which gives real QLoRA.

Any Hugging Face causal LM works: copy a preset and change `model.name_or_path`. For the Mac,
optionally set `model.mlx_name_or_path` to a pre-quantized `mlx-community/...-4bit` repo so it
doesn't have to be quantized locally.

## 4. Your data

Put JSONL at `data.train_path` (and optionally `data.val_path`; otherwise `data.val_ratio` of
the training data is held out). Each line can be any of these formats, detected automatically:

```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"instruction": "...", "input": "optional", "output": "..."}
{"prompt": "...", "completion": "..."}
{"text": "raw text, fully trained"}
```

You can also use a Hugging Face dataset: set `data.hf_dataset` (and optionally
`hf_train_split` / `hf_val_split`).

Check the data before you train:
```bash
finetune prepare --config configs/7b_qlora.yaml --backend mac
```
This prints token-length stats, the number of truncated examples, a memory estimate for your
hardware, and a **loss-mask preview** in which trained text is shown in `[[...]]`. Check that
only the answers are bracketed.

## 5. Training

```bash
# Mac
finetune train --config configs/7b_qlora.yaml --backend mac --run-name qwen7b-v1
finetune train --config configs/7b_qlora.yaml --backend mac --dry-run-steps 20     # quick check that loss falls
finetune train --config configs/7b_qlora.yaml --set training.learning_rate=1e-4 --set lora.r=32 --wandb

# Modal
modal run modal_app.py --config configs/7b_qlora.yaml --run-name qwen7b-v1
modal run --detach modal_app.py --config configs/gpt_oss_20b_lora.yaml --wandb    # keeps running after you close the terminal
```

**Resume:** re-run the same command with the same `--run-name`. Training continues from the
latest checkpoint, including optimizer state and data position. On Modal a pre-empted or
timed-out run also retries once automatically.

## 6. Checking that training is on track

Everything for a run lives in `runs/<run-name>/` (on Modal, in the `finetune-runs` volume):

```
runs/<run-name>/
├── preflight.txt       dataset stats, memory estimate, loss-mask preview
├── metrics.jsonl       every metric (always written)
├── plots/*.png         loss (train vs val, best marked), perplexity, lr, grad norm, throughput, memory
├── samples.jsonl       generations for fixed prompts: baseline → during training → best
├── report.html         everything above + automatic health checks + config (open in a browser)
├── tensorboard/        if logging.tensorboard: true
├── best/               ★ adapter with the lowest validation loss (used by eval/merge)
├── final/              adapter at the last step
├── adapters/           top-k adapters by validation loss
└── checkpoints/        resumable checkpoints (adapter + optimizer)
```

- **Live in the terminal:** step, loss, learning rate, tokens/s and memory, plus a line for each
  evaluation (`★ new best` or `no improvement xN`).
- **Plots and report during a run:** `finetune report --run runs/<name>`.
- **TensorBoard:**
  - Mac: `tensorboard --logdir runs/`.
  - Modal: `modal serve modal_app.py` gives a live URL.
- **Weights & Biases:** set `logging.wandb.enabled: true` or pass `--wandb`. Runs appear under
  the project in `logging.wandb.project`, and re-running the same `run_name` continues the same
  W&B run. If no API key is found, it logs offline instead of failing.
- **Health checks** (in the report and printed at the end of training):
  - is the loss falling or flat
  - how much validation loss improved vs. the baseline
  - **overfitting** (validation loss rising after its best point)
  - plateaus, gradient spikes, truncated data
- **Guardrails** (config section `training:`):

  | Setting | What it does |
  |---|---|
  | `abort_on_nan` | stops the run if the loss becomes NaN |
  | `loss_spike_factor` | stops the run if the loss diverges (jumps far above its recent average) |
  | `early_stopping_patience` | stops after this many evaluations with no improvement in validation loss |
  | `max_hours` | time budget |

  Each of these saves a checkpoint before stopping.

Get Modal results onto your machine:
```bash
modal volume get finetune-runs <run-name> ./runs/
```

## 7. Evaluate, merge, export

```bash
finetune eval  --run runs/<name>                       # base vs fine-tuned: val loss/perplexity + side-by-side answers
finetune merge --run runs/<name>                       # best adapter → runs/<name>/merged
finetune merge --run runs/<name> --gguf --llama-cpp ~/src/llama.cpp   # + GGUF for llama.cpp / Ollama
modal run modal_app.py::merge --run-name <name>        # merge big models on Modal (128 GB RAM)
```

Other ways to use the result:
- **MLX adapters** work directly with mlx-lm:
  `mlx_lm.generate --model <mlx model> --adapter-path runs/<name>/best --prompt "..."`
- **CUDA adapters** are standard PEFT adapters: `PeftModel.from_pretrained(base, "runs/<name>/best")`.

## 8. Tuning tips

| Symptom | Try |
|---|---|
| Loss flat from the start | Raise `learning_rate` (LoRA tolerates 1e-4 to 3e-4) and check the mask preview |
| Loss NaN or spiking | Lower `learning_rate`; keep `max_grad_norm: 1.0` |
| Validation loss rises while training loss falls | Fewer epochs, more or more varied data, lower `lora.r`, higher `lora.dropout` |
| Out of memory | Lower `batch_size` (and raise `grad_accum_steps`), lower `max_seq_length`, use `qlora`, or set `lora.num_layers` (Mac) |
| Slow on Modal | Raise `batch_size` if memory allows; use `attn_implementation: flash_attention_2` with the flash-attn package |

## Project layout

```
configs/                 YAML presets (inherit from base.yaml)
src/finetune/
  config.py              typed config + overrides
  data.py                loading, formats, chat template, assistant-only masking
  pipeline.py            backend-agnostic run orchestration (pre-flight, eval/best/early-stop, finalize)
  backends/hf_backend.py CUDA: transformers + PEFT + bitsandbytes
  backends/mlx_backend.py Mac: MLX + mlx-lm LoRA layers
  tracking.py            JSONL + TensorBoard + W&B
  monitor.py             guardrails, top-k adapters, run state
  plots.py, report.py    charts, HTML report, health diagnostics
  memory.py              memory estimator
  evaluate.py, merge.py  base-vs-tuned comparison, merge + GGUF export
  cli.py                 `finetune` command
modal_app.py             Modal functions (train, merge, TensorBoard)
scripts/                 setup + Mac GPU wired-memory helper
tests/                   pytest suite (offline tiny model; HF and MLX end-to-end)
```

Run the tests with `pytest`. They use a tiny randomly initialized model built offline and need
no GPU or network.
