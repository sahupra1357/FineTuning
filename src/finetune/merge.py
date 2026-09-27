"""Merge a trained adapter into its base model and optionally export GGUF (llama.cpp / Ollama).

CUDA/HF adapters (PEFT) are merged into the **bf16** base model, even when trained
with QLoRA (merging into 4-bit weights would lose precision). MLX adapters are
fused with mlx-lm; pass ``dequantize=True`` to get fp16 weights (needed for GGUF
or to load the merged model with transformers).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)


def merge_hf_adapter(cfg: Config, adapter_dir: Path, out_dir: Path, push_to_hub: str | None = None) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .backends.hf_backend import _is_gpt_oss

    kwargs = dict(trust_remote_code=cfg.model.trust_remote_code, revision=cfg.model.revision,
                  low_cpu_mem_usage=True)
    import transformers

    kwargs["dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"] = torch.bfloat16
    if _is_gpt_oss(cfg):
        from transformers import Mxfp4Config

        kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"

    print(f"Loading base model {cfg.model.name_or_path} (bf16) ...")
    base = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    print("Merging adapter ...")
    model = model.merge_and_unload()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tok_src = adapter_dir if (Path(adapter_dir) / "tokenizer_config.json").exists() else cfg.model.name_or_path
    AutoTokenizer.from_pretrained(str(tok_src), trust_remote_code=cfg.model.trust_remote_code).save_pretrained(str(out_dir))
    if push_to_hub:
        model.push_to_hub(push_to_hub, private=True)
    print(f"Merged model saved to {out_dir}")
    return out_dir


def merge_mlx_adapter(cfg: Config, adapter_dir: Path, out_dir: Path, dequantize: bool = False) -> Path:
    import json

    from mlx.utils import tree_unflatten
    from mlx_lm.utils import dequantize_model, load, save

    adapter_cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    model_path = adapter_cfg.get("model")
    if not model_path:
        from .backends.mlx_backend import resolve_model_path

        model_path = resolve_model_path(cfg)
    model, tokenizer, config = load(model_path, adapter_path=str(adapter_dir), return_config=True)
    fused = [(n, m.fuse(dequantize=dequantize)) for n, m in model.named_modules() if hasattr(m, "fuse")]
    if fused:
        model.update_modules(tree_unflatten(fused))
    if dequantize:
        model = dequantize_model(model)
        config.pop("quantization", None)
        config.pop("quantization_config", None)
    out_dir = Path(out_dir)
    save(out_dir, model_path, model, tokenizer, config, donate_model=False)
    print(f"Fused MLX model saved to {out_dir}")
    return out_dir


def export_gguf(merged_dir: Path, llama_cpp_dir: Path, outtype: str = "q8_0") -> Path:
    """Convert a merged HF-format model to GGUF with llama.cpp's converter.

    ``outtype``: f16 | bf16 | q8_0 (other quantizations: run llama.cpp's ``llama-quantize`` afterwards).
    """
    script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    if not script.exists():
        raise FileNotFoundError(f"{script} not found — clone https://github.com/ggml-org/llama.cpp and "
                                "pip install -r llama.cpp/requirements.txt")
    out = Path(merged_dir) / f"model-{outtype}.gguf"
    subprocess.run([sys.executable, str(script), str(merged_dir), "--outfile", str(out), "--outtype", outtype],
                   check=True)
    print(f"GGUF written to {out}\n  Ollama: echo 'FROM {out.resolve()}' > Modelfile && ollama create my-model -f Modelfile")
    return out
