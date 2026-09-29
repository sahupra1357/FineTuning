"""Run the fine-tuning pipeline on Modal GPUs.

Setup (once):
    pip install modal && modal setup
    modal secret create huggingface HF_TOKEN=hf_xxx          # gated models / faster downloads
    modal secret create wandb WANDB_API_KEY=xxx              # only if logging.wandb.enabled

Train (GPU type, timeout, secrets all come from the YAML `modal:` section):
    modal run modal_app.py --config configs/7b_qlora.yaml
    modal run modal_app.py --config configs/7b_qlora.yaml --wandb --run-name my-exp
    modal run modal_app.py --config configs/7b_qlora.yaml --dry-run-steps 20       # quick sanity run
    modal run modal_app.py --config configs/7b_qlora.yaml --overrides "training.learning_rate=1e-4;lora.r=32"
    modal run --detach modal_app.py --config ...     # keep running after you close the terminal

Resume: re-run with the same --run-name; training continues from the latest checkpoint.
(Modal also retries once automatically after a pre-emption or timeout.)

Watch / fetch results:
    modal serve modal_app.py                          # prints a live TensorBoard URL over all runs
    modal volume ls finetune-runs
    modal volume get finetune-runs <run-name> ./runs/  # report.html, plots/, best/ adapter, ...

Merge the best adapter into the base model (saved to the runs volume):
    modal run modal_app.py::merge --run-name <run-name>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

# The local entrypoint imports transformers (tokenizer config) without torch; hide that advisory.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))  # local import of finetune.config for the entrypoint

APP_NAME = "finetune"
RUNS_VOL = os.environ.get("FINETUNE_RUNS_VOLUME", "finetune-runs")
CACHE_VOL = os.environ.get("FINETUNE_CACHE_VOLUME", "finetune-hf-cache")
DATA_VOL = os.environ.get("FINETUNE_DATA_VOLUME", "finetune-data")
# Fallbacks for Modal client versions without Function.with_options (set before `modal run`).
DEFAULT_GPU = os.environ.get("FINETUNE_GPU", "A100-80GB")
DEFAULT_SECRETS = [s for s in os.environ.get("FINETUNE_SECRETS", "").split(",") if s]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.55",   # gpt-oss support
        "peft>=0.17",
        "accelerate>=1.0",
        "bitsandbytes>=0.45",
        "datasets>=3.0",
        "safetensors",
        "huggingface_hub",
        "pydantic>=2.5",
        "pyyaml",
        "matplotlib",
        "tqdm",
        "tensorboard",
        "tensorboardX",
        "wandb",
    )
    .env({
        "HF_HOME": "/cache/hf",
        "HF_XET_HIGH_PERFORMANCE": "1",  # fast Xet downloads (replaces deprecated hf_transfer)
        "PYTHONPATH": "/root/src",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir(str(ROOT / "src" / "finetune"), remote_path="/root/src/finetune")
)

app = modal.App(APP_NAME, image=image)
runs_volume = modal.Volume.from_name(RUNS_VOL, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOL, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOL, create_if_missing=True)
VOLUMES = {"/runs": runs_volume, "/cache": cache_volume, "/data": data_volume}


def _secrets(names: list[str]) -> list[modal.Secret]:
    return [modal.Secret.from_name(n) for n in names]


@app.function(
    gpu=DEFAULT_GPU,
    volumes=VOLUMES,
    timeout=24 * 3600,
    retries=modal.Retries(max_retries=1, initial_delay=10.0),
    secrets=_secrets(DEFAULT_SECRETS),
)
def train_remote(cfg_dict: dict, dry_run_steps: int = 0, force: bool = False) -> str:
    from finetune.config import Config
    from finetune.pipeline import run_training

    cfg = Config.model_validate(cfg_dict)
    cfg.output_dir = "/runs"
    cfg.backend = "hf"

    def persist() -> None:
        try:
            runs_volume.commit()
        except Exception as e:  # commit failures must not crash training
            print(f"volume commit failed: {e}")

    try:
        run_dir = run_training(cfg, persist=persist, dry_run_steps=dry_run_steps or None, force=force)
    finally:
        persist()
        try:
            cache_volume.commit()
        except Exception:
            pass
    return str(run_dir)


@app.function(volumes=VOLUMES, timeout=3 * 3600, memory=131072, secrets=_secrets(DEFAULT_SECRETS))
def merge(run_name: str, adapter: str = "best", push_to_hub: str = "") -> str:
    """Merge an adapter into its bf16 base model (CPU, 128 GB RAM) -> /runs/<run>/merged."""
    from finetune.config import load_config
    from finetune.merge import merge_hf_adapter

    run_dir = Path("/runs") / run_name
    cfg = load_config(run_dir / "config.yaml")
    out = merge_hf_adapter(cfg, run_dir / adapter, run_dir / "merged", push_to_hub=push_to_hub or None)
    runs_volume.commit()
    return str(out)


@app.function(volumes={"/runs": runs_volume}, max_containers=1, timeout=24 * 3600)
@modal.concurrent(max_inputs=100)
@modal.web_server(6006, startup_timeout=120)
def tensorboard() -> None:
    """Live TensorBoard over every run in the volume (reloads the volume every 30 s)."""
    import subprocess
    import threading
    import time

    def refresh() -> None:
        while True:
            time.sleep(30)
            try:
                runs_volume.reload()
            except Exception:
                pass

    threading.Thread(target=refresh, daemon=True).start()
    subprocess.Popen("tensorboard --logdir /runs --host 0.0.0.0 --port 6006 --reload_interval 30", shell=True)


def _upload_data(cfg, run_name: str) -> None:
    """Upload local dataset files to the data volume and point the config at them."""
    uploads = {}
    for attr in ("train_path", "val_path"):
        path = getattr(cfg.data, attr)
        if path and Path(path).exists():
            remote = f"/{run_name}/{Path(path).name}"
            uploads[remote] = path
            setattr(cfg.data, attr, f"/data{remote}")
    if uploads:
        with data_volume.batch_upload(force=True) as batch:
            for remote, local in uploads.items():
                batch.put_file(local, remote)
        print(f"Uploaded dataset(s) to volume {DATA_VOL}: {list(uploads)}")


@app.local_entrypoint()
def main(
    config: str,
    run_name: str = "",
    wandb: bool = False,
    no_tensorboard: bool = False,
    dry_run_steps: int = 0,
    overrides: str = "",
    force: bool = False,
):
    from finetune.config import load_config
    from finetune.memory import estimate_memory

    ov = [o.strip() for o in overrides.split(";") if o.strip()]
    if wandb:
        ov.append("logging.wandb.enabled=true")
    if no_tensorboard:
        ov.append("logging.tensorboard=false")
    cfg = load_config(config, ov, run_name=run_name or None, backend="hf")

    secrets = list(dict.fromkeys(cfg.modal.secrets + (["wandb"] if cfg.logging.wandb.enabled else [])))
    gpu = cfg.modal.gpu if cfg.modal.gpu_count == 1 else f"{cfg.modal.gpu}:{cfg.modal.gpu_count}"
    est = estimate_memory(cfg)
    print(f"Run {cfg.run_name}: {cfg.model.name_or_path} [{cfg.method}] on {gpu}")
    print(f"Memory estimate: ~{est['total_gb']} GB of {est.get('available_gb')} GB ({est.get('fits', '?')})")
    if est.get("fits") == "no" and not force:
        raise SystemExit("Estimated not to fit on this GPU; pick a larger modal.gpu or pass --force.")

    _upload_data(cfg, str(cfg.run_name))

    fn = train_remote
    try:
        fn = train_remote.with_options(gpu=gpu, timeout=int(cfg.modal.timeout_hours * 3600),
                                       secrets=_secrets(secrets))
    except AttributeError:
        print(f"modal client lacks Function.with_options; using FINETUNE_GPU={DEFAULT_GPU} and "
              f"FINETUNE_SECRETS={DEFAULT_SECRETS} (set these env vars to change them).")
    run_dir = fn.remote(cfg.to_dict(), dry_run_steps=dry_run_steps, force=force)
    print(f"\nDone: {run_dir}")
    print(f"Download results: modal volume get {RUNS_VOL} {cfg.run_name} ./runs/")
