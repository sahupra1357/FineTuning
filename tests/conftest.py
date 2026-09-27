import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from tiny_model import build_tiny_model

    return build_tiny_model(tmp_path_factory.mktemp("tiny") / "tiny-llama", ROOT / "data" / "sample_chat.jsonl")


@pytest.fixture
def make_cfg(tiny_model_dir, tmp_path):
    from finetune.config import load_config

    def _make(*overrides, **top):
        ov = [f"model.name_or_path={tiny_model_dir}", "logging.tensorboard=false",
              "training.max_steps=6", "training.eval_every_steps=3", "training.save_every_steps=3",
              "logging.log_every_steps=1", "logging.sample_max_new_tokens=4", *overrides]
        top.setdefault("output_dir", str(tmp_path / "runs"))
        return load_config(ROOT / "configs" / "smoke_test.yaml", ov, **top)

    return _make
