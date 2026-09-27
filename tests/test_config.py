from pathlib import Path

import pytest
from pydantic import ValidationError

from finetune.config import Config, load_config

CONFIGS = sorted(p for p in (Path(__file__).parents[1] / "configs").glob("*.yaml") if p.name != "base.yaml")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_presets_load(path):
    cfg = load_config(path)
    assert cfg.run_name and cfg.model.name_or_path
    assert cfg.effective_batch_size >= 1


def test_overrides_and_aliases():
    cfg = load_config(CONFIGS[0], ["training.learning_rate=1e-5", "lora.target_modules=[q_proj,v_proj]"],
                      backend="mac", run_name="x")
    assert cfg.training.learning_rate == 1e-5
    assert cfg.lora.target_modules == ["q_proj", "v_proj"]
    assert cfg.backend == "mlx" and cfg.run_name == "x"


def test_base_inheritance_keeps_defaults():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "7b_qlora.yaml")
    assert cfg.logging.sample_prompts  # from base.yaml
    assert cfg.modal.gpu == "A10G"     # from the preset


def test_rejects_unknown_keys_and_missing_data():
    with pytest.raises(ValidationError):
        Config.model_validate({"model": {"name_or_path": "x"}, "data": {"train_path": "a"}, "bogus": 1})
    with pytest.raises(ValidationError):
        Config.model_validate({"model": {"name_or_path": "x"}, "data": {}})
