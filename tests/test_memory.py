from pathlib import Path

from finetune.config import load_config
from finetune.memory import estimate_memory

CFG = Path(__file__).parents[1] / "configs" / "7b_lora.yaml"


def test_estimate_uses_real_sequence_lengths():
    cfg = load_config(CFG, ["model.name_or_path=/nonexistent/offline-model"])  # heuristics, no network
    worst = estimate_memory(cfg)
    stats = {"train": {"max_len": 85}, "val": {"max_len": 89}}
    real = estimate_memory(cfg, stats)
    assert worst["seq_len"] == cfg.model.max_seq_length and real["seq_len"] == 89
    assert real["seq_len_source"] == "longest example in dataset"
    assert real["logits_gb"] < worst["logits_gb"] / 10
    assert real["total_gb"] < worst["total_gb"]
    assert real["weights_gb"] == worst["weights_gb"]  # weights don't depend on data


def test_estimate_never_exceeds_max_seq_length():
    cfg = load_config(CFG, ["model.name_or_path=/nonexistent/offline-model", "model.max_seq_length=512"])
    est = estimate_memory(cfg, {"train": {"max_len": 512}})
    assert est["seq_len"] == 512
