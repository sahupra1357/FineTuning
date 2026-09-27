"""Dataset loading, normalization, chat templating and loss masking.

Both backends use this module so that the exact same tokens (and the same
"which tokens are trained" mask) are produced on Mac and on CUDA.

Accepted record formats (auto-detected per record):
  chat         {"messages": [{"role": "user", "content": ...}, {"role": "assistant", ...}]}
  instruction  {"instruction": ..., "input": optional, "output": ...}
  completion   {"prompt": ..., "completion": ...}
  text         {"text": ...}   (whole text is trained)
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import Config, DataConfig

log = logging.getLogger(__name__)

IGNORE_INDEX = -100

# Minimal template used only when a tokenizer ships without a chat template (base models).
FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>\n' + message['content'] + eos_token + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
)


# --------------------------------------------------------------------------- loading
def _read_json_records(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict):  # {"data": [...]} style
        for v in data.values():
            if isinstance(v, list):
                return v
        raise ValueError(f"{path}: JSON object without a list of records")
    return data


def _load_source(path: str | None, cfg: DataConfig, split: str | None) -> list[dict]:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"dataset file not found: {p}")
        return _read_json_records(p)
    if cfg.hf_dataset and split:
        from datasets import load_dataset

        ds = load_dataset(cfg.hf_dataset, cfg.hf_config, split=split)
        return [dict(r) for r in ds]
    return []


def load_records(cfg: DataConfig) -> tuple[list[dict], list[dict]]:
    """Return (train_records, val_records) as raw dicts."""
    train = _load_source(cfg.train_path, cfg, cfg.hf_train_split)
    val = _load_source(cfg.val_path, cfg, cfg.hf_val_split) if (cfg.val_path or cfg.hf_val_split) else []

    rng = random.Random(cfg.seed)
    if not val and cfg.val_ratio > 0 and len(train) > 1:
        idx = list(range(len(train)))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(train) * cfg.val_ratio)))
        val_idx = set(idx[:n_val])
        val = [train[i] for i in idx[:n_val]]
        train = [r for i, r in enumerate(train) if i not in val_idx]

    if cfg.max_train_samples:
        train = train[: cfg.max_train_samples]
    if cfg.max_val_samples:
        val = val[: cfg.max_val_samples]
    return train, val


# --------------------------------------------------------------------------- normalize
def detect_format(record: dict) -> str:
    if "messages" in record or "conversations" in record:
        return "chat"
    if "instruction" in record and ("output" in record or "response" in record):
        return "instruction"
    if "prompt" in record and ("completion" in record or "response" in record):
        return "completion"
    if "text" in record:
        return "text"
    raise ValueError(f"cannot detect record format from keys {sorted(record)}")


_ROLE_MAP = {"human": "user", "gpt": "assistant", "bot": "assistant", "model": "assistant"}


def _norm_messages(msgs: list[dict]) -> list[dict]:
    out = []
    for m in msgs:
        role = m.get("role", m.get("from"))
        content = m.get("content", m.get("value"))
        if role is None or content is None:
            raise ValueError(f"bad chat message: {m}")
        out.append({"role": _ROLE_MAP.get(role, role), "content": str(content)})
    return out


def normalize_record(record: dict, fmt: str = "auto", system_prompt: str | None = None) -> dict:
    """Convert any supported record into ``{"messages": [...]}`` or ``{"text": ...}``."""
    fmt = detect_format(record) if fmt == "auto" else fmt
    if fmt == "text":
        return {"text": str(record["text"])}

    if fmt == "chat":
        messages = _norm_messages(record.get("messages") or record.get("conversations"))
    elif fmt == "instruction":
        user = str(record["instruction"])
        if record.get("input"):
            user = f"{user}\n\n{record['input']}"
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(record.get("output", record.get("response")))},
        ]
    elif fmt == "completion":
        messages = [
            {"role": "user", "content": str(record["prompt"])},
            {"role": "assistant", "content": str(record.get("completion", record.get("response")))},
        ]
    else:
        raise ValueError(f"unknown format {fmt}")

    if system_prompt and messages[0]["role"] != "system":
        messages = [{"role": "system", "content": system_prompt}] + messages
    if not any(m["role"] == "assistant" for m in messages):
        raise ValueError("chat record has no assistant turn to train on")
    return {"messages": messages}


# --------------------------------------------------------------------------- tokenizer
def load_tokenizer(cfg: Config, name_or_path: str | None = None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        name_or_path or cfg.model.name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
        revision=cfg.model.revision,
    )
    prepare_tokenizer(tok, cfg.model.chat_template)
    return tok


def prepare_tokenizer(tok, chat_template: str | None = None):
    if chat_template:
        tok.chat_template = chat_template
    elif not getattr(tok, "chat_template", None):
        log.warning("tokenizer has no chat template; using a minimal fallback template")
        tok.chat_template = FALLBACK_CHAT_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def render_prompt(tok, messages: list[dict]) -> str:
    """Chat-templated prompt ending with the assistant generation header."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# --------------------------------------------------------------------------- tokenize
def _encode(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def _chat_segments(tok, messages: list[dict], completions_only: bool) -> tuple[list[int], list[int]]:
    """Tokenize a conversation; mask marks the tokens that contribute to the loss.

    For every assistant turn k, the trained span is the text between
    ``template(messages[:k], add_generation_prompt=True)`` and ``template(messages[:k+1])``
    — i.e. the assistant reply plus its end-of-turn token, but not the header.
    """
    full_text = tok.apply_chat_template(messages, tokenize=False)
    if not completions_only:
        ids = _encode(tok, full_text)
        return ids, [1] * len(ids)

    # Build the text as alternating (untrained, trained) character spans.
    spans: list[tuple[str, int]] = []
    cursor = 0
    for k, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix = tok.apply_chat_template(messages[:k], tokenize=False, add_generation_prompt=True)
        upto = tok.apply_chat_template(messages[: k + 1], tokenize=False)
        if not (full_text.startswith(prefix) and full_text.startswith(upto) and len(prefix) >= cursor):
            return _chat_segments_by_search(tok, messages, full_text)
        spans.append((full_text[cursor : len(prefix)], 0))
        spans.append((full_text[len(prefix) : len(upto)], 1))
        cursor = len(upto)
    spans.append((full_text[cursor:], 0))

    ids: list[int] = []
    mask: list[int] = []
    for text, m in spans:
        if not text:
            continue
        piece = _encode(tok, text)
        ids.extend(piece)
        mask.extend([m] * len(piece))
    return ids, mask


def _chat_segments_by_search(tok, messages: list[dict], full_text: str) -> tuple[list[int], list[int]]:
    """Fallback for templates that are not prefix-consistent: locate assistant contents by string search."""
    enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    char_mask = [0] * len(full_text)
    pos = 0
    for msg in messages:
        idx = full_text.find(msg["content"], pos)
        if idx < 0:
            continue
        end = idx + len(msg["content"])
        if msg["role"] == "assistant":
            for i in range(idx, end):
                char_mask[i] = 1
        pos = end
    mask = [1 if any(char_mask[s:e]) else 0 for s, e in offsets]
    # Train the token right after each assistant span too (end-of-turn marker).
    for i in range(len(mask) - 1, 0, -1):
        if mask[i - 1] == 1 and mask[i] == 0:
            mask[i] = 1
    return ids, mask


def tokenize_example(tok, example: dict, max_seq_length: int, completions_only: bool = True) -> dict:
    if "text" in example:
        ids = _encode(tok, example["text"])
        if tok.eos_token_id is not None and (not ids or ids[-1] != tok.eos_token_id):
            ids.append(tok.eos_token_id)
        mask = [1] * len(ids)
    else:
        ids, mask = _chat_segments(tok, example["messages"], completions_only)

    truncated = len(ids) > max_seq_length
    ids, mask = ids[:max_seq_length], mask[:max_seq_length]
    labels = [t if m else IGNORE_INDEX for t, m in zip(ids, mask)]
    return {"input_ids": ids, "labels": labels, "truncated": truncated, "n_trainable": sum(mask)}


@dataclass
class DatasetStats:
    split: str
    examples: int = 0
    dropped: int = 0
    truncated: int = 0
    total_tokens: int = 0
    trainable_tokens: int = 0
    mean_len: float = 0.0
    p50_len: int = 0
    p95_len: int = 0
    max_len: int = 0
    lengths: list[int] = field(default_factory=list, repr=False)

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("lengths")
        d["trainable_fraction"] = round(self.trainable_tokens / max(1, self.total_tokens), 3)
        return d


def tokenize_records(
    tok, records: Iterable[dict], cfg: Config, split: str = "train"
) -> tuple[list[dict], DatasetStats]:
    stats = DatasetStats(split=split)
    out: list[dict] = []
    for raw in records:
        ex = normalize_record(raw, cfg.data.format, cfg.data.system_prompt)
        t = tokenize_example(tok, ex, cfg.model.max_seq_length, cfg.data.train_on_completions_only)
        if t["n_trainable"] == 0 or (t["truncated"] and cfg.data.drop_long_examples):
            stats.dropped += 1
            continue
        stats.truncated += int(t["truncated"])
        stats.total_tokens += len(t["input_ids"])
        stats.trainable_tokens += t["n_trainable"]
        stats.lengths.append(len(t["input_ids"]))
        out.append({"input_ids": t["input_ids"], "labels": t["labels"]})

    stats.examples = len(out)
    if stats.lengths:
        s = sorted(stats.lengths)
        stats.mean_len = round(sum(s) / len(s), 1)
        stats.p50_len = s[len(s) // 2]
        stats.p95_len = s[min(len(s) - 1, int(len(s) * 0.95))]
        stats.max_len = s[-1]
    return out, stats


def prepare_datasets(cfg: Config, tok) -> tuple[list[dict], list[dict], dict[str, dict]]:
    """Load, normalize and tokenize train/val. Returns (train, val, stats_by_split)."""
    train_raw, val_raw = load_records(cfg.data)
    train, tstats = tokenize_records(tok, train_raw, cfg, "train")
    val, vstats = tokenize_records(tok, val_raw, cfg, "val")
    if not train:
        raise ValueError("no usable training examples after tokenization (check format / masking)")
    return train, val, {"train": tstats.summary(), "val": vstats.summary()}


def render_mask_preview(tok, example: dict, max_chars: int = 1500) -> str:
    """Human-readable view of what is trained: trained text is wrapped in [[ ]]."""
    parts: list[str] = []
    cur: list[int] = []
    cur_trained: bool | None = None
    for t, lab in zip(example["input_ids"], example["labels"]):
        trained = lab != IGNORE_INDEX
        if cur_trained is not None and trained != cur_trained:
            text = tok.decode(cur)
            parts.append(f"[[{text}]]" if cur_trained else text)
            cur = []
        cur.append(t)
        cur_trained = trained
    if cur:
        text = tok.decode(cur)
        parts.append(f"[[{text}]]" if cur_trained else text)
    s = "".join(parts)
    return s if len(s) <= max_chars else s[:max_chars] + " …"


def epoch_batches(n: int, batch_size: int, seed: int, epoch: int) -> list[list[int]]:
    """Deterministic shuffled index batches for one epoch."""
    rng = random.Random(seed + epoch)
    idx = list(range(n))
    rng.shuffle(idx)
    return [idx[i : i + batch_size] for i in range(0, n, batch_size)]
