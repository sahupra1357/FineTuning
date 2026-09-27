import pytest

from finetune.config import DataConfig
from finetune.data import (IGNORE_INDEX, detect_format, load_records, load_tokenizer, normalize_record,
                           tokenize_example, tokenize_records)


def test_detect_and_normalize_formats():
    assert detect_format({"messages": []}) == "chat"
    assert detect_format({"instruction": "a", "output": "b"}) == "instruction"
    assert detect_format({"prompt": "a", "completion": "b"}) == "completion"
    assert detect_format({"text": "a"}) == "text"
    ex = normalize_record({"instruction": "Do", "input": "this", "output": "ok"}, system_prompt="sys")
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
    assert ex["messages"][1]["content"] == "Do\n\nthis"
    ex = normalize_record({"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]})
    assert ex["messages"][1] == {"role": "assistant", "content": "yo"}
    with pytest.raises(ValueError):
        normalize_record({"messages": [{"role": "user", "content": "no answer"}]})


def test_val_split_is_deterministic(tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text("\n".join(f'{{"text": "row {i}"}}' for i in range(40)))
    cfg = DataConfig(train_path=str(f), val_ratio=0.25, seed=1)
    t1, v1 = load_records(cfg)
    t2, v2 = load_records(cfg)
    assert len(v1) == 10 and len(t1) == 30 and v1 == v2
    assert not {r["text"] for r in t1} & {r["text"] for r in v1}


def test_only_assistant_tokens_are_trained(make_cfg):
    cfg = make_cfg()
    tok = load_tokenizer(cfg)
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is two plus two?"},
            {"role": "assistant", "content": "Four."}, {"role": "user", "content": "And three plus three?"},
            {"role": "assistant", "content": "Six."}]
    out = tokenize_example(tok, {"messages": msgs}, 512, completions_only=True)
    trained = tok.decode([t for t, lab in zip(out["input_ids"], out["labels"]) if lab != IGNORE_INDEX])
    assert "Four." in trained and "Six." in trained
    assert "two plus two" not in trained and "Be brief" not in trained
    assert trained.count("<|im_end|>") == 2  # the end-of-turn token is learned so the model stops
    full = tokenize_example(tok, {"messages": msgs}, 512, completions_only=False)
    assert all(lab != IGNORE_INDEX for lab in full["labels"])


def test_truncation_and_stats(make_cfg):
    cfg = make_cfg("model.max_seq_length=40")
    tok = load_tokenizer(cfg)
    rows, stats = tokenize_records(tok, [{"text": "word " * 200}, {"prompt": "q", "completion": "a"}], cfg)
    assert stats.truncated == 1 and stats.examples == 2 and stats.max_len == 40
    cfg = make_cfg("model.max_seq_length=40", "data.drop_long_examples=true")
    rows, stats = tokenize_records(tok, [{"text": "word " * 200}], cfg)
    assert stats.dropped == 1 and not rows
