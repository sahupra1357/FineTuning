"""Build a tiny random Llama model + BPE tokenizer with a ChatML template, fully offline.

Used by the tests and for smoke-testing the pipeline without network access.
"""

from __future__ import annotations

import json
from pathlib import Path

CHATML = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def build_tiny_model(out_dir: Path, corpus_file: Path | None = None) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    out_dir = Path(out_dir)
    if (out_dir / "config.json").exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = ["hello world", "the quick brown fox jumps over the lazy dog"]
    if corpus_file and corpus_file.exists():
        for line in corpus_file.read_text().splitlines():
            rec = json.loads(line)
            texts += [m["content"] for m in rec.get("messages", [])]

    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=600, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  bos_token=None, unk_token=None)
    tok.chat_template = CHATML
    tok.save_pretrained(out_dir)

    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True)
    LlamaForCausalLM(cfg).save_pretrained(out_dir)
    return out_dir


if __name__ == "__main__":
    import sys

    print(build_tiny_model(Path(sys.argv[1]), Path("data/sample_chat.jsonl")))
