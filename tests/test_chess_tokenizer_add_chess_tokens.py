from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.utils.chess_tokenizer import add_chess_tokens


@dataclass
class _DummyTokenizer:
    vocab: Dict[str, int] = field(default_factory=dict)
    additional_special_tokens: List[str] = field(default_factory=list)

    def __len__(self) -> int:  # pragma: no cover
        return len(self.vocab)

    def add_special_tokens(self, spec: Dict[str, List[str]]) -> int:
        tokens = list(spec.get("additional_special_tokens") or [])
        self.additional_special_tokens = tokens
        added = 0
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def add_tokens(self, tokens: List[str], special_tokens: bool = False) -> int:  # noqa: ARG002
        added = 0
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added


def test_add_chess_tokens_preserves_existing_additional_special_tokens():
    tokenizer = _DummyTokenizer(
        vocab={
            "<|im_start|>": 0,
            "<|im_end|>": 1,
        },
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )

    add_chess_tokens(tokenizer=tokenizer, model=None, mode="tags_only", add_think_tags=True)

    assert "<|im_start|>" in tokenizer.additional_special_tokens
    assert "<|im_end|>" in tokenizer.additional_special_tokens
    assert "<think>" in tokenizer.additional_special_tokens
    assert "</think>" in tokenizer.additional_special_tokens
    assert "<uci_move>" in tokenizer.additional_special_tokens
    assert "</uci_move>" in tokenizer.additional_special_tokens


def test_add_chess_tokens_does_not_duplicate_tokens():
    tokenizer = _DummyTokenizer(
        vocab={"<|im_start|>": 0},
        additional_special_tokens=["<|im_start|>", "<uci_move>"],
    )

    add_chess_tokens(tokenizer=tokenizer, model=None, mode="tags_only", add_think_tags=False)

    assert tokenizer.additional_special_tokens.count("<|im_start|>") == 1
    assert tokenizer.additional_special_tokens.count("<uci_move>") == 1
    assert tokenizer.additional_special_tokens.count("</uci_move>") == 1
