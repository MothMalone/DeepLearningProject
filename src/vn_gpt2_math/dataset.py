"""SFT dataset, collator, and loss-weight construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from .config import PROMPT_TEMPLATE, SAFE_EOS_ID, TrainingConfig


class SFTDataset(Dataset):
    """Tokenize prompt/target pairs and add arithmetic/answer-anchor loss weights."""

    def __init__(self, records: list[dict], tokenizer, config: TrainingConfig):
        self.records = records
        self.tok = tokenizer
        self.config = config
        self.stable_anchor_ids = self.tok("####", add_special_tokens=False)["input_ids"]
        self.stable_anchor_len = len(self.stable_anchor_ids)
        self.operator_token_ids = self._find_operator_token_ids(["=", "+", "-", "*", "/"])

    def __len__(self) -> int:
        return len(self.records)

    def _find_operator_token_ids(self, operators: list[str]) -> set[int]:
        token_ids = set()
        vocab_size = getattr(self.tok, "vocab_size", 50257)
        for token_id in range(vocab_size):
            try:
                decoded = self.tok.decode([token_id]).strip()
            except Exception:
                continue
            if decoded in operators:
                token_ids.add(token_id)
        return token_ids

    def _build_arithmetic_weights(self, ids: list[int], prompt_len: int) -> list[float]:
        weights = [0.0] * prompt_len + [1.0] * (len(ids) - prompt_len)
        response_ids = ids[prompt_len:]
        window = 3

        for pos, token_id in enumerate(response_ids):
            if token_id in self.operator_token_ids:
                for rel in range(max(0, pos - window), min(len(response_ids), pos + window + 1)):
                    weights[prompt_len + rel] = max(weights[prompt_len + rel], self.config.computation_weight)

        anchor_start = -1
        for pos in range(len(response_ids) - self.stable_anchor_len + 1):
            if response_ids[pos : pos + self.stable_anchor_len] == self.stable_anchor_ids:
                anchor_start = pos
                break

        if anchor_start >= 0:
            start = prompt_len + anchor_start
        else:
            start = max(prompt_len, len(ids) - 8)
        for idx in range(start, len(ids)):
            weights[idx] = max(weights[idx], self.config.anchor_weight)

        if ids and ids[-1] == SAFE_EOS_ID:
            weights[-1] = max(weights[-1], self.config.eos_weight)
        return weights

    def __getitem__(self, index: int) -> Dict[str, List[int] | List[float]]:
        record = self.records[index]
        prompt = PROMPT_TEMPLATE.format(q=record["query_vi"].strip())
        response = record.get("_target", record.get("response_vi", "")).strip()

        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = self.tok(response, add_special_tokens=False)["input_ids"]

        max_len_without_eos = self.config.max_length - 1
        ids = (p_ids + r_ids)[:max_len_without_eos]
        labels = ([-100] * len(p_ids) + r_ids)[:max_len_without_eos]
        ids.append(SAFE_EOS_ID)
        labels.append(SAFE_EOS_ID)

        prompt_end = min(len(p_ids), len(ids) - 1)
        weights = self._build_arithmetic_weights(ids, prompt_end)
        ids = [min(t, SAFE_EOS_ID) for t in ids]
        labels = [(-100 if t == -100 else min(t, SAFE_EOS_ID)) for t in labels]

        assert len(ids) == len(labels) == len(weights)
        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": [1] * len(ids),
            "loss_weights": weights,
        }


@dataclass
class PadCollator:
    pad_id: int = SAFE_EOS_ID

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": [], "loss_weights": []}
        for item in batch:
            pad = max_len - len(item["input_ids"])
            out["input_ids"].append(item["input_ids"] + [self.pad_id] * pad)
            out["attention_mask"].append(item["attention_mask"] + [0] * pad)
            out["labels"].append(item["labels"] + [-100] * pad)
            out["loss_weights"].append(item["loss_weights"] + [0.0] * pad)
        return {
            "input_ids": torch.tensor(out["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(out["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(out["labels"], dtype=torch.long),
            "loss_weights": torch.tensor(out["loss_weights"], dtype=torch.float32),
        }
