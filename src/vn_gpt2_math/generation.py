"""Greedy, beam, and self-consistency generation helpers."""

from __future__ import annotations

from collections import Counter

import torch
from tqdm.auto import tqdm

from .answers import clean_model_output, parse_number
from .config import PROMPT_TEMPLATE, SAFE_EOS_ID, DecodingConfig


def extract_numeric_answer_for_voting(text: str) -> str:
    import re

    match = re.search(r"(?i)(?:####\s*|đáp án là\s*[:：]?\s*|câu trả lời là\s*[:：]?\s*|\\boxed\{)+(-?\d+(?:[.,]\d+)?)", text)
    if match:
        return match.group(1).replace(",", ".")
    numbers = re.findall(r"(-?\d+(?:[.,]\d+)?)", text)
    return numbers[-1].replace(",", ".") if numbers else "NONE"


def generate_outputs(model, tokenizer, records: list[dict], config: DecodingConfig, batch_size: int = 8, device: str = "cuda") -> list[dict]:
    """Generate competition-format outputs. The model must already be frozen/eval."""
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = SAFE_EOS_ID
    outputs = []

    for start in tqdm(range(0, len(records), batch_size), desc="Generating"):
        batch = records[start : start + batch_size]
        prompts = [PROMPT_TEMPLATE.format(q=r["query_vi"].strip()) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True, max_length=256, add_special_tokens=False).to(device)
        ids = enc["input_ids"].clamp(max=model.config.vocab_size - 1)
        attention_mask = enc["attention_mask"]
        prompt_len = ids.shape[1]

        with torch.inference_mode():
            if config.use_beam_search:
                generated = model.generate(
                    input_ids=ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    num_beams=config.num_beams,
                    num_return_sequences=1,
                    length_penalty=0.8,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    pad_token_id=SAFE_EOS_ID,
                    eos_token_id=SAFE_EOS_ID,
                )
                n_ret = 1
            else:
                generated = model.generate(
                    input_ids=ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=True,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    num_return_sequences=config.num_samples,
                    pad_token_id=SAFE_EOS_ID,
                    eos_token_id=SAFE_EOS_ID,
                )
                n_ret = config.num_samples

        for j, record in enumerate(batch):
            candidates = tokenizer.batch_decode(generated[j * n_ret : (j + 1) * n_ret, prompt_len:], skip_special_tokens=True)
            if n_ret == 1:
                best_output = candidates[0]
            else:
                parsed = []
                for candidate in candidates:
                    answer = extract_numeric_answer_for_voting(candidate)
                    number = parse_number(answer) if answer != "NONE" else None
                    if number is not None:
                        parsed.append((number, candidate))
                if parsed:
                    consensus = Counter(round(num, 4) for num, _ in parsed).most_common(1)[0][0]
                    best_output = sorted([text for num, text in parsed if round(num, 4) == consensus], key=len)[0]
                else:
                    best_output = sorted(candidates, key=len)[0]

            outputs.append(
                {
                    "id": record.get("id", len(outputs)),
                    "query_vi": record["query_vi"],
                    "type": record.get("type"),
                    "model_output": clean_model_output(best_output),
                }
            )
    return outputs
