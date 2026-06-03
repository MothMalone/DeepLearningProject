"""Greedy, beam, self-consistency, and frozen-checkpoint ensemble helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from .answers import clean_model_output, parse_number, vote_candidate_texts
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


def collect_sampling_candidates(
    model,
    tokenizer,
    records: list[dict],
    *,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    batch_size: int = 8,
    device: str = "cuda",
) -> list[list[str]]:
    """Return raw sampled completions grouped by input record for one frozen model."""
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = SAFE_EOS_ID
    grouped: list[list[str]] = [[] for _ in records]

    for start in tqdm(range(0, len(records), batch_size), desc="Collecting candidates"):
        batch = records[start : start + batch_size]
        prompts = [PROMPT_TEMPLATE.format(q=r["query_vi"].strip()) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True, max_length=256, add_special_tokens=False).to(device)
        ids = enc["input_ids"].clamp(max=model.config.vocab_size - 1)
        attention_mask = enc["attention_mask"]
        prompt_len = ids.shape[1]

        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_samples,
                pad_token_id=SAFE_EOS_ID,
                eos_token_id=SAFE_EOS_ID,
            )

        for batch_idx in range(len(batch)):
            rows = generated[batch_idx * num_samples : (batch_idx + 1) * num_samples, prompt_len:]
            grouped[start + batch_idx].extend(tokenizer.batch_decode(rows, skip_special_tokens=True))

    return grouped


def generate_checkpoint_ensemble_outputs(
    checkpoint_paths: list[str | Path],
    tokenizer,
    records: list[dict],
    *,
    samples_per_checkpoint: int = 5,
    max_new_tokens: int = 80,
    temperature: float = 0.4,
    top_k: int = 40,
    top_p: float = 0.90,
    batch_size: int = 8,
    device: str = "cuda",
) -> list[dict]:
    """
    Inference-only ensemble over frozen epoch checkpoints.

    Each checkpoint samples a small candidate set. The final answer is selected by
    numeric majority vote across all candidates, with shortest-output tie-breaking.
    No gradients or optimizer steps are used.
    """
    candidate_bank: list[list[str]] = [[] for _ in records]
    for checkpoint in checkpoint_paths:
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint), local_files_only=True)
        model.config.pad_token_id = SAFE_EOS_ID
        model.config.eos_token_id = SAFE_EOS_ID
        model.to(device)
        model.eval()
        grouped = collect_sampling_candidates(
            model,
            tokenizer,
            records,
            num_samples=samples_per_checkpoint,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            batch_size=batch_size,
            device=device,
        )
        for idx, candidates in enumerate(grouped):
            candidate_bank[idx].extend(candidates)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    outputs = []
    for idx, record in enumerate(records):
        chosen, _debug = vote_candidate_texts(candidate_bank[idx])
        outputs.append(
            {
                "id": record.get("id", idx),
                "query_vi": record["query_vi"],
                "type": record.get("type"),
                "model_output": clean_model_output(chosen),
            }
        )
    return outputs
