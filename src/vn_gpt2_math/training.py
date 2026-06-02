"""Training helpers for weighted full fine-tuning."""

from __future__ import annotations

import torch
from transformers import Trainer


class WeightedLossTrainer(Trainer):
    """HF Trainer that applies token-level loss weights to response tokens."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_weights = inputs.pop("loss_weights", None)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)

        mask = (shift_labels != -100).float()
        if loss_weights is not None:
            shift_weights = loss_weights[:, 1:].contiguous().to(per_token_loss.device)
            per_token_loss = per_token_loss * shift_weights
            weight_sum = (mask * shift_weights).sum().clamp(min=1.0)
            loss = (per_token_loss * mask).sum() / weight_sum
        else:
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)

        return (loss, outputs) if return_outputs else loss
