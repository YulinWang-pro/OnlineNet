"""Standalone OnlineNet model with fixed prequential logit adjustment."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOGIT_ADJUSTMENT_TAU = 0.25
_BINARY_LABEL_ERROR = "OnlineNetTau025 requires labels exactly 0 or 1."


def validate_binary_labels(labels):
    """Validate that every label is a finite numeric zero or one."""
    valid = False
    try:
        if isinstance(labels, torch.Tensor):
            if not labels.is_complex():
                valid = bool(
                    torch.isfinite(labels).all().item()
                    and ((labels == 0) | (labels == 1)).all().item()
                )
        else:
            values = np.asarray(labels)
            if values.dtype.kind in "biuf":
                valid = bool(
                    np.all(np.isfinite(values))
                    and np.all((values == 0) | (values == 1))
                )
    except (TypeError, ValueError, RuntimeError):
        valid = False

    if not valid:
        raise ValueError(_BINARY_LABEL_ERROR)


class PrequentialLogitAdjustmentState:
    def __init__(self):
        self.seen = [0, 0]

    def feedback(self):
        total = sum(self.seen)
        prior_0 = (self.seen[0] + 1) / (total + 2)
        prior_1 = (self.seen[1] + 1) / (total + 2)
        log_prior_0 = math.log(prior_0)
        log_prior_1 = math.log(prior_1)
        log_prior_gap = log_prior_0 - log_prior_1
        return {
            "prior_0": prior_0,
            "prior_1": prior_1,
            "log_prior_0": log_prior_0,
            "log_prior_1": log_prior_1,
            "log_prior_gap": log_prior_gap,
            "adjustment_gap": LOGIT_ADJUSTMENT_TAU * log_prior_gap,
        }

    def update(self, label):
        validate_binary_labels(label)
        if isinstance(label, torch.Tensor):
            if label.numel() != 1:
                raise ValueError(
                    "PrequentialLogitAdjustmentState.update requires a single "
                    "binary label."
                )
            value = int(label.item())
        else:
            values = np.asarray(label)
            if values.size != 1:
                raise ValueError(
                    "PrequentialLogitAdjustmentState.update requires a single "
                    "binary label."
                )
            value = int(values.reshape(-1)[0])
        self.seen[value] += 1


def prequential_logit_adjustment_loss(logits, labels, log_priors):
    validate_binary_labels(labels)
    labels = torch.as_tensor(labels, device=logits.device, dtype=torch.long)
    adjustment = LOGIT_ADJUSTMENT_TAU * logits.new_tensor(log_priors)
    return F.cross_entropy(logits + adjustment, labels)


class OnlineNetTau025Classifier(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_layers=2, n_classes=2):
        super().__init__()
        self.num_features = int(num_features)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.n_classes = int(n_classes)
        if self.num_layers < 1:
            raise ValueError("OnlineNetTau025 requires num_layers >= 1.")
        if self.n_classes != 2:
            raise ValueError("OnlineNetTau025 requires n_classes=2.")

        self.slow_layers = nn.ModuleList(
            nn.Linear(
                self.num_features if index == 0 else self.hidden_dim,
                self.hidden_dim,
            )
            for index in range(self.num_layers)
        )
        self.fast_layers = nn.ModuleList(
            nn.Linear(
                2 * self.num_features if index == 0 else self.hidden_dim,
                self.hidden_dim,
            )
            for index in range(self.num_layers)
        )
        self.classifier = nn.Linear(self.hidden_dim, 2)

    def encode(self, slow_x, fast_x, mask):
        slow_state = slow_x
        fast_state = torch.cat([fast_x, mask], dim=-1)
        for slow_layer, fast_layer in zip(self.slow_layers, self.fast_layers):
            slow_state = F.relu(slow_layer(slow_state))
            fast_state = F.relu(fast_layer(fast_state))
            slow_state = slow_state * (2.0 * torch.sigmoid(fast_state))
        return slow_state

    def forward(self, slow_x, fast_x, mask):
        return self.classifier(self.encode(slow_x, fast_x, mask))


__all__ = [
    "LOGIT_ADJUSTMENT_TAU",
    "OnlineNetTau025Classifier",
    "PrequentialLogitAdjustmentState",
    "prequential_logit_adjustment_loss",
    "validate_binary_labels",
]
