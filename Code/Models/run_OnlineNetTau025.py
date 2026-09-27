"""Dedicated online runner for the fixed-tau OnlineNet variant."""

import time
from itertools import product

import numpy as np
import torch
from tqdm import tqdm

from Models.OnlineNetTau025 import (
    LOGIT_ADJUSTMENT_TAU,
    OnlineNetTau025Classifier,
    PrequentialLogitAdjustmentState,
    prequential_logit_adjustment_loss,
    validate_binary_labels,
)
from Utils.metric_utils import get_all_metrics
from Utils.utils import seed_everything


METHOD_NAME = "OnlineNetTau025"
PARAMETER_DEFAULTS = {
    "hidden_dim": 64,
    "num_layers": 2,
    "lr": 1e-3,
    "optimizer": "AdamW",
    "logit_adjustment_tau": 0.25,
}


class RunningStats:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = None
        self.max = None

    def update(self, value):
        value = float(value)
        if not np.isfinite(value):
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.min = value if self.min is None else min(self.min, value)
        self.max = value if self.max is None else max(self.max, value)

    def summary(self, prefix):
        variance = max(self.m2 / self.count, 0.0) if self.count else 0.0
        return {
            f"{prefix}_count": self.count,
            f"{prefix}_mean": self.mean,
            f"{prefix}_std": variance ** 0.5,
            f"{prefix}_min": self.min,
            f"{prefix}_max": self.max,
        }


class MaskAwareOnlineScaler:
    def __init__(self, num_features):
        self.n = np.zeros(num_features, dtype=np.float64)
        self.mean = np.zeros(num_features, dtype=np.float64)
        self.m2 = np.zeros(num_features, dtype=np.float64)

    def scale(self, x, mask):
        active = mask > 0
        scaled = np.zeros_like(x, dtype=np.float32)
        variance = np.ones_like(self.mean)
        learned = self.n > 1
        variance[learned] = self.m2[learned] / (self.n[learned] - 1)
        std = np.sqrt(np.maximum(variance[active], 0.0))
        centered = x[active] - self.mean[active]
        valid = std > 1e-8
        scaled_active = centered.astype(np.float32)
        scaled_active[valid] /= std[valid]
        scaled[active] = scaled_active
        return scaled

    def update(self, x, mask):
        active = mask > 0
        self.n[active] += 1
        delta = x[active] - self.mean[active]
        self.mean[active] += delta / self.n[active]
        self.m2[active] += delta * (x[active] - self.mean[active])


def _as_list(value):
    return value if isinstance(value, list) else [value]


def _exact_integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be an exact integer >= {minimum}.")
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}.")
    return int(numeric)


def _positive_float(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and greater than zero.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and greater than zero.") from error
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return numeric


def _normalize_parameter(name, value):
    if name in ("hidden_dim", "num_layers"):
        return _exact_integer(value, name, 1)
    if name == "lr":
        return _positive_float(value, name)
    if name == "optimizer":
        if not isinstance(value, str) or value.lower() != "adamw":
            raise ValueError(f"{METHOD_NAME} supports only AdamW.")
        return "AdamW"
    try:
        tau = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{METHOD_NAME} requires logit_adjustment_tau=0.25."
        ) from error
    if not np.isfinite(tau) or tau != LOGIT_ADJUSTMENT_TAU:
        raise ValueError(f"{METHOD_NAME} requires logit_adjustment_tau=0.25.")
    return tau


def _param_grid(config):
    keys = list(PARAMETER_DEFAULTS)
    values = []
    for key in keys:
        configured = _as_list(config.get(key, PARAMETER_DEFAULTS[key]))
        if not configured:
            raise ValueError(f"{key} parameter grid must not be empty.")
        values.append([_normalize_parameter(key, value) for value in configured])
    for combination in product(*values):
        yield dict(zip(keys, combination))


def _model_kwargs(params, num_features, n_classes):
    return {
        "num_features": int(num_features),
        "hidden_dim": int(params["hidden_dim"]),
        "num_layers": int(params["num_layers"]),
        "n_classes": int(n_classes),
    }


def _prepare_stream(X_haphazard, mask, Y):
    x_values = np.asarray(X_haphazard)
    mask_values = np.asarray(mask)
    if x_values.ndim != 2 or mask_values.ndim != 2:
        raise ValueError("X_haphazard and mask must be two-dimensional.")
    if x_values.shape != mask_values.shape:
        raise ValueError("X_haphazard and mask must have the same shape.")
    if x_values.shape[0] < 1 or x_values.shape[1] < 1:
        raise ValueError("X_haphazard and mask must have a nonzero shape.")

    if np.iscomplexobj(mask_values):
        raise ValueError("mask values must be finite and exactly 0 or 1.")
    try:
        mask_numeric = mask_values.astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "mask values must be finite and exactly 0 or 1."
        ) from error
    valid_mask = np.isfinite(mask_numeric) & (
        (mask_numeric == 0) | (mask_numeric == 1)
    )
    if not np.all(valid_mask):
        raise ValueError("mask values must be finite and exactly 0 or 1.")
    mask_stream = mask_numeric.astype(np.float32)

    if np.iscomplexobj(x_values):
        raise ValueError("Active features must be finite.")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            x_numeric = x_values.astype(np.float32, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Feature values must be numeric.") from error
    active = mask_stream == 1
    if not np.all(np.isfinite(x_numeric[active])):
        raise ValueError("Active features must be finite.")

    labels_values = np.asarray(Y)
    validate_binary_labels(labels_values)
    labels = labels_values.astype(np.int64, copy=False).reshape(-1)
    if x_values.shape[0] != labels.shape[0]:
        raise ValueError(
            "X_haphazard, mask, and Y must contain the same number of instances."
        )

    x_stream = np.where(active, x_numeric, 0.0).astype(np.float32)
    return x_stream, mask_stream, labels


def _branch_inputs(raw_x, raw_mask, scaler):
    fast_x = np.where(raw_mask > 0, raw_x, 0.0).astype(np.float32)
    slow_x = scaler.scale(fast_x, raw_mask)
    return slow_x, fast_x


def _run_once(x_stream, mask_stream, labels, params, config, run_id, device):
    n_classes = _exact_integer(config.get("n_classes", 2), "n_classes", 1)
    if n_classes != 2:
        raise ValueError(f"{METHOD_NAME} requires n_classes=2.")
    seed_everything(run_id)
    model = OnlineNetTau025Classifier(
        **_model_kwargs(params, x_stream.shape[1], n_classes)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]))
    scaler = MaskAwareOnlineScaler(x_stream.shape[1])
    prior_state = PrequentialLogitAdjustmentState()
    statistics = {
        name: RunningStats()
        for name in (
            "log_prior_gap",
            "log_prior_gap_abs",
            "adjustment_gap",
            "adjustment_gap_abs",
            "original_training_ce",
            "logit_adjusted_ce",
        )
    }
    predictions = []
    scores = []
    start_time = time.time()

    for index in tqdm(range(len(labels)), desc=f"{METHOD_NAME} run {run_id + 1}"):
        feedback = prior_state.feedback()
        raw_x = x_stream[index]
        raw_mask = mask_stream[index]
        slow_x, fast_x = _branch_inputs(raw_x, raw_mask, scaler)
        slow_tensor = torch.as_tensor(slow_x.reshape(1, -1), device=device)
        fast_tensor = torch.as_tensor(fast_x.reshape(1, -1), device=device)
        mask_tensor = torch.as_tensor(raw_mask.reshape(1, -1), device=device)
        label_tensor = torch.as_tensor(
            [labels[index]], dtype=torch.long, device=device
        )

        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(
                model(slow_tensor, fast_tensor, mask_tensor), dim=1
            )
        predictions.append(int(probabilities.argmax(dim=1).item()))
        scores.append(float(probabilities[0, 1].item()))

        model.train()
        optimizer.zero_grad()
        training_logits = model(slow_tensor, fast_tensor, mask_tensor)
        loss = prequential_logit_adjustment_loss(
            training_logits,
            label_tensor,
            (feedback["log_prior_0"], feedback["log_prior_1"]),
        )
        with torch.no_grad():
            original_ce = torch.nn.functional.cross_entropy(
                training_logits.detach(), label_tensor
            )
        loss.backward()
        optimizer.step()
        scaler.update(raw_x, raw_mask)
        prior_state.update(labels[index])

        statistics["log_prior_gap"].update(feedback["log_prior_gap"])
        statistics["log_prior_gap_abs"].update(abs(feedback["log_prior_gap"]))
        statistics["adjustment_gap"].update(feedback["adjustment_gap"])
        statistics["adjustment_gap_abs"].update(abs(feedback["adjustment_gap"]))
        statistics["original_training_ce"].update(original_ce.item())
        statistics["logit_adjusted_ce"].update(loss.detach().item())

    taken_time = time.time() - start_time
    metrics = get_all_metrics(
        labels.reshape(-1, 1),
        np.asarray(predictions).reshape(-1, 1),
        np.asarray(scores).reshape(-1, 1),
        taken_time,
    )
    total = sum(prior_state.seen)
    adjustment_summary = {
        "prediction_steps": len(predictions),
        "training_steps": statistics["logit_adjusted_ce"].count,
        "grad_steps": statistics["logit_adjusted_ce"].count,
        "seen_0": prior_state.seen[0],
        "seen_1": prior_state.seen[1],
        "prior_0": (prior_state.seen[0] + 1) / (total + 2),
        "prior_1": (prior_state.seen[1] + 1) / (total + 2),
    }
    for name, statistic in statistics.items():
        adjustment_summary.update(statistic.summary(name))
    return metrics, {"logit_adjustment_feedback": adjustment_summary}


def run_OnlineNetTau025(X_haphazard, mask, Y, num_runs, config):
    x_stream, mask_stream, labels = _prepare_stream(X_haphazard, mask, Y)
    n_classes = _exact_integer(config.get("n_classes", 2), "n_classes", 1)
    if n_classes != 2:
        raise ValueError(f"{METHOD_NAME} requires n_classes=2.")
    run_count = _exact_integer(num_runs, "num_runs", 1)
    use_cuda = bool(config.get("use_cuda", False)) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    result = {}

    for params in _param_grid(config):
        key = METHOD_NAME + "_" + "_".join(
            f"{name}_{value}" for name, value in params.items()
        )
        result[key] = []
        for run_id in range(run_count):
            metrics, _ = _run_once(
                x_stream, mask_stream, labels, params, config, run_id, device
            )
            result[key].append(metrics)
    return result


__all__ = ["run_OnlineNetTau025"]
