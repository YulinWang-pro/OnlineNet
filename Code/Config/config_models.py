# Model Configs
import numpy as np
from Utils.utils import dummy_feat, impute_data


WELFORD_BASELINE_SINGLE_RUN_METHODS = {
    "MODLWelfordZScore",
    "auxdropWelfordZScore",
    "OVFMWelfordZScore",
    "MODL_three_learners_25PacketNoScaler",
    "MODL_only_mlpNoScaler",
    "MODL_only_oblrNoScaler",
    "MODL_only_oblr_mlpNoScaler",
    "memory_calibrated_residual",
    "memory_calibrated_residual_StableOnly",
    "memory_calibrated_residual_FastOnly",
    "memory_calibrated_residual_NoDistill",
    "memory_calibrated_residual_RuleGate",
}


def config_welford_baseline_runs(method_name):
    if method_name in WELFORD_BASELINE_SINGLE_RUN_METHODS:
        return 1
    return None


def _default_neural_runs(data_name):
    if "f1" in data_name:
        return 3
    return 3 if data_name in ["susy", "higgs"] else 5


def config_DualNet(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [64],
        "replay_batch_size": [8],
        "ssl_weight": [0.05],
        "distill_weight": [0.1],
        "beta": [0.1],
        "optimizer": ["AdamW"],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetWelfordZScore(data_name):
    return config_DualNet(data_name)


def config_DualNetNoReplay(data_name):
    n_runs, config_dict = config_DualNet(data_name)
    config_dict = dict(config_dict)
    config_dict["memory_size"] = [0]
    config_dict["replay_batch_size"] = [0]
    config_dict["distill_weight"] = [0.0]
    return n_runs, config_dict


def config_DualNetNoReplayWelfordZScore(data_name):
    return config_DualNetNoReplay(data_name)


def config_DualNetNoReplayWelfordZScoreNoSSL(data_name):
    n_runs, config_dict = config_DualNetNoReplayWelfordZScore(data_name)
    config_dict = dict(config_dict)
    config_dict["ssl_weight"] = [0.0]
    config_dict["beta"] = [0.0]
    return n_runs, config_dict


def config_DualNetPrototypeMemory(data_name):
    n_runs, config_dict = config_DualNet(data_name)
    config_dict = dict(config_dict)
    config_dict["memory_size"] = [0]
    config_dict["replay_batch_size"] = [0]
    config_dict["prototype_gamma"] = [0.9]
    config_dict["prototype_weight"] = [0.1]
    return n_runs, config_dict


def config_DualNetPrototypeMemoryWelfordZScore(data_name):
    return config_DualNetPrototypeMemory(data_name)


def _add_dualnet_adaptive_fast_gate_params(config_dict):
    config_dict = dict(config_dict)
    config_dict["adapter_hidden"] = [32]
    config_dict["memory_slots"] = [32]
    config_dict["gamma"] = [0.9]
    config_dict["fast_gamma"] = [0.3]
    config_dict["tau"] = [0.75]
    config_dict["grad_moment_gamma"] = [0.8]
    config_dict["trigger_patience"] = [2]
    config_dict["mask_gamma"] = [0.9]
    config_dict["mask_fast_gamma"] = [0.3]
    config_dict["mask_shift_tau"] = [0.05]
    return config_dict


def _add_dualnet_reliability_params(config_dict):
    config_dict = dict(config_dict)
    config_dict["reliability_alpha_min"] = [0.1]
    config_dict["reliability_alpha_max"] = [1.0]
    config_dict["reliability_hidden"] = [16]
    return config_dict


def config_DualNetAdaptiveFastGateWelfordZScore(data_name):
    n_runs, config_dict = config_DualNetWelfordZScore(data_name)
    return n_runs, _add_dualnet_adaptive_fast_gate_params(config_dict)


def config_DualNetGradientConflictAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetGradientConflictTriggerThresholdAuditAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "adapter_blend_tau": [0.75],
        "conflict_trigger_threshold": [-0.75, -0.5, -0.25, 0.0],
        "candidate_trigger_thresholds": [-0.75, -0.5, -0.25, 0.0, 0.25],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [1, 2],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetZeroInitGradientConflictAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetScaledGradientConflictAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [False],
        "adapter_scale": [0.05, 0.1, 0.2, 0.5, 1.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetZeroInitScaledGradientConflictAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "adapter_scale": [0.05, 0.1, 0.2, 0.5, 1.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetZeroInitScaledNoSoftBoundedDeltaGradientConflictAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "adapter_scale": [0.2],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_OnlineDualNet_Simple(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "lr": [1e-3],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "adapter_scale": [0.2],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def _config_online_dualnet_simple_depth(data_name, num_layers):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "num_layers": [int(num_layers)],
        "lr": [1e-3],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "adapter_scale": [0.2],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_OnlineDualNet_Simple_L1(data_name):
    n_runs, config_dict = _config_online_dualnet_simple_depth(data_name, 1)
    config_dict["num_layers"] = [1]
    return n_runs, config_dict


def config_OnlineDualNet_Simple_L2(data_name):
    n_runs, config_dict = _config_online_dualnet_simple_depth(data_name, 2)
    if data_name=="spam":
        config_dict["lr"] =0.0005
    if data_name=="spambase":
        config_dict["lr"] =0.002
    if data_name=="wbc":
        config_dict["lr"] =0.002
    config_dict["num_layers"] = [2]
    return n_runs, config_dict


def config_OnlineDualNet_Simple_L2_Tune(data_name):
    n_runs = _default_neural_runs(data_name)


    config_dict = {
        "hidden_dim": [64],
        "num_layers": [2],
        #"lr": [1e-3],
        "lr": [1e-4,1e-5,2e-5],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "adapter_scale": [0.2],
        "diagnostics_dir": "",
        "diagnostics_dataset": data_name,
        "diagnostics_parameter": "",
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict

def config_OnlineDualNet_Simple_L2_Tune_single_lr(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "num_layers": [2],
        "lr": [1e-4,1e-5],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "adapter_scale": [0.2],
        "diagnostics_dir": "",
        "diagnostics_dataset": data_name,
        "diagnostics_parameter": "",
        "n_classes": 2,
        "use_cuda": False,
    }


    return n_runs, config_dict


def config_OnlineNetTau025(data_name):
    n_runs = 3 if "f1" in data_name or data_name in ["susy", "higgs"] else 5
    config_dict = {
        "hidden_dim": [64],
        "num_layers": [2],
        "lr": [1e-3],
        "optimizer": ["AdamW"],
        "n_classes": 2,
        "use_cuda": False,
        "logit_adjustment_tau": [0.25],
    }
    if data_name in ["wbc", "spambase"]:
        config_dict["lr"] = [2e-3]
    elif data_name in ["spam", "a8a"]:
        config_dict["lr"] = [5e-4]
    if data_name == "a8a":
        config_dict["num_layers"] = [1]
    return n_runs, config_dict


def config_OnlineDualNet_Simple_L4(data_name):
    n_runs, config_dict = _config_online_dualnet_simple_depth(data_name, 4)
    config_dict["num_layers"] = [4]
    return n_runs, config_dict


def config_OnlineDualNet_Simple_L8(data_name):
    n_runs, config_dict = _config_online_dualnet_simple_depth(data_name, 8)
    config_dict["num_layers"] = [8]
    return n_runs, config_dict


def _config_online_dualnet_late_interaction(data_name, slow_depth, fast_depth, interaction):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "slow_depth": [int(slow_depth)],
        "fast_depth": [int(fast_depth)],
        "interaction": [str(interaction)],
        "lr": [1e-3],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "adapter_scale": [0.2],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_LateGate_S2F1(data_name):
    n_runs, config_dict = _config_online_dualnet_late_interaction(data_name, 2, 1, "gate")
    overrides = {"slow_depth": [2], "fast_depth": [1], "interaction": ["gate"]}
    config_dict.update(overrides)
    return n_runs, config_dict


def config_LateFiLM_S2F1(data_name):
    n_runs, config_dict = _config_online_dualnet_late_interaction(data_name, 2, 1, "film")
    overrides = {"slow_depth": [2], "fast_depth": [1], "interaction": ["film"]}
    config_dict.update(overrides)
    return n_runs, config_dict


def config_LateFiLM_S4F1(data_name):
    n_runs, config_dict = _config_online_dualnet_late_interaction(data_name, 4, 1, "film")
    overrides = {"slow_depth": [4], "fast_depth": [1], "interaction": ["film"]}
    config_dict.update(overrides)
    return n_runs, config_dict


def config_LateFiLM_S8F1(data_name):
    n_runs, config_dict = _config_online_dualnet_late_interaction(data_name, 8, 1, "film")
    overrides = {"slow_depth": [8], "fast_depth": [1], "interaction": ["film"]}
    config_dict.update(overrides)
    return n_runs, config_dict


def config_DualNetZeroInitLearnableScaledGradientConflictAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "learnable_adapter_scale": [True],
        "init_adapter_scale": [0.5],
        "scale_mode": ["layerwise"],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetZeroInitTanhBoundedLearnableScaledGradientConflictAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "learnable_adapter_scale": [True],
        "init_adapter_scale": [0.5],
        "scale_mode": ["layerwise"],
        "learnable_scale_transform": ["tanh_centered"],
        "scale_center": [0.5],
        "scale_radius": [0.25],
        "scale_lr_multiplier": [1.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetZeroInitSlowScaleLearnableScaledGradientConflictAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "learnable_adapter_scale": [True],
        "init_adapter_scale": [0.5],
        "scale_mode": ["layerwise"],
        "learnable_scale_transform": ["sigmoid"],
        "scale_lr_multiplier": [0.1],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetZeroInitTanhBoundedSlowScaleLearnableScaledGradientConflictAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "learnable_adapter_scale": [True],
        "init_adapter_scale": [0.5],
        "scale_mode": ["layerwise"],
        "learnable_scale_transform": ["tanh_centered"],
        "scale_center": [0.5],
        "scale_radius": [0.25],
        "scale_lr_multiplier": [0.1],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetZeroInitHardBoundedDeltaGradientConflictAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "delta_clip": [0.5, 1.0, 2.0, 5.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetZeroInitSoftBoundedDeltaGradientConflictAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "delta_clip": [10.0, 20.0, 40.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetZeroInitScaledSoftBoundedDeltaGradientConflictAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "adapter_scale": [0.2],
        "delta_clip": [10.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_RGCDualNet(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "lr": [1e-3],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "adapter_scale": [0.2],
        "delta_clip": [10.0],
        "probe_views": [2],
        "probe_drop_prob": [0.2],
        "fast_loss_weight": [1.0],
        "probe_loss_weight": [0.5],
        "slow_loss_weight": [1.0],
        "transfer_weight": [0.2],
        "reliability_min": [0.05],
        "reliability_ema_gamma": [0.9],
        "grad_ema_gamma": [0.9],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_RGCDualNetV2(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "lr": [1e-3],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "adapter_scale": [0.2],
        "delta_clip": [10.0],
        "probe_views": [1],
        "probe_drop_prob": [0.2],
        "base_fast_loss_weight": [1.0],
        "fast_adaptation_alpha": [1.5],
        "slow_loss_weight": [1.0],
        "slow_consolidation_beta": [0.75],
        "probe_loss_weight": [0.25],
        "transfer_weight": [0.2],
        "reliability_min": [0.05],
        "reliability_ema_gamma": [0.9],
        "grad_ema_gamma": [0.9],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_RGCDualNetV3(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "lr": [1e-3],
        "optimizer": ["AdamW"],
        "slow_layers": [2],
        "slow_dropout": [0.0],
        "adapter_hidden": [32],
        "adapter_scale": [0.2],
        "delta_clip": [10.0],
        "probe_views": [1],
        "probe_drop_prob": [0.2],
        "base_fast_loss_weight": [1.0],
        "fast_adaptation_alpha": [1.5],
        "slow_loss_weight": [1.0],
        "slow_consolidation_beta": [0.75],
        "probe_loss_weight": [0.25],
        "transfer_weight": [0.2],
        "residual_penalty_weight": [0.01],
        "reliability_min": [0.05],
        "reliability_ema_gamma": [0.9],
        "grad_ema_gamma": [0.9],
        "slow_grad_clip": [1.0],
        "fast_grad_clip": [2.0],
        "head_grad_clip": [1.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_RGCDualNetV2a(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "lr": [1e-3],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "adapter_scale": [0.2],
        "delta_clip": [10.0],
        "probe_views": [1],
        "probe_drop_prob": [0.2],
        "base_fast_loss_weight": [1.0],
        "fast_adaptation_alpha": [1.5],
        "slow_loss_weight": [1.0],
        "slow_consolidation_beta": [0.75],
        "probe_loss_weight": [0.25],
        "transfer_weight": [0.2],
        "reliability_min": [0.05],
        "reliability_ema_gamma": [0.9],
        "grad_ema_gamma": [0.9],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_RGCDualNetV2aTune(data_name):
    n_runs, config_dict = config_RGCDualNetV2a(data_name)
    config_dict = dict(config_dict)
    config_dict.update({
        "probe_views": [3, 4],
        "method_label": "RGC-DualNet-v2a-Tune-probe_views",
    })
    return n_runs, config_dict


def config_DualNetResidualWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "residual_hidden": [32],
        "reliability_hidden": [16],
        "residual_scale": [1.0],
        "gate_min": [0.0],
        "gate_max": [1.0],
        "reliability_warmup": [5.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetResidualZeroInitScaledSoftBoundedWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "residual_hidden": [32],
        "reliability_hidden": [16],
        "zero_init_residual": [True],
        "residual_scale": [0.2],
        "delta_clip": [10.0],
        "gate_min": [0.0],
        "gate_max": [1.0],
        "reliability_warmup": [5.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetResidualConsolidationZeroInitScaledSoftBoundedWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "residual_hidden": [32],
        "reliability_hidden": [16],
        "zero_init_residual": [True],
        "residual_scale": [0.2],
        "delta_clip": [10.0],
        "consolidation_weight": [0.1],
        "consolidation_temperature": [2.0],
        "gate_min": [0.0],
        "gate_max": [1.0],
        "reliability_warmup": [5.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetResidualSelectiveConsolidationWeight1ZeroInitScaledSoftBoundedWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "residual_hidden": [32],
        "reliability_hidden": [16],
        "zero_init_residual": [True],
        "residual_scale": [0.2],
        "delta_clip": [10.0],
        "consolidation_weight": [1.0],
        "consolidation_temperature": [2.0],
        "selection_margin": [0.0],
        "gate_min": [0.0],
        "gate_max": [1.0],
        "reliability_warmup": [5.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetResidualAdvantageWeightedConsolidationZeroInitScaledSoftBoundedWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "residual_hidden": [32],
        "reliability_hidden": [16],
        "zero_init_residual": [True],
        "residual_scale": [0.2],
        "delta_clip": [10.0],
        "consolidation_weight": [1.0],
        "consolidation_temperature": [2.0],
        "advantage_temperature": [0.01],
        "advantage_center": [0.0],
        "gate_min": [0.0],
        "gate_max": [1.0],
        "reliability_warmup": [5.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetResidualUncertaintyGatedZeroInitScaledSoftBoundedWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "residual_hidden": [32],
        "reliability_hidden": [16],
        "zero_init_residual": [True],
        "residual_scale": [0.2],
        "delta_clip": [10.0],
        "uncertainty_center": [0.55],
        "uncertainty_temperature": [0.15],
        "uncertainty_floor": [0.85],
        "uncertainty_ceiling": [1.35],
        "evidence_floor": [0.8],
        "evidence_power": [0.5],
        "gate_min": [0.0],
        "gate_max": [1.0],
        "reliability_warmup": [5.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetResidualUncertaintyAttentiveFastZeroInitScaledSoftBoundedWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "residual_hidden": [32],
        "reliability_hidden": [16],
        "zero_init_residual": [True],
        "residual_scale": [0.2],
        "delta_clip": [10.0],
        "uncertainty_center": [0.55],
        "uncertainty_temperature": [0.15],
        "uncertainty_floor": [0.85],
        "uncertainty_ceiling": [1.35],
        "evidence_floor": [0.8],
        "evidence_power": [0.5],
        "attention_hidden": [16],
        "attention_temperature": [1.0],
        "attention_dropout": [0.0],
        "use_uncertainty_conditioned_attention": [True],
        "use_reliability_in_attention": [True],
        "gate_min": [0.0],
        "gate_max": [1.0],
        "reliability_warmup": [5.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetZeroInitScaledSoftBoundedDeltaGradientConflictAdaptiveFastGateNoWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "adapter_scale": [0.2],
        "delta_clip": [10.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetZeroInitScaledNoSoftBoundedDeltaGradientConflictAdaptiveFastGateNoWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "adapter_scale": [0.2],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetZeroInitSoftBoundedDeltaNoAdapterScaleGradientConflictAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "adapter_scale": [1.0],
        "delta_clip": [10.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetNoAdaptiveFastAdapterAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "adapter_scale": [0.2],
        "delta_clip": [10.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetSlowOnlyZeroInitScaledSoftBoundedDeltaGradientConflictAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "adapter_scale": [0.2],
        "delta_clip": [10.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetNoMaskFastGateZeroInitScaledSoftBoundedDeltaGradientConflictAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "adapter_scale": [0.2],
        "delta_clip": [10.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetZeroInitScaledSoftBoundedDeltaGradientConflictAdaptiveFastGateSSLWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.05],
        "distill_weight": [0.0],
        "beta": [0.1],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "adapter_scale": [0.2],
        "delta_clip": [10.0, 20.0, 40.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetZeroInitScaledSoftBoundedDeltaGradientConflictAdaptiveFastGateSSLSweepWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.005, 0.01, 0.02, 0.05],
        "distill_weight": [0.0],
        "beta": [0.02, 0.05, 0.1],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "adapter_scale": [0.2],
        "delta_clip": [20.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetZeroInitScaledSoftBoundedDeltaGradientConflictAdaptiveFastGateTemporalGatedSSLWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.02],
        "distill_weight": [0.0],
        "beta": [0.05],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "adapter_scale": [0.2],
        "delta_clip": [20.0],
        "temporal_tau": [0.7],
        "temporal_temperature": [0.1],
        "temporal_ema_momentum": [0.95],
        "ssl_ratio_target": [0.1],
        "warmup_steps": [5],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetPacketT1SlowAugZeroInitScaledSoftBoundedDeltaGradientConflictAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "adapter_scale": [0.2],
        "delta_clip": [20.0],
        "alpha_packet": [1.0],
        "packet_hidden_size": [128],
        "packet_d_k": [8],
        "packet_aggregator": ["max"],
        "use_var_embedding": [True],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetPacketT1SlowAugAlphaSweepZeroInitScaledSoftBoundedDeltaGradientConflictAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "adapter_scale": [0.2],
        "delta_clip": [20.0],
        "alpha_packet": [0.1, 0.25, 0.5],
        "packet_hidden_size": [128],
        "packet_d_k": [8],
        "packet_aggregator": ["max"],
        "use_var_embedding": [True],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetPacketT1SlowLNResidualZeroInitScaledSoftBoundedDeltaGradientConflictAdaptiveFastGateWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "adapter_scale": [0.2],
        "delta_clip": [20.0],
        "packet_hidden_size": [128],
        "packet_d_k": [8],
        "packet_aggregator": ["max"],
        "use_var_embedding": [True],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_DualNetAdaptiveFastGateNoReplayWelfordZScore(data_name):
    n_runs, config_dict = config_DualNetNoReplayWelfordZScore(data_name)
    return n_runs, _add_dualnet_adaptive_fast_gate_params(config_dict)


def config_DualNetAdaptiveFastGateNoReplayWelfordZScoreNoSSL(data_name):
    n_runs, config_dict = config_DualNetAdaptiveFastGateNoReplayWelfordZScore(data_name)
    config_dict = dict(config_dict)
    config_dict["ssl_weight"] = [0.0]
    config_dict["beta"] = [0.0]
    return n_runs, config_dict


def config_DualNetAdaptiveFastGatePrototypeMemoryWelfordZScore(data_name):
    n_runs, config_dict = config_DualNetPrototypeMemoryWelfordZScore(data_name)
    return n_runs, _add_dualnet_adaptive_fast_gate_params(config_dict)


def config_DualNetAdaptiveFastGateTrainStepWelfordZScore(data_name):
    n_runs, config_dict = config_DualNetWelfordZScore(data_name)
    return n_runs, _add_dualnet_adaptive_fast_gate_params(config_dict)


def config_DualNetReliabilityAdaptiveFastGateWelfordZScore(data_name):
    n_runs, config_dict = config_DualNetAdaptiveFastGateWelfordZScore(data_name)
    return n_runs, _add_dualnet_reliability_params(config_dict)


def config_DualNetLearnedReliabilityAdaptiveFastGateWelfordZScore(data_name):
    n_runs, config_dict = config_DualNetAdaptiveFastGateWelfordZScore(data_name)
    return n_runs, _add_dualnet_reliability_params(config_dict)


def config_DualNetReliabilityTrainStepAdaptiveFastGateWelfordZScore(data_name):
    n_runs, config_dict = config_DualNetAdaptiveFastGateTrainStepWelfordZScore(data_name)
    return n_runs, _add_dualnet_reliability_params(config_dict)


def config_FSNet(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "adapter_hidden": [32],
        "lr": [1e-3],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_FSNetFill0(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "adapter_hidden": [32],
        "lr": [1e-3],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_FSNetFill0WelfordZScore(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "adapter_hidden": [32],
        "lr": [1e-3],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_FSNetWelfordZScore(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "adapter_hidden": [32],
        "lr": [1e-3],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_FSNetEWMAScaler(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "adapter_hidden": [32],
        "lr": [1e-3],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "scaler_beta": [0.9],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_FSNetWelfordClipZScore(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "adapter_hidden": [32],
        "lr": [1e-3],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "clip_value": [3.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_CleanMLPFill0(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "adapter_hidden": [32],
        "lr": [1e-3],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_CleanMLPFill0WelfordZScore(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "adapter_hidden": [32],
        "lr": [1e-3],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_CleanMLPMaskInputWelfordZScore(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "adapter_hidden": [32],
        "lr": [1e-3],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_MLPZeroInitScaledSoftBoundedDeltaGradientConflictWelfordZScoreDiagnostics(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "projection_dim": [32],
        "lr": [1e-3],
        "memory_size": [0],
        "replay_batch_size": [0],
        "ssl_weight": [0.0],
        "distill_weight": [0.0],
        "beta": [0.0],
        "optimizer": ["AdamW"],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "zero_init_adapter": [True],
        "adapter_scale": [0.2],
        "delta_clip": [10.0],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_FSNetGradStats(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "adapter_hidden": [32],
        "lr": [1e-3],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_FSNetStableTrigger(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "adapter_hidden": [32],
        "lr": [1e-3],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_FSNetGradStatsWelfordZScore(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "adapter_hidden": [32],
        "lr": [1e-3],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_FSNetStableTriggerWelfordZScore(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "adapter_hidden": [32],
        "lr": [1e-3],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_FSNetMaskAware(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "adapter_hidden": [32],
        "lr": [1e-3],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "grad_moment_gamma": [0.8],
        "trigger_patience": [2],
        "mask_gamma": [0.9],
        "mask_fast_gamma": [0.3],
        "mask_shift_tau": [0.05],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_OneNet(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "lr_expert": [1e-3],
        "lr_weight": [1e-3],
        "lr_bias": [1e-3],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_OneNetWelfordZScore(data_name):
    return config_OneNet(data_name)


def config_OneNetFill0(data_name):
    return config_OneNet(data_name)


def config_OneNetFill0WelfordZScore(data_name):
    return config_OneNet(data_name)


def config_OneNetFSNetFill0WelfordZScore(data_name):
    n_runs = _default_neural_runs(data_name)
    config_dict = {
        "hidden_dim": [64],
        "adapter_hidden": [32],
        "memory_slots": [32],
        "gamma": [0.9],
        "fast_gamma": [0.3],
        "tau": [0.75],
        "lr_expert": [1e-3],
        "lr_weight": [1e-3],
        "lr_bias": [1e-3],
        "n_classes": 2,
        "use_cuda": False,
    }
    return n_runs, config_dict


def config_OneNetFSNetFixedWeightWelfordZScore(data_name):
    return config_OneNetFSNetFill0WelfordZScore(data_name)


def config_OneNetFSNetFixedWeightLazyFSNetWelfordZScore(data_name):
    n_runs, config_dict = config_OneNetFSNetFill0WelfordZScore(data_name)
    config_dict["fsnet_update_interval"] = [1, 2, 5, 10]
    return n_runs, config_dict


def config_OneNetFSNetNoBiasWelfordZScore(data_name):
    return config_OneNetFSNetFill0WelfordZScore(data_name)


def config_OneNetFixedWeightWelfordZScore(data_name):
    return config_OneNet(data_name)


def config_OneNetNoBiasWelfordZScore(data_name):
    return config_OneNet(data_name)


def config_MODL(data_name, merge_strategy="sum"):
    """
    MODL 及其融合消融实验参数配置。
    merge_strategy 选项 (对应 Table 4):
        - "sum"  : Score Sum (MODL官方最优形态)
        - "mul"  : Multiplication
        - "soft" : Greedy Weighing
        - "ens"  : Ensemble
        - "moe"  : Mix. of Experts
    """

    # 基于论文 Table 13 提取的通用架构与学习率参数
    # 结构: {lr, MLP层数, MLP宽度, SetLearner块数, 每块层数, SetLearner宽度}
    params_list = {
        'german': {'lr_variance':0.01,'lr': 0.01, 'mlp_layers': 3, 'mlp_width': 250, 'set_blocks': 6, 'set_layers': 3, 'set_width': 250},
        'svmguide3': {'lr_variance':0.01,'lr': 0.01, 'mlp_layers': 3, 'mlp_width': 250, 'set_blocks': 6, 'set_layers': 3,
                      'set_width': 250},
        'magic04': {'lr_variance':0.001,'lr': 0.001, 'mlp_layers': 3, 'mlp_width': 250, 'set_blocks': 6, 'set_layers': 3, 'set_width': 250},
        'a8a': {'lr_variance':0.001,'lr': 0.001, 'mlp_layers': 3, 'mlp_width': 250, 'set_blocks': 6, 'set_layers': 3, 'set_width': 250},
        'cifar10': {'lr_variance':0.001,'lr': 0.00005, 'mlp_layers': 3, 'mlp_width': 250, 'set_blocks': 6, 'set_layers': 3,
                    'set_width': 250},
        'imnist': {'lr_variance':0.001,'lr': 0.00005, 'mlp_layers': 3, 'mlp_width': 250, 'set_blocks': 6, 'set_layers': 3,
                   'set_width': 250},
        'susy': {'lr_variance':0.001,'lr': 0.001, 'mlp_layers': 3, 'mlp_width': 250, 'set_blocks': 6, 'set_layers': 3, 'set_width': 250},
        'higgs': {'lr_variance':0.001,'lr': 0.001, 'mlp_layers': 3, 'mlp_width': 250, 'set_blocks': 6, 'set_layers': 3, 'set_width': 250},
    }

    # 获取特定数据集的参数，如果数据集不在列表中，默认使用 german 的配置
    config_dict = params_list.get(data_name, params_list['german'])

    # 注入融合策略
    config_dict['merge'] = merge_strategy

    # 论文中明确指出：小型和中型数据集运行 20 次独立试验，大型数据集（HIGGS, SUSY）运行 5 次
    if "f1" in data_name:
        n_runs = 3
    else:
        if data_name in ['susy', 'higgs']:
            n_runs = 3
        else:
            n_runs = 5

    return n_runs, config_dict


def config_MODLWelfordZScore(data_name):
    _, config_dict = config_MODL(data_name)
    return 1, config_dict

# --------- NB3 ------------
def config_nb3(data_name):
    '''
        numTopFeats_percent = [.2, .4, .6, .8, 1]
    '''
    # config_dict = {}
    # config_dict["numTopFeats_percent"] = numTopFeats_percent

    params_list = {
        'wpbc':         {"numTopFeats_percent":[1]},
        'ionosphere':   {"numTopFeats_percent":[0.2]},
        'wdbc':         {"numTopFeats_percent":[0.6]},
        'australian':   {"numTopFeats_percent":[0.8]},
        'wbc':          {"numTopFeats_percent":[0.2]},
        'diabetes_f':   {"numTopFeats_percent":[1]},
        'german':       {"numTopFeats_percent":[1]},
        'ipd':          {"numTopFeats_percent":[0.6]},
        'svmguide3':    {"numTopFeats_percent":[0.2]},
        'krvskp':       {"numTopFeats_percent":[1]},
        'spambase':     {"numTopFeats_percent":[1]},
        'magic04':      {"numTopFeats_percent":[0.6]},
        'a8a':          {"numTopFeats_percent":[0.2]},
        'susy':         {"numTopFeats_percent":[1]},
        'higgs':        {"numTopFeats_percent":[0.2]},
        'diabetes_us':  {"numTopFeats_percent":[0.8]},
        'spamassasin':  {"numTopFeats_percent":[0.2]},
        'imdb':         {"numTopFeats_percent":[0.4]},
        'crowdsense_c3':{"numTopFeats_percent":[0.6]},
        'crowdsense_c5':{"numTopFeats_percent":[0.8]},
    }
    
    n_runs = 1 # NB3 is a deterministic model. So, everytime, it will produce same result for the same data. So, the num_runs is kept 1.
    config_dict = params_list[data_name]
    return n_runs, config_dict

# --------- HI2 ------------
def config_HI2(data_name):
    params_list = {
        'wpbc': {"spacing": [0.3], "lr": [2e-5]},
        'ionosphere': {"spacing": [0.3], "lr": [2e-5]},
        'wdbc': {"spacing": [0.3], "lr": [2e-5]},
        'australian': {"spacing": [0.3], "lr": [2e-5]},
        'wbc': {"spacing": [0.3], "lr": [2e-5]},
        'diabetes_f': {"spacing": [0.3], "lr": [2e-5]},
        'german': {"spacing": [0.3], "lr": [2e-5]},
        'ipd': {"spacing": [0.3], "lr": [2e-5]},
        'svmguide3': {"spacing": [0.3], "lr": [2e-5]},
        'krvskp': {"spacing": [0.3], "lr": [2e-5]},
        'spambase': {"spacing": [0.3], "lr": [2e-5]},
        'magic04': {"spacing": [0.3], "lr": [2e-5]},
        'a8a': {"spacing": [0.3], "lr": [2e-5]},
        'susy': {"spacing": [0.3], "lr": [2e-5]},
        'higgs': {"spacing": [0.3], "lr": [2e-5]},
        'diabetes_us': {"spacing": [0.3], "lr": [2e-5]},
        'spamassasin': {"spacing": [0.3], "lr": [2e-5]},
        'imdb': {"spacing": [0.3], "lr": [2e-5]},
        'crowdsense_c3': {"spacing": [0.3], "lr": [2e-5]},
        'crowdsense_c5': {"spacing": [0.3], "lr": [2e-5]},
        'elec': {"spacing": [0.3], "lr": [2e-5]},
        'weather': {"spacing": [0.3], "lr": [2e-5]},
        'spam': {"spacing": [0.3], "lr": [2e-5]},
        'phishing': {"spacing": [0.3], "lr": [2e-5]},
        'chessweka': {"spacing": [0.3], "lr": [2e-5]},
        'airlines': {"spacing": [0.3], "lr": [2e-5]},
        'LUdata': {"spacing": [0.3], "lr": [2e-5]},
        "default": {"spacing": [0.3], "lr": [2e-5]}
    }

    if "f1" in data_name:
        config_dict = params_list["default"]
    else:
        config_dict = params_list[data_name]

    if "f1" in data_name:
        n_runs = 3
    else:
        if data_name=="susy" or data_name=="higgs":
            n_runs = 3  # 神经网络随机性，运行多次取平均
        else:
            n_runs=5

    return n_runs, config_dict

# --------- FAE ------------
def config_fae(data_name):
    # Based on original paper
    config_dict = {}
    n_runs = 1 # FAE is a deterministic model. So, everytime, it will produce same result for the same data. So, the num_runs is kept 1.
    m = 5    # (maturity) Number of instances needed before a learner’s classifications are used by the ensemble
    p = 3    # (probation time) is the number of times in a row a learner is allowed to be under the threshold before being removed
    f = 0.15 # (feature change threshold) is the threshold placed on the amount of change between the
            # youngest learner’s set of features (yfs) and the top M features (mfs);
    r = 10   # (growth rate) is the number of instances between when the last learner was added and
            # when the ensemble’s accuracy is checked for the addition of a new learner
    N = 50   # Number of instances over which to compute an accuracy measure;
    params_list = {
        'wpbc':         {"numTopFeats_percent":[0.4]},
        'ionosphere':   {"numTopFeats_percent":[0.2]},
        'wdbc':         {"numTopFeats_percent":[0.2]},
        'australian':   {"numTopFeats_percent":[1]},
        'wbc':          {"numTopFeats_percent":[0.8]},
        'diabetes_f':   {"numTopFeats_percent":[1]},
        'german':       {"numTopFeats_percent":[0.2]},
        'ipd':          {"numTopFeats_percent":[0.8]},
        'svmguide3':    {"numTopFeats_percent":[0.4]},
        'krvskp':       {"numTopFeats_percent":[1]},
        'spambase':     {"numTopFeats_percent":[0.2]},
        'magic04':      {"numTopFeats_percent":[1]},
        'a8a':          {"numTopFeats_percent":[0.2]},
        'susy':         {"numTopFeats_percent":[0.6]},
        'higgs':        {"numTopFeats_percent":[0.4]},
        'diabetes_us':  {"numTopFeats_percent":[0.4]},
        'spamassasin':  {"numTopFeats_percent":[0.2]},
        'imdb':         {"numTopFeats_percent":[0.8]},
        'crowdsense_c3':{"numTopFeats_percent":[0.2]},
        'crowdsense_c5':{"numTopFeats_percent":[0.8]},
    }
    # M = [.2, .4, .6, .8, 1]  # Number of features (here in proportion) selected by the feature selection algorithm for a newly created learner
    # if data_name == "higgs":
        # M = [.2, .4, .6] # For .8 it takes 31 hrs and for 1 it takes 202 hrs
    # Store all the config parameters
    config_dict["m"] = m
    config_dict["p"] = p
    config_dict["f"] = f
    config_dict["r"] = r
    config_dict["N"] = N
    config_dict["M"] = params_list[data_name]["numTopFeats_percent"]

    return n_runs, config_dict

# --------- OLVF ------------
def config_olvf(data_name, num_feat):
    n_runs = 1 # All w is 0. So it is deterministic
    '''
    Hyperparameter here means:

        'B':
        'C':
        'C_bar':
        'reg':
        'n_feat0':
    '''
    params_list = {
        'wpbc':         {'B':[1],       'C':[1],           'C_bar':[1],        'reg':[0.0001]},
        'ionosphere':   {'B':[1],       'C':[1],           'C_bar':[1],        'reg':[0.0001]},
        'wdbc':         {'B':[1],       'C':[0.0001],      'C_bar':[0.0001],   'reg':[0.0001]},
        'australian':   {'B':[1],       'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'wbc':          {'B':[1],       'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'diabetes_f':   {'B':[0.3],     'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'german':       {'B':[0.01],    'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'ipd':          {'B':[1],       'C':[1],           'C_bar':[0.01],     'reg':[0.0001]},
        'svmguide3':    {'B':[1],       'C':[1],           'C_bar':[1],        'reg':[0.0001]},
        'krvskp':       {'B':[1],       'C':[1],           'C_bar':[1],        'reg':[0.0001]},
        'spambase':     {'B':[1],       'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'magic04':      {'B':[1],       'C':[0.0001],      'C_bar':[0.0001],   'reg':[0.0001]},
        'imdb':         {'B':[1],       'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'a8a':          {'B':[1],       'C':[1],           'C_bar':[0.0001],   'reg':[0.0001]},
        'crowdsense_c3':{'B':[1],       'C':[0.0001],      'C_bar':[0.0001],   'reg':[0.0001]},
        'crowdsense_c5':{'B':[1],       'C':[0.0001],      'C_bar':[0.0001],   'reg':[0.0001]},
        'susy':         {'B':[1],       'C':[0.01],        'C_bar':[0.01],     'reg':[0.0001]},
        'higgs':        {'B':[1],       'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'diabetes_us':  {'B':[1],       'C':[0.0001],      'C_bar':[0.0001],   'reg':[0.0001]},
        'spamassasin':  {'B':[1],       'C':[1],           'C_bar':[0.0001],   'reg':[0.0001]},
        'elec':         {'B':[1],       'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'weather':      {'B':[1],       'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'spam':         {'B':[1],       'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'phishing':     {'B':[1],       'C':[0.01],        'C_bar':[0.0001],   'reg':[0.0001]},
        'LUdata':       {'B':[1],       'C':[1],           'C_bar':[0.01],     'reg':[0.0001]},
        "default": {'B':[1],       'C':[0.0001],      'C_bar':[0.0001],   'reg':[0.0001]}
    }




    # data_list_hyper = ['wbc', 'svmguide3', 'wpbc', 'ionosphere', 'magic04', 'german',
    #                     'spambase', 'wdbc', 'a8a']
    # data_list_hyper = []
    # config_dict = {}
    # if data_name in data_list_hyper:
    #     config_dict = params_list[data_name]
    # else:
    #     config_dict['B'] = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    #     config_dict['C_bar'] = [0.0001, 0.01, 1]
    #     config_dict['C'] = [0.0001, 0.01, 1]
    # config_dict['reg'] = [0.0001, 0.01, 1]
    
    if "f1" in data_name:
        config_dict = params_list["default"]
    else:
        config_dict = params_list[data_name]

    config_dict['n_feat0'] = num_feat
    
    return n_runs, config_dict

# --------- OCDS ------------
def config_ocds(num_feat, data_name):
    config_dict = {}
    gamma = [np.round(150/num_feat, 3)]# It is based on the number of features. The rule is to keep U_t < 150.
    if gamma[0] > 1:
        gamma = [1] # gamma cannot be more than 1

    params_list = {
        'wpbc':         {'alpha': [0.0001],
                        'lamda': [0.0001],
                        'beta1': [1.5e-5],
                        'gamma': gamma,
                        'tau': [None]},
        'ionosphere':   {'alpha': [0.01],
                        'lamda': [0.0001],
                        'beta1': [5e-5],
                        'gamma': gamma,
                        'tau': [None]},
        'wdbc':         {'alpha': [0.0001],
                        'lamda': [0.0001],
                        'beta1': [1.5e-5],
                        'gamma': gamma,
                        'tau': [0.01]},
        'australian':   {'alpha': [0.01],
                        'lamda': [0.0001],
                        'beta1': [1.5e-5],
                        'gamma': gamma,
                        'tau': [0.01]},
        'wbc':          {'alpha': [1],
                        'lamda': [0.0001],
                        'beta1': [1.5e-5],
                        'gamma': gamma,
                        'tau': [0.1]},
        'diabetes_f':   {'alpha': [0.001],
                        'lamda': [0.0001],
                        'beta1': [1.5e-5],
                        'gamma': gamma,
                        'tau': [0.0001]},
        'crowdsense_c3':{'alpha': [0.0001],
                        'lamda': [0.0001],
                        'beta1': [1.5e-5],
                        'gamma': gamma,
                        'tau': [None]},
        'crowdsense_c5':{'alpha': [0.0001],
                        'lamda': [0.0001],
                        'beta1': [1.5e-5],
                        'gamma': gamma,
                        'tau': [None]},
        'german':       {'alpha': [0.001],
                        'lamda': [0.0001],
                        'beta1': [1.5e-5],
                        'gamma': gamma,
                        'tau': [0.0001]},
        'ipd':          { 'alpha': [0.0001],
                        'lamda': [0.0001],
                        'beta1': [1.5e-5],
                        'gamma': gamma,
                        'tau': [0.001]},
        'svmguide3':    {'alpha': [0.1],
                        'lamda': [0.001],
                        'beta1': [5e-4],
                        'gamma': gamma,
                        'tau': [0.01]},
        'krvskp':       {'alpha': [0.0001],
                        'lamda': [0.0001],
                        'beta1': [1.5e-5],
                        'gamma': gamma,
                        'tau': [0.1]},
        'spambase':     {'alpha': [0.001],
                        'lamda': [0.0001],
                        'beta1': [5e-5],
                        'gamma': gamma,
                        'tau': [1]},
        'spamassasin':  {'alpha': [0.1],
                        'lamda': [0.01],
                        'beta1': [5e-3],
                        'gamma': gamma,
                        'tau': [0.001]},
        'magic04':      {'alpha': [0.1],
                        'lamda': [0.1],
                        'beta1': [0.05],
                        'gamma': gamma,
                        'tau': [1]},
        'imdb':         {'alpha': [0.1],
                        'lamda': [0.01],
                        'beta1': [5e-3],
                        'gamma': gamma,
                        'tau': [0.001]},
        'a8a':          {'alpha': [0.1],
                        'lamda': [0.01],
                        'beta1': [5e-3],
                        'gamma': gamma,
                        'tau': [0.001]},
        'diabetes_us':  {'alpha': [0.1],
                        'lamda': [0.1],
                        'beta1': [1.5e-2],
                        'gamma': gamma,
                        'tau': [None]},
        'susy':         {'alpha': [1],
                        'lamda': [0.001],
                        'beta1': [8.5e-4],
                        'gamma': gamma,
                        'tau': [None]},
        'higgs':        {'alpha': [0.01],
                        'lamda': [0.001],
                        'beta1': [5e-4],
                        'gamma': gamma,
                        'tau': [0.001]},
    }
    config_dict = params_list[data_name]

    # if data_name in ["imdb", "susy", "higgs", "spamassasin"]:
    #     config_dict = params_list[data_name]
    # else:
    #     T = [8, 16] # Update after T instances
    #     # gamma = [0.01, 0.1, 1] 
    #     alpha = [0.0001, 0.01, 1] # According to original paper
    #     # beta0 = [0.0000001] # We introduced this to absrob the magnitude of the first expression in equation 8
    #     beta0 = [0.0001, 0.01, 1] # We introduced this to absrob the magnitude of the first expression in equation 8
    #     beta1 = [0.0001, 0.01, 1] # According to original paper
    #     beta2 = [0.0001, 0.01, 1] # According to original paper
    #     config_dict['T'] = T
    #     config_dict['gamma'] = gamma
    #     config_dict['alpha'] = alpha
    #     config_dict['beta0'] = beta0
    #     config_dict['beta1'] = beta1
    #     config_dict['beta2'] = beta2

    
    return config_dict

# --------- OLIFL ------------
def config_olifl(data_name):
    config_dict = {}

    params_list = {
        'wpbc':         {"C": [2e-6],   "option": [0]},
        'ionosphere':   {"C": [10],     "option": [0]},
        'wdbc':         {"C": [1],      "option": [1]},
        'australian':   {"C": [2e-1],   "option": [1]},
        'wbc':          {"C": [5e-5],   "option": [0]},
        'diabetes_f':   {"C": [5e-6],   "option": [1]},
        'crowdsense_c3':{"C": [1],      "option": [1]},
        'crowdsense_c5':{"C": [1],      "option": [1]},
        'german':       {"C": [1e-5],   "option": [0]},
        'ipd':          {"C": [5],      "option": [1]},
        'svmguide3':    {"C": [7],      "option": [0]},
        'krvskp':       {"C": [2e-2],   "option": [1]},
        'spambase':     {"C": [1e-5],   "option": [1]},
        'spamassasin':  {"C": [1e-2],   "option": [1]},
        'diabetes_us':  {"C": [1e-5],   "option": [1]},
        'magic04':      {"C": [1],      "option": [0]},
        'imdb':         {"C": [7e-3],   "option": [1]},
        'a8a':          {"C": [2],      "option": [1]},
        'elec':         {"C": [0.5],    "option": [1]},
        'weather':      {"C": [1e-2],   "option": [1]},
        'spam':         {"C": [1e-2],   "option": [1]},
        'phishing':     {"C": [0.1],    "option": [1]},
        'LUdata':       {"C": [0.05],   "option": [1]},
        'susy':         {"C": [0.3],    "option": [1]},
        'higgs':        {"C": [8e-2],   "option": [1]},
        "default": {"C": [1], "option": [1]}
    }

    # config_dict = params_dict[data_name]
    if "f1" in data_name:
        config_dict = params_list["default"]
    else:
        config_dict = params_list[data_name]

    return config_dict


# --------- OVFM ------------
def config_ovfm(data_name):
    config_dict = {}
    '''

    'decay_choice': Possible values - 0, 1, 2, 3, 4

    'contribute_error_rate': 

    'decay_coef_change': This is used to update decay coefficient (this_decay_coef). Set as False
    
    'batch_size_denominator': This is used to update decay coefficient (this_decay_coef). 
                              If 'decay_coef_change' is False, then the value of 'batch_size_denominator'
                              does not matter.
    
    Below the hyperparameters corresponding to different datasets are defined.

        Taken from Original Paper Code: ["ionosphere", "wdbc", "australian", 
                                        "diabetes_f", "german"]

        Heuristically Chosen: The following datasets requires significant time to run. Therefore,
                              we heuristically chose the hyperparameters based on the best performance
                              parameters of other similar size dataset.
                              ["imdb", "a8a", "susy", "higgs", "spamassasin", "diabetes_us"] 
        
        Hyperparameters Exhaustive Searching: Best hyperparameters for the following dataset were 
                              exhaustively search.
                              ["wpbc", "ipd", "svmguide3", "krvskp", "spambase", 
                               "magic04", "crowdsense_c3", "crowdsense_c5", ]
    '''
    params_dict = {
        "wpbc"          : {'decay_choice': [2], 'contribute_error_rate': [0.01],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "ionosphere"    : {'decay_choice': [4], 'contribute_error_rate': [0.02],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},                                     
        "wdbc"          : {'decay_choice': [0], 'contribute_error_rate': [0.02],
                            'decay_coef_change':[False] ,'batch_size_denominator': [8]}, 
        "australian"    : {'decay_choice': [4], 'contribute_error_rate': [0.01],
                            'decay_coef_change':[False] ,'batch_size_denominator': [10]},
        "wbc"           : {'decay_choice': [2], 'contribute_error_rate': [0.02],
                            'decay_coef_change':[False] ,'batch_size_denominator': [8]},
        "diabetes_f"    : {'decay_choice': [2], 'contribute_error_rate': [0.05],
                            'decay_coef_change':[False] ,'batch_size_denominator': [8]},
        "crowdsense_c3" : {'decay_choice': [4], 'contribute_error_rate': [0.01],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "crowdsense_c5" : {'decay_choice': [4], 'contribute_error_rate': [0.05],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},    
        "german"        : {'decay_choice': [3], 'contribute_error_rate': [0.005],
                            'decay_coef_change':[False] ,'batch_size_denominator': [8]},
        "ipd"           : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "svmguide3"     : {'decay_choice': [0], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "krvskp"        : {'decay_choice': [4], 'contribute_error_rate': [0.005],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "spambase"      : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "spamassasin"   : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "magic04"       : {'decay_choice': [3], 'contribute_error_rate': [0.005],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "imdb"          : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "a8a"           : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "diabetes_us"   : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "susy"          : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "higgs"         : {'decay_choice': [4], 'contribute_error_rate': [0.001],
                            'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "elec": {'decay_choice': [4], 'contribute_error_rate': [0.001],
                  'decay_coef_change': [False], 'batch_size_denominator': [20]},
        "weather": {'decay_choice': [4], 'contribute_error_rate': [0.001],
                 'decay_coef_change': [False], 'batch_size_denominator': [20]},
        # "default"       : {'decay_choice': [0, 1, 2, 3, 4], 'contribute_error_rate': [0.001, 0.005, 0.01, 0.02, 0.05],
        #                     'decay_coef_change':[False] ,'batch_size_denominator': [20]},
        "default": {'decay_choice': [4], 'contribute_error_rate': [0.001],
                    'decay_coef_change': [False], 'batch_size_denominator': [20]},
    }
    #config_dict = params_dict[data_name]
    if data_name not in ["ionosphere", "australian", "wbc", "diabetes_f", "german",
                         "imdb", "a8a", "susy", "higgs", "spamassasin", "diabetes_us"]:
        config_dict = params_dict["default"]
    else:
        config_dict = params_dict[data_name]

    if "f1" in data_name:
        n_runs = 3
    else:
        if data_name in ['susy', 'higgs']:
            n_runs = 3
        else:
            n_runs = 5
    
    return n_runs,config_dict


def config_OVFMWelfordZScore(data_name):
    return config_ovfm(data_name)
    
# --------- DynFo ------------
def config_dynfo(num_of_instances, data_name):
    ''' Dynfo takes a lot of time to run, because at each instance, the model undergoes many 
    relearning operations. To make sure, that the model does not undergo many relearning 
    operation, we need to set higher beta and theta1 values.
    '''
    config_dict = {}
    # Setting the value of N as 10% of the data or 20 instances. Whichever is less
    N = int(num_of_instances*.1)
    if N > 20:
        N = 20

    ''' Original paper provides the best hyperparameter for imdb dataset. We only change 
        values of beta and theta1 such that it is feasible to run the experiment.'''
    params_list = {
        "wpbc":        {"alpha": [0.5], "beta": [0.8], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "ionosphere":  {"alpha": [0.1], "beta": [0.8], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "wdbc":        {"alpha": [0.1], "beta": [0.8], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.8], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "australian":  {"alpha": [0.1], "beta": [0.8], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.8], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "wbc":         {"alpha": [0.1], "beta": [0.5], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "diabetes_f":  {"alpha": [0.5], "beta": [0.8], "delta": [0.001], "epsilon": [0.01],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "crowdsense_c3":{"alpha": [0.5], "beta": [0.5], "delta": [0.01], "epsilon": [0.01],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "crowdsense_c5":{"alpha": [0.5], "beta": [0.5], "delta": [0.01], "epsilon": [0.001],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "german":      {"alpha": [0.5], "beta": [0.5], "delta": [0.001], "epsilon": [0.01],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "ipd":         {"alpha": [0.1], "beta": [0.8], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "svmguide3":   {"alpha": [0.5], "beta": [0.5], "delta": [0.001], "epsilon": [0.01],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "krvskp":      {"alpha": [0.1], "beta": [0.8], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.5], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "spambase":    {"alpha": [0.1], "beta": [0.5], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.8], "M": [500], "N": N, "theta1": [0.05], "theta2": [0.75]},
        "spamassasin": {"alpha": [0.5], "beta": [0.5], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},
        "magic04":     {"alpha": [0.5], "beta": [0.5], "delta": [0.1], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},
        "imdb":        {"alpha": [0.5], "beta": [0.8], "delta": [0.001], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},
        "a8a":         {"alpha": [0.5], "beta": [0.5], "delta": [0.03], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},
        "diabetes_us": {"alpha": [0.5], "beta": [0.5], "delta": [0.1], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},     
        "susy":        {"alpha": [0.5], "beta": [0.5], "delta": [0.4], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},
        "higgs":       {"alpha": [0.5], "beta": [0.5], "delta": [0.2], "epsilon": [0.001],
                        "gamma": [0.7], "M": [1000], "N": N, "theta1": [0.05], "theta2": [0.6]},
                   
    }
    config_dict = params_list[data_name]

    # if data_name in ["imdb", "spamassasin", "diabetes_us", "magic04", "a8a", "susy", "higgs"]:
    #     config_dict = params_list[data_name]
    # else:
    #     alpha = [0.1, 0.5] # Lower Alpha value is good
    #     beta = [0.5, 0.8] # a lower value for dense data streams and a higher value for sparser data streams. 
    #     delta = [0.001, 0.01] # This should be small
    #     epsilon = [0.001, 0.01]  # Weak learners weights are decreased by epsilon value
    #     gamma = [0.5, 0.8]
    #     M = [500]
    #     theta1=[0.05] # This is set for each dataset as done in the original paper
    #     theta2=[0.75] # This is set for each dataset as done in the original paper

    #     # Store all the config parameters
    #     config_dict["alpha"] = alpha
    #     config_dict["beta"] = beta
    #     config_dict["delta"] = delta
    #     config_dict["epsilon"] = epsilon
    #     config_dict["gamma"] = gamma
    #     config_dict["M"] = M
    #     config_dict["N"] = N
    #     config_dict["theta1"] = theta1
    #     config_dict["theta2"] = theta2

    return config_dict

# --------- ORF3V ------------
def config_orf3v(data_name):
    # config_dict = {}
    # forestSize = [3, 5, 10] # Number of Stumps for every feature
    # replacementInterval = [5, 10] # Instances after which to replace stumps
    # replacementChance = [0.7] # If replacement strategy is random, then this is the probability not to replace each stump
    # windowSize = [20] # Buffer - stores instances on which to determine feature statistics
    # updateStrategy = ["oldest", "random"] # replacement strategy: "oldest", "random"
    # alpha = [0.01, 0.1, 0.3, 0.5, 0.9] # weight update parameter
    # delta = [0.001] # caculates hb, which is used for pruning.
    # if data_name in ["susy", "higgs", "imdb"]: # heuristically set
    #     forestSize = [5]
    #     replacementInterval = [10]
    #     updateStrategy = ["random"] # replacement strategy: "oldest", "random"
    #     alpha = [0.1] # weight update parameter
    # config_dict["forestSize"] = forestSize
    # config_dict["replacementInterval"] = replacementInterval
    # config_dict["replacementChance"] = replacementChance
    # config_dict["windowSize"] = windowSize
    # config_dict["updateStrategy"] = updateStrategy
    # config_dict["alpha"] = alpha
    # config_dict["delta"] = delta
    '''
     原始"windowSize": [20]
    '''
    params_list = {
        "wpbc":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "ionosphere":  {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "wdbc":        {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "australian":  {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "wbc":         {"forestSize": [5], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "diabetes_f":  {"forestSize": [3], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['oldest'], "alpha": [0.3], "delta": [0.001]},
        "crowdsense_c3":{"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "crowdsense_c5":{"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "german":      {"forestSize": [5], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['oldest'], "alpha": [0.3], "delta": [0.001]},
        "ipd":         {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "svmguide3":   {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['random'], "alpha": [0.3], "delta": [0.001]},
        "krvskp":      {"forestSize": [5], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "spambase":    {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['oldest'], "alpha": [0.1], "delta": [0.001]},
        "spamassasin": {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "magic04":     {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['random'], "alpha": [0.01], "delta": [0.001]},
        "imdb":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "a8a":         {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['oldest'], "alpha": [0.1], "delta": [0.001]},
        "diabetes_us": {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "elec":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [5], "updateStrategy": ['random'], "alpha": [0.01], "delta": [0.001]},
        "weather":     {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [5], "updateStrategy": ['random'], "alpha": [0.01], "delta": [0.001]},
        "spam":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [5], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "phishing":    {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [5], "updateStrategy": ['oldest'], "alpha": [0.1], "delta": [0.001]},
        "LUdata":      {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [5], "updateStrategy": ['random'], "alpha": [0.3], "delta": [0.001]},
        "susy":        {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "higgs":       {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7], 
                        "windowSize": [5], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "default":       {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [5], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
    }
    # config_dict = params_dict[data_name]
    if "f1" in data_name:
        config_dict = params_list["default"]
    else:
        config_dict = params_list[data_name]
    if "f1" in data_name:
        n_runs = 3
    else:
        if data_name in ['susy', 'higgs']:
            n_runs = 3
        else:
            n_runs = 5

    return n_runs,config_dict

def config_orf3v_20(data_name):
    # config_dict = {}
    # forestSize = [3, 5, 10] # Number of Stumps for every feature
    # replacementInterval = [5, 10] # Instances after which to replace stumps
    # replacementChance = [0.7] # If replacement strategy is random, then this is the probability not to replace each stump
    # windowSize = [20] # Buffer - stores instances on which to determine feature statistics
    # updateStrategy = ["oldest", "random"] # replacement strategy: "oldest", "random"
    # alpha = [0.01, 0.1, 0.3, 0.5, 0.9] # weight update parameter
    # delta = [0.001] # caculates hb, which is used for pruning.
    # if data_name in ["susy", "higgs", "imdb"]: # heuristically set
    #     forestSize = [5]
    #     replacementInterval = [10]
    #     updateStrategy = ["random"] # replacement strategy: "oldest", "random"
    #     alpha = [0.1] # weight update parameter
    # config_dict["forestSize"] = forestSize
    # config_dict["replacementInterval"] = replacementInterval
    # config_dict["replacementChance"] = replacementChance
    # config_dict["windowSize"] = windowSize
    # config_dict["updateStrategy"] = updateStrategy
    # config_dict["alpha"] = alpha
    # config_dict["delta"] = delta
    '''
     原始"windowSize": [20]
    '''
    params_list = {
        "wpbc":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "ionosphere":  {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "wdbc":        {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "australian":  {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "wbc":         {"forestSize": [5], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "diabetes_f":  {"forestSize": [3], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.3], "delta": [0.001]},
        "crowdsense_c3":{"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "crowdsense_c5":{"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "german":      {"forestSize": [5], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.3], "delta": [0.001]},
        "ipd":         {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "svmguide3":   {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.3], "delta": [0.001]},
        "krvskp":      {"forestSize": [5], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "spambase":    {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.1], "delta": [0.001]},
        "spamassasin": {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "magic04":     {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.01], "delta": [0.001]},
        "imdb":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "a8a":         {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.1], "delta": [0.001]},
        "diabetes_us": {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "elec":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.01], "delta": [0.001]},
        "weather":     {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.01], "delta": [0.001]},
        "spam":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "phishing":    {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['oldest'], "alpha": [0.1], "delta": [0.001]},
        "LUdata":      {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.3], "delta": [0.001]},
        "susy":        {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "higgs":       {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "default":       {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [20], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
    }
    # config_dict = params_dict[data_name]
    if "f1" in data_name:
        config_dict = params_list["default"]
    else:
        config_dict = params_list[data_name]
    if "f1" in data_name:
        n_runs = 3
    else:
        if data_name in ['susy', 'higgs']:
            n_runs = 3
        else:
            n_runs = 5

    return n_runs,config_dict

def config_orf3v_50(data_name):
    # config_dict = {}
    # forestSize = [3, 5, 10] # Number of Stumps for every feature
    # replacementInterval = [5, 10] # Instances after which to replace stumps
    # replacementChance = [0.7] # If replacement strategy is random, then this is the probability not to replace each stump
    # windowSize = [20] # Buffer - stores instances on which to determine feature statistics
    # updateStrategy = ["oldest", "random"] # replacement strategy: "oldest", "random"
    # alpha = [0.01, 0.1, 0.3, 0.5, 0.9] # weight update parameter
    # delta = [0.001] # caculates hb, which is used for pruning.
    # if data_name in ["susy", "higgs", "imdb"]: # heuristically set
    #     forestSize = [5]
    #     replacementInterval = [10]
    #     updateStrategy = ["random"] # replacement strategy: "oldest", "random"
    #     alpha = [0.1] # weight update parameter
    # config_dict["forestSize"] = forestSize
    # config_dict["replacementInterval"] = replacementInterval
    # config_dict["replacementChance"] = replacementChance
    # config_dict["windowSize"] = windowSize
    # config_dict["updateStrategy"] = updateStrategy
    # config_dict["alpha"] = alpha
    # config_dict["delta"] = delta
    '''
     原始"windowSize": [20]
    '''
    params_list = {
        "wpbc":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "ionosphere":  {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "wdbc":        {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "australian":  {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "wbc":         {"forestSize": [5], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "diabetes_f":  {"forestSize": [3], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['oldest'], "alpha": [0.3], "delta": [0.001]},
        "crowdsense_c3":{"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "crowdsense_c5":{"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "german":      {"forestSize": [5], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['oldest'], "alpha": [0.3], "delta": [0.001]},
        "ipd":         {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "svmguide3":   {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['random'], "alpha": [0.3], "delta": [0.001]},
        "krvskp":      {"forestSize": [5], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "spambase":    {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['oldest'], "alpha": [0.1], "delta": [0.001]},
        "spamassasin": {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "magic04":     {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['random'], "alpha": [0.01], "delta": [0.001]},
        "imdb":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "a8a":         {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['oldest'], "alpha": [0.1], "delta": [0.001]},
        "diabetes_us": {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "elec":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['random'], "alpha": [0.01], "delta": [0.001]},
        "weather":     {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['random'], "alpha": [0.01], "delta": [0.001]},
        "spam":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "phishing":    {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['oldest'], "alpha": [0.1], "delta": [0.001]},
        "LUdata":      {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['random'], "alpha": [0.3], "delta": [0.001]},
        "susy":        {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "higgs":       {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "default":       {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [50], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
    }
    # config_dict = params_dict[data_name]
    if "f1" in data_name:
        config_dict = params_list["default"]
    else:
        config_dict = params_list[data_name]
    if "f1" in data_name:
        n_runs = 3
    else:
        if data_name in ['susy', 'higgs']:
            n_runs = 3
        else:
            n_runs = 5

    return n_runs,config_dict

def config_orf3v_100(data_name):
    # config_dict = {}
    # forestSize = [3, 5, 10] # Number of Stumps for every feature
    # replacementInterval = [5, 10] # Instances after which to replace stumps
    # replacementChance = [0.7] # If replacement strategy is random, then this is the probability not to replace each stump
    # windowSize = [20] # Buffer - stores instances on which to determine feature statistics
    # updateStrategy = ["oldest", "random"] # replacement strategy: "oldest", "random"
    # alpha = [0.01, 0.1, 0.3, 0.5, 0.9] # weight update parameter
    # delta = [0.001] # caculates hb, which is used for pruning.
    # if data_name in ["susy", "higgs", "imdb"]: # heuristically set
    #     forestSize = [5]
    #     replacementInterval = [10]
    #     updateStrategy = ["random"] # replacement strategy: "oldest", "random"
    #     alpha = [0.1] # weight update parameter
    # config_dict["forestSize"] = forestSize
    # config_dict["replacementInterval"] = replacementInterval
    # config_dict["replacementChance"] = replacementChance
    # config_dict["windowSize"] = windowSize
    # config_dict["updateStrategy"] = updateStrategy
    # config_dict["alpha"] = alpha
    # config_dict["delta"] = delta
    '''
     原始"windowSize": [20]
    '''
    params_list = {
        "wpbc":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "ionosphere":  {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "wdbc":        {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "australian":  {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "wbc":         {"forestSize": [5], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "diabetes_f":  {"forestSize": [3], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['oldest'], "alpha": [0.3], "delta": [0.001]},
        "crowdsense_c3":{"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "crowdsense_c5":{"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "german":      {"forestSize": [5], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['oldest'], "alpha": [0.3], "delta": [0.001]},
        "ipd":         {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "svmguide3":   {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['random'], "alpha": [0.3], "delta": [0.001]},
        "krvskp":      {"forestSize": [5], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "spambase":    {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['oldest'], "alpha": [0.1], "delta": [0.001]},
        "spamassasin": {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "magic04":     {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['random'], "alpha": [0.01], "delta": [0.001]},
        "imdb":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "a8a":         {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['oldest'], "alpha": [0.1], "delta": [0.001]},
        "diabetes_us": {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "elec":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['random'], "alpha": [0.01], "delta": [0.001]},
        "weather":     {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['random'], "alpha": [0.01], "delta": [0.001]},
        "spam":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "phishing":    {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['oldest'], "alpha": [0.1], "delta": [0.001]},
        "LUdata":      {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['random'], "alpha": [0.3], "delta": [0.001]},
        "susy":        {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "higgs":       {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "default":       {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [100], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
    }
    # config_dict = params_dict[data_name]
    if "f1" in data_name:
        config_dict = params_list["default"]
    else:
        config_dict = params_list[data_name]
    if "f1" in data_name:
        n_runs = 3
    else:
        if data_name in ['susy', 'higgs']:
            n_runs = 3
        else:
            n_runs = 5

    return n_runs,config_dict

def config_orf3v_200(data_name):
    # config_dict = {}
    # forestSize = [3, 5, 10] # Number of Stumps for every feature
    # replacementInterval = [5, 10] # Instances after which to replace stumps
    # replacementChance = [0.7] # If replacement strategy is random, then this is the probability not to replace each stump
    # windowSize = [20] # Buffer - stores instances on which to determine feature statistics
    # updateStrategy = ["oldest", "random"] # replacement strategy: "oldest", "random"
    # alpha = [0.01, 0.1, 0.3, 0.5, 0.9] # weight update parameter
    # delta = [0.001] # caculates hb, which is used for pruning.
    # if data_name in ["susy", "higgs", "imdb"]: # heuristically set
    #     forestSize = [5]
    #     replacementInterval = [10]
    #     updateStrategy = ["random"] # replacement strategy: "oldest", "random"
    #     alpha = [0.1] # weight update parameter
    # config_dict["forestSize"] = forestSize
    # config_dict["replacementInterval"] = replacementInterval
    # config_dict["replacementChance"] = replacementChance
    # config_dict["windowSize"] = windowSize
    # config_dict["updateStrategy"] = updateStrategy
    # config_dict["alpha"] = alpha
    # config_dict["delta"] = delta
    '''
     原始"windowSize": [20]
    '''
    params_list = {
        "wpbc":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "ionosphere":  {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "wdbc":        {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "australian":  {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "wbc":         {"forestSize": [5], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['oldest'], "alpha": [0.9], "delta": [0.001]},
        "diabetes_f":  {"forestSize": [3], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['oldest'], "alpha": [0.3], "delta": [0.001]},
        "crowdsense_c3":{"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "crowdsense_c5":{"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "german":      {"forestSize": [5], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['oldest'], "alpha": [0.3], "delta": [0.001]},
        "ipd":         {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "svmguide3":   {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['random'], "alpha": [0.3], "delta": [0.001]},
        "krvskp":      {"forestSize": [5], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "spambase":    {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['oldest'], "alpha": [0.1], "delta": [0.001]},
        "spamassasin": {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "magic04":     {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['random'], "alpha": [0.01], "delta": [0.001]},
        "imdb":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "a8a":         {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['oldest'], "alpha": [0.1], "delta": [0.001]},
        "diabetes_us": {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "elec":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['random'], "alpha": [0.01], "delta": [0.001]},
        "weather":     {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['random'], "alpha": [0.01], "delta": [0.001]},
        "spam":        {"forestSize": [10], "replacementInterval": [5], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['oldest'], "alpha": [0.01], "delta": [0.001]},
        "phishing":    {"forestSize": [10], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['oldest'], "alpha": [0.1], "delta": [0.001]},
        "LUdata":      {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['random'], "alpha": [0.3], "delta": [0.001]},
        "susy":        {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "higgs":       {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
        "default":       {"forestSize": [5], "replacementInterval": [10], "replacementChance": [0.7],
                        "windowSize": [200], "updateStrategy": ['random'], "alpha": [0.1], "delta": [0.001]},
    }
    # config_dict = params_dict[data_name]
    if "f1" in data_name:
        config_dict = params_list["default"]
    else:
        config_dict = params_list[data_name]
    if "f1" in data_name:
        n_runs = 3
    else:
        if data_name in ['susy', 'higgs']:
            n_runs = 3
        else:
            n_runs = 5

    return n_runs,config_dict

# --------- Aux-Net ------------
def config_auxnet(data_name):
    config_dict = {}

    params_list = {
        "wpbc":        {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.001]},
        "ionosphere":  {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.001]},
        "wdbc":        {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.01]},
        "australian":  {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.01]},
        "wbc":         {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.001]},
        "diabetes_f":  {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.05]},
        "crowdsense_c3":{"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.001]},
        "crowdsense_c5":{"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.001]},
        "german":      {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.001]},
        "ipd":         {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.001]},
        "svmguide3":   {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.1]},
        "krvskp":      {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.001]},
        "spambase":    {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.005]},
        "spamassasin": {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.01]},
        "magic04":     {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [.5]},
        "imdb":        {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.01]},
        "a8a":         {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.01]},
        "diabetes_us": {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.05]},
        "susy":        {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.05]},
        "higgs":       {"no_of_base_layers": [5], "no_of_end_layers": [5], "nodes_in_each_layer": [50], 
                        "b": [0.99], "s": [0.2], "lr": [0.05]},
    }
    
    config_dict = params_list[data_name]

    return config_dict


# --------- Aux-Drop ------------
def config_auxdrop(if_auxdrop_no_assumption_arch_change, X, data_name,
                    if_imputation, if_dummy_feat, n_dummy_feat, X_haphazard,
                    mask, imputation_type, dummy_type):
    config_dict = {}
    # max_num_hidden_layers - Number of hidden layers
    # qtd_neuron_per_hidden_layer - Number of nodes in each hidden layer except the AuxLayer
    # n_classes - The total number of classes (output labels)
    # n_neuron_aux_layer - The total numebr of neurons in the AuxLayer
    # batch_size - The batch size is always 1 since it is based on stochastic gradient descent
    # b - discount rate
    # n - learning rate
    # s - smoothing rate
    # dropout_p - The dropout rate in the AuxLayer
    # n_aux_feat - Number of auxiliary features
    # aux_feat_prob - The probability of each auxiliary feature being available at each point in time
    
    # We are choosing approximately 5 times the number of features. Note that this is only for faster computation. We can also add as and when new features comes in.
    n_neuron_aux_layer_dict = {"australian": 100, "wbc": 100, "diabetes_f": 100, "german": 100,
                              "ipd": 100, "svmguide3": 100, "magic04": 100, "susy": 100, "higgs": 100,
                              "wpbc": 200, "ionosphere": 200, "wdbc": 200, "krvskp": 200, "diabetes_us": 200,  
                              "spambase": 300, "a8a": 500,
                              "crowdsense_c3": 5000, "crowdsense_c5": 5000,
                              "spamassasin": 30000, "imdb": 30000,
                               "elec":100,"weather":100,"spam":3000,"phishing":200,"chessweka":100,"airlines":100,"LUdata":200,"default":100
                               }
    # n_dict = {"wpbc": [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5], 
    # }

    max_num_hidden_layers = [6] # Number of hidden layers
    qtd_neuron_per_hidden_layer = [50] # Number of nodes in each hidden layer except the AuxLayer
    n_classes = 2 # The total number of classes (output labels)
    batch_size = 1 # The batch size is always 1 since it is based on stochastic gradient descent
    b = [0.99] # discount rate
    s = [0.2] # smoothing parameter

    # n = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5] # learning rate
    n = {"wpbc": [0.01], "ionosphere": [0.5], "wdbc": [0.01], "australian": [0.005],
         "wbc": [0.1], "diabetes_f": [0.001], "german": [0.001], "ipd": [0.3],
         "svmguide3": [0.001], "krvskp": [0.1], "spambase": [0.005], "magic04": [0.01],
         "a8a": [0.01], "susy": [0.05], "higgs": [0.05], 
         "imdb": [0.01], "crowdsense_c3": [0.001], "crowdsense_c5": [0.001],
         "diabetes_us": [0.05], "spamassasin": [0.01],
         "elec": [0.01], "weather": [0.01], "spam": [0.01], "phishing": [0.01], "chessweka": [0.01], "airlines": [0.01], "LUdata": [0.01],"default":[0.01]
         }
    
    # dropout_p = [0.3, 0.5] # The dropout rate in the AuxLayer
    dropout_p = {"wpbc": [0.5], "ionosphere": [0.3], "wdbc": [0.5], "australian": [0.5],
         "wbc": [0.3], "diabetes_f": [0.5], "german": [0.5], "ipd": [0.3],
         "svmguide3": [0.3], "krvskp": [0.3], "spambase": [0.5], "magic04": [0.3],
         "a8a": [0.3], "susy": [0.3], "higgs": [0.3], 
         "imdb": [0.3], "crowdsense_c3": [0.5], "crowdsense_c5": [0.3],
         "diabetes_us": [0.3], "spamassasin": [0.3],
         "elec": [0.3], "weather": [0.3], "spam": [0.3], "phishing": [0.3], "chessweka": [0.3], "airlines": [0.3],"LUdata": [0.3],"default":[0.3]
                 }



    n_aux_feat = X.shape[1] # Number of auxiliary features
    if "f1" in data_name:
        n_neuron_aux_layer = n_neuron_aux_layer_dict["default"]
    else:
        n_neuron_aux_layer = n_neuron_aux_layer_dict[data_name] # The total numebr of neurons in the AuxLayer


    use_cuda = False
    config_dict["max_num_hidden_layers"] = max_num_hidden_layers
    config_dict["qtd_neuron_per_hidden_layer"] = qtd_neuron_per_hidden_layer
    config_dict["n_classes"] = n_classes
    config_dict["n_neuron_aux_layer"] = n_neuron_aux_layer
    config_dict["batch_size"] = batch_size
    config_dict["b"] = b
    config_dict["s"] = s

    if "f1" in data_name:
        config_dict["n"] = n["default"]
    else:
        config_dict["n"] = n[data_name]



    if "f1" in data_name:
        config_dict["dropout_p"] = dropout_p["default"]
    else:
        config_dict["dropout_p"] = dropout_p[data_name]

    config_dict["n_aux_feat"] = n_aux_feat
    config_dict["use_cuda"] = use_cuda

    if "f1" in data_name:
        n_runs = 3
    else:
        if data_name in ['susy', 'higgs']:
            n_runs = 3
        else:
            n_runs = 5

    if if_auxdrop_no_assumption_arch_change:
        return n_runs,config_dict
    else:
        # features_size - Number of base features
        # aux_layer - The position of auxiliary layer. This code does not work if the AuxLayer position is 1. 
        features_size =  2 # number of base features
        aux_layer = [3] # The position of auxiliary layer. This code does not work if the AuxLayer position is 1.
        if if_imputation:
            n_aux_feat = X.shape[1] - features_size # We impute some features (feature_size) to create base features. 
                # Therefore number of base features would be total number of features - total number of base features
            # Create dataset
            X_base = impute_data(X_haphazard[:, :features_size],
                                mask[:, :features_size], imputation_type)
            X_aux_new = X_haphazard[:, features_size:]
            aux_mask = mask[:, features_size:]
        elif if_dummy_feat:
            features_size = n_dummy_feat # We create dummy feature as base feature
            # Create dataset
            X_base = dummy_feat(X_haphazard.shape[0], features_size, dummy_type)
            X_aux_new = X_haphazard
            aux_mask = mask
        config_dict["features_size"] = features_size
        config_dict["aux_layer"] = aux_layer
        config_dict["n_aux_feat"] = n_aux_feat

        if "f1" in data_name:
            n_runs = 3
        else:
            if data_name in ['susy', 'higgs']:
                n_runs = 3
            else:
                n_runs = 5

        return n_runs,config_dict, X_base, X_aux_new, aux_mask


def config_auxdropWelfordZScore(if_auxdrop_no_assumption_arch_change, X, data_name,
                                if_imputation, if_dummy_feat, n_dummy_feat, X_haphazard,
                                mask, imputation_type, dummy_type):
    return config_auxdrop(
        if_auxdrop_no_assumption_arch_change,
        X,
        data_name,
        if_imputation,
        if_dummy_feat,
        n_dummy_feat,
        X_haphazard,
        mask,
        imputation_type,
        dummy_type,
    )

def config_Qwen_LoRA(data_name):
    config_dict = {}
    params_list = {
        "default": {"lr": [1e-4,5e-5], "use_scaler": [True,False]}
    }

    # 获取对应数据集配置，如果没有则使用 default
    config_dict = params_list.get(data_name, params_list["default"])
    n_runs = 3  # 神经网络具有随机性，通常需要多次运行消除误差
    return n_runs, config_dict
# --------- BERT Online ------------
def config_NLPbertonline(data_name):
    config_dict = {}
    params_list = {
        "default": {"lr": [1e-3],"model_type":["bert"],"mode":["input_pairs"], "use_scaler": [True]}#"model_type":["distilbert"],"mode":["values","input_pairs"]
    }

    # 获取对应数据集配置，如果没有则使用 default
    config_dict = params_list.get(data_name, params_list["default"])
    n_runs = 3  # 神经网络具有随机性，通常需要多次运行消除误差
    return n_runs, config_dict

def config_bertonline(data_name):
    config_dict = {}
    params_list = {
        "default": {"lr": [1e-3], "use_scaler": False}
    }

    # 获取对应数据集配置，如果没有则使用 default
    config_dict = params_list.get(data_name, params_list["default"])
    n_runs = 3  # 神经网络具有随机性，通常需要多次运行消除误差
    return n_runs, config_dict

# --------- MLP Online ------------
def config_mlponline(data_name):
    config_dict = {}
    params_list = {
        "wpbc": {"lr": [1e-3, 1e-4], "use_scaler": True, "hidden_size": 64},
        "default": {"lr": [1e-3], "use_scaler": True, "hidden_size": 64}
    }
    config_dict = params_list.get(data_name, params_list["default"])
    n_runs = 5
    return n_runs, config_dict

# --------- resnet Online ------------
def config_resnetonline(data_name):
    config_dict = {}
    params_list = {
        "wpbc": {"lr": [1e-3, 1e-4], "use_scaler": True, "hidden_size": 256,"num_layers":4},
        "default": {"lr": [1e-3], "use_scaler": True, "hidden_size": 256,"num_layers":4}
    }
    config_dict = params_list.get(data_name, params_list["default"])
    n_runs = 5
    return n_runs, config_dict

# ----------- HBP -----------------
def config_HBP(data_name):
    config_dict = {}

    max_num_hidden_layers = [12]  # Number of hidden layers
    qtd_neuron_per_hidden_layer = [100]  # Number of nodes in each hidden layer except the AuxLayer
    n_classes = 2  # The total number of classes (output labels)
    batch_size = 1  # The batch size is always 1 since it is based on stochastic gradient descent
    b = [0.99]  # discount rate
    s = [0.2]  # smoothing parameter

    n = {"default": [0.01],"f1_default":[0.00001]}

    use_cuda = False
    config_dict["max_num_hidden_layers"] = max_num_hidden_layers
    config_dict["qtd_neuron_per_hidden_layer"] = qtd_neuron_per_hidden_layer
    config_dict["n_classes"] = n_classes
    config_dict["batch_size"] = batch_size
    config_dict["b"] = b
    config_dict["s"] = s
    config_dict["n"] = n["default"]

    # config_dict = params_dict[data_name]
    if "f1" in data_name:
        config_dict["n"] = n["f1_default"]
    else:
        config_dict["n"] = n["default"]

    config_dict["use_cuda"] = use_cuda

    if "f1" in data_name:
        n_runs = 3
    else:
        if data_name in ['susy', 'higgs']:
            n_runs = 3
        else:
            n_runs = 5

    return n_runs,config_dict


def config_HBP_WelfordZScore(data_name):
    return config_HBP(data_name)


def config_HBPWelfordZScore(data_name):
    return config_HBP_WelfordZScore(data_name)


def config_Transformer(data_name):
    config_dict = {}
    params_list = {
        "wpbc": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "ionosphere": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "wdbc": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "australian": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "wbc": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "diabetes_f": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "crowdsense_c3": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "crowdsense_c5": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "german": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "ipd": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "svmguide3": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "krvskp": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "spambase": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "spamassasin": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "magic04": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "imdb": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "a8a": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "diabetes_us": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "susy": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
        "higgs": {"lr": [1e-3],"d_model": [64], "nhead": [4], "num_layers": [6], "beta": [0.99], "s": [0.2]},
    }


    config_dict = params_list.get(data_name, params_list['higgs'])
    n_runs = 5

    ###########不变##########
    config_dict["use_cuda"] = False
    config_dict["batch_size"]=1
    config_dict["n_classes"] = 2
    ###########不变##########
    if data_name=="susy" or data_name=="higgs":
        n_runs = 3  # 神经网络随机性，运行多次取平均
    else:
        n_runs=5
    return n_runs,config_dict


def config_packetLSTM(data_name):
    """
    返回 packetLSTM 的超参数配置以及运行次数。
    数据集类型判断：
        - 合成数据集：magic04, a8a, SUSY, HIGGS 使用 Max 聚合
        - 真实数据集：imdb 使用 Mean 聚合
        - 其余数据集默认 Max 聚合
    学习率按数据集分别指定，其余默认 0.001。
    """
    # 学习率映射表
    lr_map = {
        "magic04": 0.0006,
        "imdb": 0.0008,
        "a8a": 0.0009,
        "susy": 0.0008,
        "higgs": 0.0002,
    }
    # 聚合函数映射表
    agg_map = {
        "imdb": "mean",  # 真实数据集用 Mean
        # 合成数据集及其他默认用 Max
    }

    default_lr = 0.001
    default_agg = "max"

    lr = lr_map.get(data_name, default_lr)
    agg_func = agg_map.get(data_name, default_agg)

    config_dict = {
        "hidden_size": [64],  # [修改点] 加上中括号变成列表，供 for 循环使用
        "lr": [lr],  # [修改点] 加上中括号变成列表
        "aggregator": [agg_func],  # [修改点] 键名统一为 aggregator，并加上中括号
        #"use_scaler": [True, False],  # 本身就是列表，正常遍历
        "use_scaler": [True],
        # 以下是不参与网格搜索遍历的全局静态参数
        "optimizer": "AdamW",
        "loss": "CrossEntropyLoss",
        "n_classes": 2,
    }
    if data_name=="susy" or data_name=="higgs":
        n_runs = 3  # 神经网络随机性，运行多次取平均
    else:
        n_runs=5
    return n_runs, config_dict

def config_MODL_three_learners(data_name):
    """
    返回 Packet-T1 的超参数配置。
    包含 CHead 注意力层特有的映射参数 d_k。
    """
    lr_map = {
        "magic04": 0.0006,
        "imdb": 0.0008,
        "a8a": 0.0009,
        "SUSY": 0.0008,
        "HIGGS": 0.0002,
    }

    agg_map = {
        "imdb": "mean",
    }

    default_lr = 0.001
    default_agg = "max"

    lr = lr_map.get(data_name, default_lr)
    agg_func = agg_map.get(data_name, default_agg)
    if data_name not in ["magic04","imdb","a8a","susy","higgs","airlines","elec","svmguide3","krvskp","spam"]:
        optimiz=["AdamW"]
    else:
        optimiz=["AdamW"]

    config_dict = {
        "hidden_size": [128],  # 通道数 C
        "d_k": [8],  # 注意力头的子空间维度 (d_k)
        "lr": [lr],
        "aggregator": [agg_func],
        "use_scaler": [True],
        "optimizer": optimiz,
        "loss": "CrossEntropyLoss",
        "n_classes": 2,
        "data_name":data_name,
        "mlp_width":[128],
        "mlp_layers":[4]
    }
    if data_name=="susy" or data_name=="higgs":
        n_runs = 3  # 神经网络随机性，运行多次取平均
    else:
        n_runs=5
    return n_runs, config_dict


def config_MODL_three_learners_no_scaler(data_name):
    n_runs, config_dict = config_MODL_three_learners(data_name)
    config_dict = dict(config_dict)
    config_dict["use_scaler"] = [False]
    return n_runs, config_dict


def config_memory_calibrated_residual(data_name):
    """
    Configuration for Memory-Calibrated Residual Adaptation (MCRA).
    This keeps the Packet-T1 memory branch but replaces independent score-sum
    fusion with a memory-conditioned residual adapter, reliability gate, and
    distillation-style fast-to-stable consolidation.
    """
    lr_map = {
        "magic04": 0.0006,
        "imdb": 0.0008,
        "a8a": 0.0009,
        "susy": 0.0008,
        "higgs": 0.0002,
    }
    agg_map = {
        "imdb": "mean",
    }

    default_lr = 0.001
    default_agg = "max"
    lr = lr_map.get(data_name, default_lr)
    agg_func = agg_map.get(data_name, default_agg)

    config_dict = {
        "hidden_size": [128],
        "d_k": [8],
        "lr": [lr],
        "aggregator": [agg_func],
        "use_scaler": [True],
        "optimizer": ["AdamW"],
        "loss": "CrossEntropyLoss",
        "n_classes": 2,
        "data_name": data_name,
        "lr_variance": lr,
        "use_var_embedding": [True],
        "residual_scale": [1.0],
        "recency_tau": [0.05],
        "distill_lambda": [0.1],
        "gate_hidden_size": [16],
        "gate_type": ["learned"],
        "ablation_mode": ["full"],
        "method_label": "MCRA",
    }
    if data_name == "susy" or data_name == "higgs":
        n_runs = 3
    else:
        n_runs = 5
    return n_runs, config_dict


def _config_memory_calibrated_residual_variant(data_name, method_label, **overrides):
    n_runs, config_dict = config_memory_calibrated_residual(data_name)
    config_dict = dict(config_dict)
    config_dict["method_label"] = method_label
    for key, value in overrides.items():
        config_dict[key] = value
    return n_runs, config_dict


def config_memory_calibrated_residual_StableOnly(data_name):
    return _config_memory_calibrated_residual_variant(
        data_name,
        "MCRAStableOnly",
        residual_scale=[0.0],
        distill_lambda=[0.0],
        ablation_mode=["stable_only"],
    )


def config_memory_calibrated_residual_FastOnly(data_name):
    return _config_memory_calibrated_residual_variant(
        data_name,
        "MCRAFastOnly",
        distill_lambda=[0.0],
        ablation_mode=["fast_only"],
    )


def config_memory_calibrated_residual_NoDistill(data_name):
    return _config_memory_calibrated_residual_variant(
        data_name,
        "MCRANoDistill",
        distill_lambda=[0.0],
        ablation_mode=["full"],
    )


def config_memory_calibrated_residual_RuleGate(data_name):
    return _config_memory_calibrated_residual_variant(
        data_name,
        "MCRARuleGate",
        gate_type=["rule"],
        ablation_mode=["full"],
    )
