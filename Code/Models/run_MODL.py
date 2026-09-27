from Utils.utils import seed_everything
from Utils.metric_utils import get_all_metrics
from Models.MODL import KalmanMLPproto2
from tqdm import tqdm
import numpy as np
import time
import torch
import torch.nn as nn


def print_model_parameters(model):
    # 1. 统计可通过反向传播优化的参数 (MLP 和 SetLearner)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. 统计不可训练参数 (如果有被冻结的层)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    # 3. 统计状态缓存 (即 OLR 的 theta 和 Hessian，它们没有梯度)
    buffer_params = sum(b.numel() for b in model.buffers())

    print("=" * 40)
    print(f"🚀 模型参数量统计 (MODLFramework)")
    print("=" * 40)
    print(f"可训练神经网络参数 (MLP + ProtoRes): {trainable_params:,}")
    print(f"闭式更新状态缓存 (OLR):              {buffer_params:,}")
    print(f"总计驻留显存/内存的参数量:           {trainable_params + frozen_params + buffer_params:,}")
    print("=" * 40)
def run_MODL(X, mask, Y, num_runs, config):
    result = {}
    lr_list = config.get("lr", [0.01])
    if not isinstance(lr_list, list):
        lr_list = [lr_list]

    no_instances = len(Y)
    num_features = len(X[0])

    # ---------------- 核心变动区 ----------------
    # 从 config 中动态提取模型需要的架构参数 (剥离诸如 lr 等非模型层面的训练参数)
    model_kwargs = {
        "num_features": num_features,
        "n_classes": config.get("n_classes", 2),
        "lr_variance": config.get("lr_variance", 0.001),
        "mlp_layers": config.get("mlp_layers", 3),
        "mlp_width": config.get("mlp_width", 250),
        "set_blocks": config.get("set_blocks", 6),
        "set_layers": config.get("set_layers", 3),
        "set_width": config.get("set_width", 250)
    }
    # --------------------------------------------

    print(
        f"MODL - Features: {num_features} | MLP: {model_kwargs['mlp_layers']}L | SetBlocks: {model_kwargs['set_blocks']}")

    # 预加载为 Tensor
    X_tensor = torch.FloatTensor(X)
    mask_tensor = torch.FloatTensor(mask)
    Y_tensor = torch.LongTensor(Y)
    empty_base = torch.empty((1, 0))  # 没有强行指定的 Base Features
    for lr in lr_list:
        eval_list = []

        for j in range(num_runs):
            seed_everything(j)
            Y_pred, Y_logits = [], []
            start_time = time.time()

            # 1. 实例化模型：使用 ** 动态解包参数，实现完全解耦
            model = KalmanMLPproto2(**model_kwargs)

            print_model_parameters(model)

            # 只为深层网络开启梯度更新
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
            criterion = nn.CrossEntropyLoss()

            for t in tqdm(range(no_instances), desc=f"Run {j + 1}/{num_runs} | LR {lr}"):
                x_t = X_tensor[t:t + 1]
                m_t = mask_tensor[t:t + 1]
                y_val = Y_tensor[t:t + 1].view(1)

                batch_dict = {
                    'X_base': empty_base,
                    'X_aux_new': x_t,
                    'aux_mask': m_t
                }
                # --- A. 前向传播获取三个独立结果 ---
                y_hat_lr, y_hat_mlp, proto_pred, x_aug = model(batch_dict)
                # --- B. 核心融合：Score Sum ---
                # 在这里执行相加，对应原版中的 merge="sum" 逻辑，并保留原作者特调的 0.25 权重系数
                y_hat = y_hat_lr + y_hat_mlp + 0.25 * proto_pred

                # 记录预测 (类别 1 概率)
                y_probs = torch.softmax(y_hat, dim=1).detach().cpu().numpy()[0]
                Y_pred.append(np.argmax(y_probs))
                Y_logits.append(y_probs[1])

                # --- B. 混合更新机制 ---
                loss = criterion(y_hat, y_val)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 2. 快速学习器的闭式矩阵更新 (游离于计算图外)
                model.update_olr(x_aug, y_val)

            taken_time = time.time() - start_time
            del model  # 释放内存

            # --- C. 后处理及评估 ---
            nan_list = np.where(np.isnan(Y_logits))[0]
            for k in nan_list:
                Y_logits[k] = Y_pred[k]

            eval_list.append(
                get_all_metrics(Y, np.array(Y_pred).reshape(-1, 1), np.array(Y_logits).reshape(-1, 1), taken_time)
            )

        result[f"MODL_ScoreSum_lr_{lr}"] = eval_list

    return result