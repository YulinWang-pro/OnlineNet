import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoResSetLearner(nn.Module):
    """
    100% 严格还原原始 residual.py 中的 proto_forward, project_set 和 decode
    """

    def __init__(self, size_out, num_features, num_blocks=6, num_layers_per_block=3, layer_width=250, embedding_dim=32):
        super().__init__()
        self.norm_inputs = False

        # 对应原始的 self.embeddings (原版预留了 150 的余量，这里改为动态获取 + 适当余量)
        self.embedding = nn.Embedding(num_embeddings=num_features + 100, embedding_dim=embedding_dim)

        # 对应原始的 self.project: 将 1维连续值与 32维Embedding拼接
        self.project = nn.Linear(1 + embedding_dim, layer_width)

        # 对应原始的 FCBlockNorm 堆叠
        self.stage_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            layers = []
            for _ in range(num_layers_per_block):
                layers.append(nn.Linear(layer_width, layer_width))
                layers.append(nn.LayerNorm(layer_width))
                layers.append(nn.ReLU())
            self.stage_blocks.append(nn.Sequential(*layers))

        # 对应原始的 self.output_layers (每一个 block 都有独立的输出层，用于深度监督)
        self.output_layers = nn.ModuleList([nn.Linear(layer_width, size_out) for _ in range(num_blocks)])

    def forward(self, X_base, aux_feat, aux_mask):
        device = X_base.device

        # 1. 分离并提取可用的连续特征
        avail_aux = aux_feat[aux_mask == 1].unsqueeze(0)
        x_proto = torch.cat([X_base, avail_aux], dim=1)  # [1, num_avail]

        # 2. 严格还原: norm_inputs 预处理 (对数归一化)
        if self.norm_inputs:
            x_proto = torch.log(torch.abs(x_proto + 1))

        # 3. 严格还原: 生成精准的 IDs (Base特征的ID接上Aux特征的ID)
        base_len = X_base.shape[1]
        ids = torch.cat([
            torch.arange(base_len, device=device),
            torch.nonzero(aux_mask[0]).reshape(-1) + base_len
        ])

        # 4. 严格还原 project_set: 特征值与 Embedding 拼接
        weights = torch.ones(x_proto.shape[1], device=device).unsqueeze(0).unsqueeze(-1)
        avail_vals = x_proto.unsqueeze(-1)  # [1, num_avail, 1]
        avail_embs = self.embedding(ids).unsqueeze(0)  # [1, num_avail, emb_dim]

        proj = torch.cat([avail_vals, avail_embs], dim=-1)
        proj = self.project(proj)  # [1, num_avail, layer_width]

        # 聚合 (Sum Pooling)
        weights_sum = weights.sum(dim=1, keepdim=True)
        weights_sum[weights_sum == 0.0] = 1.0
        h_pool = (proj * weights).sum(dim=1, keepdim=True) / weights_sum
        h_pool = h_pool.squeeze(1)  # [1, layer_width]

        # 5. 严格还原 decode: 深度监督与 Softmax
        backcast = h_pool
        stage_forecast = 0.0
        predictions = []
        for i, block in enumerate(self.stage_blocks):
            f = block(backcast)
            backcast = backcast + f
            stage_forecast = stage_forecast + f

            # 【核心还原】原版在这里对每一层的输出都执行了 F.softmax !
            pred_layer = F.softmax(self.output_layers[i](stage_forecast), dim=1)
            predictions.append(pred_layer)

        # 返回所有深度的预测累加和
        return torch.sum(torch.stack(predictions), dim=0)


class KalmanMLPproto2(nn.Module):
    """
    还原原始的 KalmanMLPproto2 骨干网络：仅提供预测，不执行相加
    """

    def __init__(self, **kwargs):
        super().__init__()
        size_in_mlp = kwargs["num_features"]
        n_classes = kwargs.get("n_classes", 2)
        lr_variance = kwargs.get("lr_variance", 0.001)

        # --- 1. 快速学习器 (OLR) ---
        self.size_in_lr = size_in_mlp + 1
        self.register_buffer('theta', torch.zeros(self.size_in_lr))
        self.register_buffer('Hessian', lr_variance * torch.eye(self.size_in_lr))

        # --- 2. 中等学习器 (MLP) ---
        mlp_layers = kwargs.get("mlp_layers", 3)
        mlp_width = kwargs.get("mlp_width", 250)

        self.fc_layers = nn.ModuleList([nn.Linear(size_in_mlp, mlp_width)])
        for _ in range(mlp_layers - 2):
            self.fc_layers.append(nn.Linear(mlp_width, mlp_width))
        self.fc_layers.append(nn.Linear(mlp_width, n_classes))

        # --- 3. 慢速学习器 (ProtoRes) ---
        self.set_learner = ProtoResSetLearner(
            size_out=n_classes,
            num_features=size_in_mlp,
            num_blocks=kwargs.get("set_blocks", 6),
            num_layers_per_block=kwargs.get("set_layers", 3),
            layer_width=kwargs.get("set_width", 250)
        )

    def online_logistic_regression_step(self, x):
        """完全数学等价的快速逻辑回归前向预测"""
        with torch.no_grad():
            x_aug = torch.cat([x[0], torch.ones(1, device=x.device)])
            val = torch.clamp(x_aug @ self.theta, min=-10, max=10)
            prob_pos = 1 / (1 + torch.exp(-val))
            y_hat_lr = torch.tensor([[1 - prob_pos.item(), prob_pos.item()]], device=x.device)
        return y_hat_lr, x_aug

    def forward(self, batch_dict):
        """对齐原始 forward，返回独立预测供外层融合"""
        X = batch_dict['X_base']
        aux_feat = batch_dict['X_aux_new']
        aux_mask = batch_dict['aux_mask']

        # 1. 集合学习器
        proto_pred = self.set_learner(X, aux_feat, aux_mask)

        # 2. 生成 MLP/OLR 需要的密集向量 (缺失处已由0填充的特征)
        x_dense = torch.cat([X, aux_feat * aux_mask], dim=1)

        # 3. 快速学习器 (OLR)
        y_hat_lr, x_aug = self.online_logistic_regression_step(x_dense)

        # 4. 中等学习器 (MLP - 严格还原原版的循环前向)
        h = x_dense
        for layer in self.fc_layers[:-1]:
            h = F.relu(layer(h))
        y_hat_MLP = self.fc_layers[-1](h)

        # 返回三个独立的预测及扩展后的特征(用于OLR更新)
        return y_hat_lr, y_hat_MLP, proto_pred, x_aug

    def update_olr(self, x_aug, y_true):
        """独立的 OLR RIRLS 闭式更新 (对应 Algorithm 1)"""
        with torch.no_grad():
            val = torch.clamp(x_aug @ self.theta, min=-10, max=10)
            p = 1 / (1 + torch.exp(-val))
            H_k = x_aug.unsqueeze(0)
            S_k = H_k @ self.Hessian @ H_k.T + (p * (1 - p)) ** 2
            K_k = self.Hessian @ H_k.T / S_k
            self.theta += (K_k * (y_true.float() - p)).squeeze()
            self.Hessian -= torch.outer(K_k.squeeze(), K_k.squeeze()) * S_k.squeeze()