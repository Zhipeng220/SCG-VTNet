# losses.py

import torch
from torch import nn
import math
# [FIX] 确保 numpy 被导入
import numpy as np


class HierarchicalContrastiveLoss(nn.Module):
    """
    [创新点] 分层对比损失 (InfoNCE实现).
    该损失函数旨在将同一类别的手势嵌入在特征空间中拉近，同时推开不同类别的手势嵌入。
    这对于学习细粒度的手势差异（如 ASL 中的 'A' 和 'S'）至关重要。
    """

    def __init__(self, temperature=0.1, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        device = features.device

        # L2-normalize a projeção de características
        features = nn.functional.normalize(features, dim=1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # Para estabilidade numérica
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Máscara para remover auto-similaridade da diagonal
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # Calculo do log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Computar a média do log-likelihood sobre todas as âncoras positivas
        # Por simplicidade, consideramos todas as amostras da mesma classe como positivas
        pos_count = mask.sum(1)
        valid = pos_count > 0
        mean_log_prob_pos = (mask * log_prob).sum(1) / (pos_count + 1e-8)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        if valid.any():
            return loss[valid].mean()
        return features.sum() * 0.0


class AnatomicalLoss(nn.Module):
    def __init__(self, device='cpu', angle_weight=1.0, length_weight=2.0, planarity_weight=0.5):
        super().__init__()
        self.device = device
        self.angle_weight = angle_weight
        self.length_weight = length_weight
        self.planarity_weight = planarity_weight
        self.finger_indices = {'thumb': [1, 2, 3, 4], 'index': [5, 6, 7, 8], 'middle': [9, 10, 11, 12],
                               'ring': [13, 14, 15, 16], 'pinky': [17, 18, 19, 20]}
        self.palm_indices = [0, 1, 5, 9, 13, 17]

    def _calculate_angle_loss(self, hand_kpts, confidences):
        loss = torch.tensor(0.0, device=hand_kpts.device, dtype=hand_kpts.dtype)
        if confidences is None: confidences = torch.ones_like(hand_kpts[..., 0])
        for _, indices in self.finger_indices.items():
            for i in range(len(indices) - 2):
                p0_idx, p1_idx, p2_idx = indices[i], indices[i + 1], indices[i + 2]
                p0, p1, p2 = hand_kpts[:, p0_idx], hand_kpts[:, p1_idx], hand_kpts[:, p2_idx]
                v1, v2 = p0 - p1, p2 - p1

                # [FIX] 增加微小偏移防止 norm 产生 0，导致除以 0
                v1_norm = torch.norm(v1 + 1e-7, dim=1)
                v2_norm = torch.norm(v2 + 1e-7, dim=1)
                cos_theta = torch.sum(v1 * v2, dim=1) / (v1_norm * v2_norm + 1e-8)

                # [FIX] 严格限制在 (-1+eps, 1-eps) 之间，防止 acos 在边界处导数爆炸产生 nan
                angle = torch.acos(torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7))

                constraint_confidence = torch.min(
                    torch.stack([confidences[:, p0_idx], confidences[:, p1_idx], confidences[:, p2_idx]]), dim=0).values

                theta_min = 0.05
                theta_max = math.pi - 0.05
                angle_loss = torch.relu(theta_min - angle) ** 2 + torch.relu(angle - theta_max) ** 2
                loss += torch.mean(constraint_confidence * angle_loss)
        return loss

    def _calculate_length_loss(self, hand_kpts, confidences):
        loss = torch.tensor(0.0, device=hand_kpts.device, dtype=hand_kpts.dtype)
        if confidences is None: confidences = torch.ones_like(hand_kpts[..., 0])
        for _, indices in self.finger_indices.items():
            if len(indices) < 4: continue
            p_indices = [indices[0], indices[1], indices[2], indices[3]]
            constraint_confidence = torch.min(confidences[:, p_indices], dim=1).values

            # [FIX] 对计算距离的向量预先加上 1e-7，防止退化的零点产生 norm(0) 的 nan 梯度
            ref_len = torch.norm((hand_kpts[:, indices[1]] - hand_kpts[:, indices[0]]) + 1e-7, dim=1) + 1e-8

            ratio1 = torch.norm((hand_kpts[:, indices[2]] - hand_kpts[:, indices[1]]) + 1e-7, dim=1) / ref_len
            loss += torch.mean(constraint_confidence * (torch.relu(ratio1 - 1.5) ** 2 + torch.relu(0.5 - ratio1) ** 2))

            ratio2 = torch.norm((hand_kpts[:, indices[3]] - hand_kpts[:, indices[2]]) + 1e-7, dim=1) / ref_len
            loss += torch.mean(constraint_confidence * (torch.relu(ratio2 - 1.2) ** 2 + torch.relu(0.4 - ratio2) ** 2))
        return loss

    def _calculate_planarity_loss(self, hand_kpts, confidences):
        if confidences is None: confidences = torch.ones_like(hand_kpts[..., 0])
        palm_kpts, palm_confidences = hand_kpts[:, self.palm_indices, :], confidences[:, self.palm_indices]
        centroid = torch.sum(palm_kpts * palm_confidences.unsqueeze(-1), dim=1, keepdim=True) / (
                torch.sum(palm_confidences, dim=1, keepdim=True).unsqueeze(-1) + 1e-8)
        centered_kpts = palm_kpts - centroid
        try:
            C = torch.bmm((centered_kpts * palm_confidences.unsqueeze(-1)).transpose(1, 2), centered_kpts)
            # [FIX] 向协方差矩阵对角线注入极微小的噪声，防止 SVD 对奇异(纯零)矩阵求导产生 nan
            C = C + torch.eye(3, device=C.device).unsqueeze(0) * 1e-6
            _, _, Vh = torch.linalg.svd(C, full_matrices=False)
            Vh = Vh.to(hand_kpts.device)
            dists = torch.sum(centered_kpts * Vh[:, -1, :].unsqueeze(1), dim=2) ** 2
            return 0.5 * torch.mean(palm_confidences * dists)
        except RuntimeError:
            return torch.tensor(0.0, device=hand_kpts.device, dtype=hand_kpts.dtype)

    def forward(self, xyz, labels=None, confidences=None):
        if xyz.ndim != 3:
            raise ValueError(f"AnatomicalLoss expects a 3D tensor shaped (B, 21, C); got {tuple(xyz.shape)}")

        keypoints_batch = xyz[..., :3]

        if confidences is None:
            confidences_batch = torch.ones_like(keypoints_batch[..., 0])
        else:
            confidences_batch = confidences.to(device=keypoints_batch.device, dtype=keypoints_batch.dtype)

        # [FIX] 核心修复：直接从 Batch 中过滤掉那些没有检测到手（hand_detected == 0 / 均被 zero_fill）的帧。
        # 彻底防止所有零向量进入几何运算。
        valid_batch_mask = confidences_batch[:, 0] > 0.5

        if not valid_batch_mask.any():
            # [技巧] 返回乘以 0 的结果，这样 loss 值为 0 但不切断运算图，防止分布式训练报错
            return (keypoints_batch * 0.0).sum()

        # 仅对存在有效手部关键点的帧进行提取计算
        valid_kpts = keypoints_batch[valid_batch_mask]
        valid_confs = confidences_batch[valid_batch_mask]

        angle_loss = self._calculate_angle_loss(valid_kpts, valid_confs)
        length_loss = self._calculate_length_loss(valid_kpts, valid_confs)
        planarity_loss = self._calculate_planarity_loss(valid_kpts, valid_confs)

        total_loss = (self.angle_weight * angle_loss +
                      self.length_weight * length_loss +
                      self.planarity_weight * planarity_loss)

        return total_loss
    
