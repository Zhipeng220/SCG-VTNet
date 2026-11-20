# model.py
import torch
from torch import nn
# import torchvision.models as models  # Keep import just in case


# --- [ _FFN, FourierPositionalEncoding, MLPBaseline, AdvancedAttentionBlock, AdvancedAttentionModel, GCNLayer - UNCHANGED ] ---
class _FFN(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim), nn.Dropout(dropout)
        )

    def forward(self, x): return self.main(x)


class FourierPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_freqs=10, alpha=10000.0):
        super().__init__()
        self.embed_dim, self.num_freqs, self.alpha = embed_dim, num_freqs, alpha
        self.freq_bands = nn.Parameter(alpha ** (torch.linspace(0.0, 1.0, num_freqs) ** 2), requires_grad=False)
        self.projection = nn.Linear(3 * 2 * num_freqs, embed_dim)  # Only uses 3D coords

    def forward(self, coords):  # Expects coords with shape (B, N, 3)
        B, N, _ = coords.shape
        mean = torch.mean(coords, dim=1, keepdim=True)
        std = torch.std(coords, dim=1, keepdim=True) + 1e-7
        norm_coords = (coords - mean) / std
        scaled_coords = norm_coords.unsqueeze(-1) * self.freq_bands
        fourier_features = torch.cat([torch.sin(scaled_coords), torch.cos(scaled_coords)], dim=-1)
        return self.projection(fourier_features.reshape(B, N, -1))


class MLPBaseline(nn.Module):
    def __init__(self, input_size, num_classes):  # input_size = 21 * input_dim (e.g., 21*11)
        super().__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.5)
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x_keypoints, x_image_features, hand_detected=None):
        x = x_keypoints
        features = self.main(x.reshape(x.size(0), -1))
        return self.fc(features), features


class AdvancedAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = _FFN(embed_dim, embed_dim * 4, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class AdvancedAttentionModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=4, num_heads=8, embed_dim=256, dropout=0.2):
        super().__init__()
        self.coord_embed = nn.Linear(input_dim, embed_dim)  # input_dim is engineered (e.g., 11)
        self.pos_encoder = FourierPositionalEncoding(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([AdvancedAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim * 21, num_classes)  # Flattened features

    def forward(self, x_keypoints, x_image_features, hand_detected=None):
        x = x_keypoints
        # Only use first 3 dims for pos encoding
        coord_embedding = self.coord_embed(x)
        pos_encoding = self.pos_encoder(x[..., :3])
        if hand_detected is not None:
            valid_hand = hand_detected.reshape(-1, 1, 1).to(device=x.device, dtype=x.dtype)
            coord_embedding = coord_embedding * valid_hand
            pos_encoding = pos_encoding * valid_hand
        else:
            valid_hand = None
        features = self.dropout(coord_embedding + pos_encoding)
        for block in self.blocks:
            features = block(features)
            if valid_hand is not None:
                features = features * valid_hand
        embeddings = self.final_norm(features).reshape(features.size(0), -1)  # Flatten
        logits = self.fc(embeddings)
        return logits, embeddings


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, adj):
        aggregated_features = torch.einsum('bnn,bnj->bnj', adj, x)
        return self.linear(aggregated_features)


# --- [ NEW: ImageOnlyBaseline for Ablation Study ] ---
class ImageOnlyBaseline(nn.Module):
    def __init__(self, image_feature_dim, num_classes, dropout=0.5):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(image_feature_dim, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_keypoints, x_image_features, hand_detected=None):
        if x_image_features.ndim == 3:
            x_image_features = x_image_features.mean(dim=1)
        logits = self.main(x_image_features)
        return logits, x_image_features


# --- [ NEW: End ] ---


# --- [ Modify HybridAttentionModel ] ---
class HybridAttentionModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=4, num_heads=8, embed_dim=256, dropout=0.2,
                 use_gated_fusion=True, use_dynamic_graph=True,
                 image_feature_dim=2048, visual_token_count=49,
                 use_cross_attn=True, use_gcn_branch=True,
                 use_keypoint_refinement=False, refine_scale=0.05):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_feature_dim = image_feature_dim
        self.visual_token_count = visual_token_count

        self.use_gated_fusion = use_gated_fusion
        self.use_dynamic_graph = use_dynamic_graph
        self.use_cross_attn = use_cross_attn
        self.use_gcn_branch = use_gcn_branch
        self.use_keypoint_refinement = use_keypoint_refinement
        self.refine_scale = refine_scale
        # --- [ NEW: End ] ---

        # --- [ Keypoint branch - Self-Attention ] ---
        self.coord_embed = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = FourierPositionalEncoding(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_blocks = nn.ModuleList(
            [AdvancedAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.attention_norm = nn.LayerNorm(embed_dim)

        if self.use_keypoint_refinement:
            self.kpt_refine_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 3)
            )
            self.refined_coord_embed = nn.Linear(3, embed_dim)

        # --- [ Keypoint branch - GCN (Conditional) ] ---
        if self.use_gcn_branch:
            self.gcn_blocks = nn.ModuleList()
            gcn_dims = [embed_dim] + [embed_dim] * 2
            for i in range(len(gcn_dims) - 1):
                self.gcn_blocks.append(GCNLayer(gcn_dims[i], gcn_dims[i + 1]))
                self.gcn_blocks.append(nn.ReLU())
                self.gcn_blocks.append(nn.LayerNorm(gcn_dims[i + 1]))
            self.register_buffer('adj_static', self.create_adjacency_matrix())
            if self.use_dynamic_graph:
                self.graph_q_proj = nn.Linear(embed_dim, embed_dim // 4)
                self.graph_k_proj = nn.Linear(embed_dim, embed_dim // 4)
                self.graph_activation = nn.Tanh()

        # --- [ Fusion Layers (Conditional) ] ---
        if self.use_cross_attn:
            self.cross_attn_norm_kpt = nn.LayerNorm(embed_dim)
            self.img_k_proj = nn.Linear(self.image_feature_dim, embed_dim)
            self.img_v_proj = nn.Linear(self.image_feature_dim, embed_dim)
            self.visual_pos_embed = nn.Parameter(torch.zeros(1, self.visual_token_count, embed_dim))
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            self.cross_attn_dropout = nn.Dropout(dropout)

        # --- [ Final FC (Conditional Input Size) ] ---
        keypoint_fusion_dim = embed_dim * 21  # (Att_flat)

        if self.use_gated_fusion:
            # Gating layer input dim depends on whether GCN is used
            gating_input_dim = keypoint_fusion_dim * 2 if self.use_gcn_branch else keypoint_fusion_dim
            self.gating_layer = nn.Sequential(
                nn.Linear(gating_input_dim, gating_input_dim // 4), nn.ReLU(),
                nn.Linear(gating_input_dim // 4, keypoint_fusion_dim), nn.Sigmoid()  # Output dim is Att_flat
            )
            final_fc_input_dim = keypoint_fusion_dim  # Gated output
        else:
            # No gating, input dim depends on whether GCN is used
            final_fc_input_dim = keypoint_fusion_dim * 2 if self.use_gcn_branch else keypoint_fusion_dim

        # If NOT using cross-attn, we must add image features (Late Fusion)
        if not self.use_cross_attn:
            final_fc_input_dim += self.image_feature_dim

        self.fc = nn.Linear(final_fc_input_dim, num_classes)

    def prepare_image_tokens(self, x_image_features):
        if x_image_features.ndim == 2:
            return x_image_features.unsqueeze(1)
        if x_image_features.ndim == 3:
            return x_image_features
        raise ValueError(f"Expected image features with shape (B, C) or (B, T, C), got {tuple(x_image_features.shape)}")

    def pool_image_features(self, x_image_features):
        if x_image_features.ndim == 3:
            return x_image_features.mean(dim=1)
        return x_image_features

    def create_adjacency_matrix(self):
        # ... (UNCHANGED)
        adj = torch.zeros(21, 21)
        connections = [[0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
                       [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8],
                       [9, 10], [10, 11], [11, 12], [13, 14], [14, 15], [15, 16],
                       [17, 18], [18, 19], [19, 20]]
        for i, j in connections: adj[i, j] = adj[j, i] = 1
        adj = adj + torch.eye(21)
        d_inv_sqrt = torch.pow(adj.sum(1), -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        return torch.mm(torch.diag(d_inv_sqrt), adj) @ torch.diag(d_inv_sqrt)

    def forward(self, x_keypoints, x_image_features, raw_xyz=None, hand_detected=None,
                return_diagnostics=False):
        B = x_keypoints.size(0)

        # --- Keypoint processing ---
        coord_embedding = self.coord_embed(x_keypoints)
        pos_encoding = self.pos_encoder(x_keypoints[..., :3])
        if hand_detected is not None:
            valid_hand = hand_detected.reshape(-1, 1, 1).to(device=x_keypoints.device, dtype=x_keypoints.dtype)
            coord_embedding = coord_embedding * valid_hand
            pos_encoding = pos_encoding * valid_hand
        else:
            valid_hand = None
        embedded = self.dropout(coord_embedding + pos_encoding)

        # 1. Keypoint Self-Attention
        attn_features = embedded
        for block in self.attention_blocks:
            attn_features = block(attn_features)
            if valid_hand is not None:
                attn_features = attn_features * valid_hand

        refined_xyz = None
        if self.use_keypoint_refinement:
            base_xyz = raw_xyz if raw_xyz is not None else x_keypoints[..., :3]
            delta_xyz = self.refine_scale * torch.tanh(self.kpt_refine_head(attn_features))
            if valid_hand is not None:
                base_xyz = base_xyz * valid_hand
                delta_xyz = delta_xyz * valid_hand
            refined_xyz = base_xyz + delta_xyz
            refined_embedding = self.refined_coord_embed(refined_xyz)
            if valid_hand is not None:
                refined_embedding = refined_embedding * valid_hand
            attn_features = attn_features + self.dropout(refined_embedding)
            if valid_hand is not None:
                attn_features = attn_features * valid_hand

        if self.use_gcn_branch:
            gcn_features = embedded
            if self.use_dynamic_graph:
                q = self.graph_q_proj(embedded);
                k = self.graph_k_proj(embedded)
                adj_dynamic = self.graph_activation(torch.einsum('bif,bjf->bij', q, k))
                fused_adj = self.adj_static.unsqueeze(0) + adj_dynamic
            else:
                fused_adj = self.adj_static.unsqueeze(0).expand(B, -1, -1)
            for module in self.gcn_blocks:
                if isinstance(module, GCNLayer):
                    gcn_features = module(gcn_features, fused_adj)
                elif isinstance(module, nn.LayerNorm):
                    gcn_features = module(gcn_features)
                else:
                    gcn_features = module(gcn_features)
            gcn_flat = gcn_features.reshape(B, -1)
        else:
            # If no GCN, create a zero tensor to keep fusion logic consistent
            gcn_flat = torch.zeros(B, self.embed_dim * 21, device=x_keypoints.device)

        cross_attn_weights = None
        if self.use_cross_attn:
            q_kpt = self.cross_attn_norm_kpt(attn_features)
            image_tokens = self.prepare_image_tokens(x_image_features)
            k_img = self.img_k_proj(image_tokens)
            v_img = self.img_v_proj(image_tokens)
            if k_img.size(1) == self.visual_pos_embed.size(1):
                pos = self.visual_pos_embed.to(device=k_img.device, dtype=k_img.dtype)
                k_img = k_img + pos
                v_img = v_img + pos
            fused_attn_features, cross_attn_weights = self.cross_attn(
                q_kpt, k_img, v_img,
                need_weights=return_diagnostics,
                average_attn_weights=False,
            )
            fused_attn_features = attn_features + self.cross_attn_dropout(fused_attn_features)
            attn_flat = self.attention_norm(fused_attn_features).reshape(B, -1)
        else:
            # Use original self-attention features
            attn_flat = self.attention_norm(attn_features).reshape(B, -1)

        gate = None
        if self.use_gated_fusion:
            # Gate input depends on GCN branch
            gate_input = torch.cat([attn_flat, gcn_flat], dim=1) if self.use_gcn_branch else attn_flat
            gate = self.gating_layer(gate_input)

            # If the GCN branch is disabled, the gate reweights attention features only.
            gcn_term = (1 - gate) * gcn_flat if self.use_gcn_branch else 0
            fused_keypoints = gate * attn_flat + gcn_term
        else:
            # No gating, just concatenate
            fused_keypoints = torch.cat([attn_flat, gcn_flat], dim=1) if self.use_gcn_branch else attn_flat

        if self.use_cross_attn:
            logits = self.fc(fused_keypoints)
            final_embeddings = fused_keypoints
        else:
            # Late Fusion: concatenate image features *after* fusion
            pooled_image_features = self.pool_image_features(x_image_features)
            combined_features = torch.cat([fused_keypoints, pooled_image_features], dim=1)
            logits = self.fc(combined_features)
            final_embeddings = combined_features

        if return_diagnostics:
            return {
                "logits": logits,
                "embeddings": final_embeddings,
                "gate": gate,
                "refined_xyz": refined_xyz,
                "cross_attn_weights": cross_attn_weights,
            }

        return logits, final_embeddings, gate, refined_xyz
    # --- [ Forward method END ] ---
# --- [ Modify HybridAttentionModel END ] ---

