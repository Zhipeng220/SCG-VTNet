import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import get_args
from losses import AnatomicalLoss, HierarchicalContrastiveLoss
from model import HybridAttentionModel
from train import forward_model, parse_model_outputs, unpack_batch
from utils import load_and_preprocess_train_data


def build_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def grad_vector(loss, model):
    params = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in params)
    if not getattr(loss, "requires_grad", False):
        return torch.zeros(total, device=next(model.parameters()).device)

    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = []
    for param, grad in zip(params, grads):
        if grad is None:
            flat.append(torch.zeros_like(param).reshape(-1))
        else:
            flat.append(grad.reshape(-1))
    return torch.cat(flat) if flat else torch.zeros(0, device=next(model.parameters()).device)


def cosine(a, b):
    denom = torch.norm(a) * torch.norm(b)
    if denom.item() == 0:
        return 0.0
    return float(torch.dot(a, b).detach().cpu() / denom.detach().cpu())


def row(name, loss, grad, base_grad=None):
    grad_norm = float(torch.norm(grad).detach().cpu())
    diff_norm = "" if base_grad is None else float(torch.norm(grad - base_grad).detach().cpu())
    cos = "" if base_grad is None else cosine(grad, base_grad)
    return {
        "variant": name,
        "loss_value": float(loss.detach().cpu()) if torch.is_tensor(loss) else float(loss),
        "requires_grad": bool(getattr(loss, "requires_grad", False)),
        "grad_fn": type(loss.grad_fn).__name__ if getattr(loss, "grad_fn", None) is not None else "None",
        "grad_norm": grad_norm,
        "grad_diff_from_ce_cont": diff_norm,
        "cosine_with_ce_cont": cos,
    }


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, "anat_gradient_check.csv")

    torch.manual_seed(args.seed)
    device = build_device()
    datasets, _, label_encoder, _, input_dim = load_and_preprocess_train_data(
        args.train_csv_path,
        args.image_features_path,
        use_advanced_features=True,
        seed=args.seed,
    )
    loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    keypoints, raw_xyz, image_features, hand_detected, labels = unpack_batch(next(iter(loader)), device)

    model = HybridAttentionModel(
        input_dim=input_dim,
        num_classes=len(label_encoder.classes_),
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        image_feature_dim=image_features.shape[-1],
        use_gated_fusion=True,
        use_dynamic_graph=False,
        use_cross_attn=True,
        use_gcn_branch=False,
        use_keypoint_refinement=True,
        refine_scale=args.refine_scale,
    ).to(device)
    model.train()

    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    cont_criterion = HierarchicalContrastiveLoss(temperature=args.contrastive_temperature)
    anat_criterion = AnatomicalLoss(
        device=device,
        angle_weight=args.anat_angle_weight,
        length_weight=args.anat_length_weight,
        planarity_weight=args.anat_planarity_weight,
    )

    outputs = forward_model(model, keypoints, image_features, raw_xyz=raw_xyz, hand_detected=hand_detected)
    logits, embeddings, _, refined_xyz = parse_model_outputs(outputs)
    noisy_raw_xyz = raw_xyz + torch.randn_like(raw_xyz) * 0.01
    valid_hand = hand_detected.reshape(-1, 1, 1)
    noisy_raw_xyz = noisy_raw_xyz * valid_hand
    hand_confidences = hand_detected.unsqueeze(1).expand(-1, 21)
    noisy_outputs = forward_model(model, keypoints, image_features, raw_xyz=noisy_raw_xyz, hand_detected=hand_detected)
    _, _, _, noisy_refined_xyz = parse_model_outputs(noisy_outputs)

    ce_loss = ce_criterion(logits, labels)
    cont_loss = args.contrastive_weight * cont_criterion(embeddings, labels)
    raw_anat_loss = args.anatomical_weight_final * anat_criterion(raw_xyz, labels, confidences=hand_confidences)
    refined_anat_loss = args.anatomical_weight_final * anat_criterion(refined_xyz, labels, confidences=hand_confidences)
    refined_anat_loss_noisy = args.anatomical_weight_final * anat_criterion(
        noisy_refined_xyz, labels, confidences=hand_confidences
    )

    ce_cont = ce_loss + cont_loss
    ce_cont_raw = ce_cont + raw_anat_loss
    ce_cont_refined = ce_cont + refined_anat_loss
    ce_cont_refined_noisy = ce_cont + refined_anat_loss_noisy

    grads = {
        "L_CE": grad_vector(ce_loss, model),
        "L_Cont": grad_vector(cont_loss, model),
        "L_anat_raw_input": grad_vector(raw_anat_loss, model),
        "L_anat_refined_xyz": grad_vector(refined_anat_loss, model),
        "L_anat_refined_xyz_noisy": grad_vector(refined_anat_loss_noisy, model),
        "L_CE_plus_L_Cont": grad_vector(ce_cont, model),
        "L_CE_plus_L_Cont_plus_L_anat_raw_input": grad_vector(ce_cont_raw, model),
        "L_CE_plus_L_Cont_plus_L_anat_refined_xyz": grad_vector(ce_cont_refined, model),
        "L_CE_plus_L_Cont_plus_L_anat_refined_xyz_noisy": grad_vector(ce_cont_refined_noisy, model),
    }

    rows = [
        row("L_CE", ce_loss, grads["L_CE"]),
        row("L_Cont", cont_loss, grads["L_Cont"]),
        row("L_anat_raw_input", raw_anat_loss, grads["L_anat_raw_input"]),
        row("L_anat_refined_xyz", refined_anat_loss, grads["L_anat_refined_xyz"]),
        row("L_anat_refined_xyz_noisy", refined_anat_loss_noisy, grads["L_anat_refined_xyz_noisy"]),
        row("L_CE + L_Cont", ce_cont, grads["L_CE_plus_L_Cont"]),
        row("L_CE + L_Cont + L_anat_raw_input", ce_cont_raw,
            grads["L_CE_plus_L_Cont_plus_L_anat_raw_input"], grads["L_CE_plus_L_Cont"]),
        row("L_CE + L_Cont + L_anat_refined_xyz", ce_cont_refined,
            grads["L_CE_plus_L_Cont_plus_L_anat_refined_xyz"], grads["L_CE_plus_L_Cont"]),
        row("L_CE + L_Cont + L_anat_refined_xyz_noisy", ce_cont_refined_noisy,
            grads["L_CE_plus_L_Cont_plus_L_anat_refined_xyz_noisy"], grads["L_CE_plus_L_Cont"]),
    ]

    fieldnames = [
        "variant",
        "loss_value",
        "requires_grad",
        "grad_fn",
        "grad_norm",
        "grad_diff_from_ce_cont",
        "cosine_with_ce_cont",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved gradient-flow check to {output_csv}")
    for item in rows:
        print(item)


if __name__ == "__main__":
    main()
