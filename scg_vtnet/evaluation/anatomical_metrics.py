import argparse
import csv
import json
import math
import os

import torch
from torch.utils.data import DataLoader

from scg_vtnet.training.core_arguments import get_args
from scg_vtnet.models.scg_vtnet import HybridAttentionModel
from scg_vtnet.training.train_loop import forward_model, occlude_points_pair, parse_model_outputs, structured_noise_pair, unpack_batch
from scg_vtnet.datasets.multimodal import load_and_preprocess_train_data


FINGER_INDICES = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}
PALM_INDICES = [0, 1, 5, 9, 13, 17]


def load_experiment_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def build_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _xyz_only(xyz):
    if xyz.ndim != 3 or xyz.shape[1] != 21 or xyz.shape[-1] < 3:
        raise ValueError(f"Expected keypoints shaped (B, 21, C>=3), got {tuple(xyz.shape)}")
    return xyz[..., :3]


def compute_anatomical_metrics(xyz, hand_detected=None):
    xyz = _xyz_only(xyz)
    if hand_detected is not None:
        valid = hand_detected.to(device=xyz.device).bool()
        xyz = xyz[valid]
    if xyz.shape[0] == 0:
        return {
            "angle_violation_rate": 0.0,
            "length_violation_rate": 0.0,
            "palm_planarity_error": 0.0,
            "total_anat_violation_score": 0.0,
        }
    device = xyz.device
    eps = 1e-8

    angle_violations = []
    length_violations = []
    for indices in FINGER_INDICES.values():
        for i in range(len(indices) - 2):
            p0, p1, p2 = xyz[:, indices[i]], xyz[:, indices[i + 1]], xyz[:, indices[i + 2]]
            v1, v2 = p0 - p1, p2 - p1
            cos_theta = torch.sum(v1 * v2, dim=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + eps)
            angle = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
            angle_violations.append(((angle > math.pi - 0.05) | (angle < 0.05)).float())

        ref_len = torch.norm(xyz[:, indices[1]] - xyz[:, indices[0]], dim=1) + eps
        ratio1 = torch.norm(xyz[:, indices[2]] - xyz[:, indices[1]], dim=1) / ref_len
        ratio2 = torch.norm(xyz[:, indices[3]] - xyz[:, indices[2]], dim=1) / ref_len
        length_violations.append(((ratio1 > 1.5) | (ratio1 < 0.5)).float())
        length_violations.append(((ratio2 > 1.2) | (ratio2 < 0.4)).float())

    angle_violation_rate = torch.cat(angle_violations).mean() if angle_violations else torch.tensor(0.0, device=device)
    length_violation_rate = torch.cat(length_violations).mean() if length_violations else torch.tensor(0.0, device=device)

    palm = xyz[:, PALM_INDICES, :]
    centroid = palm.mean(dim=1, keepdim=True)
    centered = palm - centroid
    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        normal = vh[:, -1, :]
        palm_planarity_error = torch.sum(centered * normal.unsqueeze(1), dim=2).pow(2).mean()
    except RuntimeError:
        palm_planarity_error = torch.tensor(0.0, device=device)

    total = angle_violation_rate + length_violation_rate + palm_planarity_error
    return {
        "angle_violation_rate": float(angle_violation_rate.detach().cpu()),
        "length_violation_rate": float(length_violation_rate.detach().cpu()),
        "palm_planarity_error": float(palm_planarity_error.detach().cpu()),
        "total_anat_violation_score": float(total.detach().cpu()),
    }


def write_violation_report(rows, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = [
        "variant",
        "angle_violation_rate",
        "length_violation_rate",
        "palm_planarity_error",
        "total_anat_violation_score",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_model_violation_rows(model, data_loader, device, variant, include_noise=True):
    model.eval()
    raw_batches, refined_batches, hand_batches = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            keypoints, raw_xyz, image_features, hand_detected, _ = unpack_batch(batch, device)
            outputs = forward_model(model, keypoints, image_features, raw_xyz=raw_xyz, hand_detected=hand_detected)
            _, _, _, refined_xyz = parse_model_outputs(outputs)
            raw_batches.append(raw_xyz.detach().cpu())
            hand_batches.append(hand_detected.detach().cpu())
            if refined_xyz is not None:
                refined_batches.append(refined_xyz.detach().cpu())

    hand_all = torch.cat(hand_batches, dim=0)
    rows = [{"variant": f"{variant}:raw_clean_xyz", **compute_anatomical_metrics(torch.cat(raw_batches, dim=0), hand_all)}]
    if refined_batches:
        rows.append({"variant": f"{variant}:refined_xyz", **compute_anatomical_metrics(torch.cat(refined_batches, dim=0), hand_all)})

    if include_noise:
        noise_scenarios = {
            "gaussian_0.5": {"feature_sigma": 0.5, "raw_sigma": 0.005},
            "gaussian_1.0": {"feature_sigma": 1.0, "raw_sigma": 0.01},
            "occlusion_3": {"occlusions": 3},
            "structured_finger": {"structured": True},
        }
        for scenario, noise_config in noise_scenarios.items():
            noisy_raw_batches, noisy_refined_batches, noisy_hand_batches = [], [], []
            with torch.no_grad():
                for batch in data_loader:
                    keypoints, raw_xyz, image_features, hand_detected, _ = unpack_batch(batch, device)
                    noisy_keypoints = keypoints.clone()
                    noisy_raw_xyz = raw_xyz.clone()
                    if "feature_sigma" in noise_config:
                        noisy_keypoints = noisy_keypoints + torch.randn_like(noisy_keypoints) * noise_config["feature_sigma"]
                        noisy_raw_xyz = noisy_raw_xyz + torch.randn_like(noisy_raw_xyz) * noise_config["raw_sigma"]
                    elif "occlusions" in noise_config:
                        noisy_keypoints, noisy_raw_xyz = occlude_points_pair(
                            noisy_keypoints, noisy_raw_xyz, num_occlusions=noise_config["occlusions"]
                        )
                    elif noise_config.get("structured"):
                        noisy_keypoints, noisy_raw_xyz = structured_noise_pair(noisy_keypoints, noisy_raw_xyz)
                    valid_hand = hand_detected.reshape(-1, 1, 1)
                    noisy_keypoints = noisy_keypoints * valid_hand
                    noisy_raw_xyz = noisy_raw_xyz * valid_hand
                    outputs = forward_model(
                        model,
                        noisy_keypoints,
                        image_features,
                        raw_xyz=noisy_raw_xyz,
                        hand_detected=hand_detected,
                    )
                    _, _, _, refined_xyz = parse_model_outputs(outputs)
                    noisy_raw_batches.append(noisy_raw_xyz.detach().cpu())
                    noisy_hand_batches.append(hand_detected.detach().cpu())
                    if refined_xyz is not None:
                        noisy_refined_batches.append(refined_xyz.detach().cpu())
            noisy_hand_all = torch.cat(noisy_hand_batches, dim=0)
            rows.append({
                "variant": f"{variant}:{scenario}:noisy_raw_xyz",
                **compute_anatomical_metrics(torch.cat(noisy_raw_batches, dim=0), noisy_hand_all),
            })
            if noisy_refined_batches:
                rows.append({
                    "variant": f"{variant}:{scenario}:refined_xyz",
                    **compute_anatomical_metrics(torch.cat(noisy_refined_batches, dim=0), noisy_hand_all),
                })
    return rows


def main():
    args = get_args()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output_csv", type=str, default=os.path.join(args.output_dir, "anat_violation_report.csv"))
    parser.add_argument("--model_path", type=str, default=args.model_path)
    parser.add_argument("--experiment_config", type=str, default=args.experiment_config)
    known, _ = parser.parse_known_args()

    device = build_device()
    experiment_config = load_experiment_config(known.experiment_config) if known.experiment_config else None
    use_advanced_features = experiment_config.get("use_advanced_features", True) if experiment_config else True
    datasets, _, label_encoder, _, input_dim = load_and_preprocess_train_data(
        args.train_csv_path, args.image_features_path, use_advanced_features=use_advanced_features, seed=args.seed
    )
    loader = DataLoader(datasets["test"], batch_size=args.batch_size * 2, shuffle=False)
    image_dim = datasets["test"].image_feature_dim

    rows = []
    if known.model_path:
        if experiment_config:
            if experiment_config.get("model_class") != "HybridAttentionModel":
                raise ValueError(
                    "anat_metrics.py model checkpoint evaluation currently expects HybridAttentionModel; "
                    f"got {experiment_config.get('model_class')}."
                )
            input_dim = experiment_config.get("input_dim", input_dim)
            model_kwargs = {
                "input_dim": input_dim,
                "num_classes": len(label_encoder.classes_),
                "num_layers": experiment_config.get("num_layers", args.num_layers),
                "num_heads": experiment_config.get("num_heads", args.num_heads),
                "embed_dim": experiment_config.get("embed_dim", args.embed_dim),
                "dropout": experiment_config.get("dropout", args.dropout),
                "image_feature_dim": image_dim,
                "use_gated_fusion": experiment_config.get("use_gated_fusion", True),
                "use_dynamic_graph": experiment_config.get("use_dynamic_graph", False),
                "use_cross_attn": experiment_config.get("use_cross_attn", True),
                "use_gcn_branch": experiment_config.get("use_gcn_branch", False),
                "use_keypoint_refinement": experiment_config.get("use_keypoint_refinement", False),
                "refine_scale": experiment_config.get("refine_scale", args.refine_scale),
                "visual_token_count": experiment_config.get("visual_token_count", 49),
            }
            model = HybridAttentionModel(**model_kwargs).to(device)
        else:
            visual_token_count = datasets["test"].image_feature_shape[1] if len(datasets["test"].image_feature_shape) == 3 else 1
            model = HybridAttentionModel(
                input_dim=input_dim,
                num_classes=len(label_encoder.classes_),
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                embed_dim=args.embed_dim,
                dropout=args.dropout,
                image_feature_dim=image_dim,
                use_gated_fusion=True,
                use_dynamic_graph=False,
                use_cross_attn=True,
                use_gcn_branch=False,
                use_keypoint_refinement=args.use_keypoint_refinement,
                refine_scale=args.refine_scale,
                visual_token_count=visual_token_count,
            ).to(device)
        model.load_state_dict(torch.load(known.model_path, map_location=device))
        rows.extend(collect_model_violation_rows(model, loader, device, os.path.basename(known.model_path)))
    else:
        raw_xyz = datasets["test"].tensors[1]
        hand_detected = datasets["test"].tensors[3]
        rows.append({"variant": "raw_clean_xyz", **compute_anatomical_metrics(raw_xyz, hand_detected)})

    write_violation_report(rows, known.output_csv)
    print(f"Saved anatomical violation report to {known.output_csv}")


if __name__ == "__main__":
    main()
