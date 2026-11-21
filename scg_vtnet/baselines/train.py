# train.py
import logging
import os
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
import random

# Import model classes
import csv
from pathlib import Path

from model import MLPBaseline, AdvancedAttentionModel, HybridAttentionModel, ImageOnlyBaseline, VisualTokenTransformerBaseline


def parse_model_outputs(outputs):
    """Return logits, embeddings, gate, refined_xyz for all model output styles."""
    if isinstance(outputs, dict):
        return (
            outputs.get("logits"),
            outputs.get("embeddings"),
            outputs.get("gate"),
            outputs.get("refined_xyz"),
        )
    if isinstance(outputs, (tuple, list)):
        logits = outputs[0] if len(outputs) > 0 else None
        embeddings = outputs[1] if len(outputs) > 1 else None
        gate = outputs[2] if len(outputs) > 2 else None
        refined_xyz = outputs[3] if len(outputs) > 3 else None
        return logits, embeddings, gate, refined_xyz
    return outputs, None, None, None


def unpack_batch(batch, device):
    if len(batch) == 5:
        keypoint_inputs, raw_xyz, image_features, hand_detected, labels = batch
    elif len(batch) == 4:
        keypoint_inputs, raw_xyz, image_features, labels = batch
        hand_detected = torch.ones(labels.shape[0], dtype=torch.float32)
    elif len(batch) == 3:
        keypoint_inputs, image_features, labels = batch
        raw_xyz = keypoint_inputs[..., :3]
        hand_detected = torch.ones(labels.shape[0], dtype=torch.float32)
    else:
        raise ValueError(f"Expected a 3-, 4-, or 5-tensor batch, got {len(batch)} tensors.")
    return (
        keypoint_inputs.to(device),
        raw_xyz.to(device),
        image_features.to(device=device, dtype=torch.float32),
        hand_detected.to(device),
        labels.to(device),
    )


def forward_model(model, keypoints, image_features, raw_xyz=None, hand_detected=None):
    if isinstance(model, HybridAttentionModel):
        return model(keypoints, image_features, raw_xyz=raw_xyz, hand_detected=hand_detected)
    return model(keypoints, image_features, hand_detected=hand_detected)


def apply_modality_dropout(model, keypoints, raw_xyz, image_features, hand_detected, probability):
    """Drop either skeleton or visual input per sample for ModDrop-style training."""
    if probability <= 0 or not isinstance(model, HybridAttentionModel):
        return keypoints, raw_xyz, image_features, hand_detected

    batch_size = keypoints.size(0)
    device = keypoints.device
    drop_mask = torch.rand(batch_size, device=device) < probability
    if not drop_mask.any():
        return keypoints, raw_xyz, image_features, hand_detected

    drop_skeleton = drop_mask & (torch.rand(batch_size, device=device) < 0.5)
    drop_visual = drop_mask & ~drop_skeleton

    if drop_skeleton.any():
        skel_shape = (batch_size,) + (1,) * (keypoints.ndim - 1)
        skel_keep = (~drop_skeleton).reshape(skel_shape).to(dtype=keypoints.dtype)
        keypoints = keypoints * skel_keep
        raw_xyz = raw_xyz * skel_keep
        hand_detected = hand_detected * (~drop_skeleton).to(dtype=hand_detected.dtype)

    if drop_visual.any():
        visual_shape = (batch_size,) + (1,) * (image_features.ndim - 1)
        visual_keep = (~drop_visual).reshape(visual_shape).to(dtype=image_features.dtype)
        image_features = image_features * visual_keep

    return keypoints, raw_xyz, image_features, hand_detected


def train_model(model, model_save_path, train_loader, val_loader, optimizer, scheduler, ce_criterion, anat_criterion,
                contrastive_criterion, args, device, contrastive_weight=0.1):
    """(V19) Main training loop, supports multimodal input (Fixed Cosine Warm-up)."""
    model.to(device)
    best_val_acc = -float("inf")

    patience_counter = 0
    early_stopping_patience = args.early_stopping_patience

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    warned_refined_missing = False

    if anat_criterion and hasattr(anat_criterion, 'label_classes_') and hasattr(train_loader.dataset, 'tensors'):
        try:
            pass
        except Exception as e:
            logging.warning(f"Could not set label_classes_ for AnatomicalLoss: {e}")

    for epoch in range(args.epochs):
        model.train()
        total_loss, total_ce, total_anat, total_cont = 0, 0, 0, 0

        # --- [ BUG FIX: Cosine Warm-up Schedule ] ---
        if getattr(args, "anat_schedule", "cosine") == "constant":
            current_anat_weight = args.anatomical_weight_final
        else:
            # 修正为从 initial 平滑上升到 final (Warm-up)
            progress = epoch / max(1, args.epochs - 1)
            initial_weight = getattr(args, "anatomical_weight_initial", 0.0)
            final_weight = args.anatomical_weight_final
            current_anat_weight = initial_weight + 0.5 * (1 - math.cos(math.pi * progress)) * (final_weight - initial_weight)
        # --- [ BUG FIX END ] ---

        for batch in train_loader:
            keypoint_inputs, raw_xyz, image_features, hand_detected, labels = unpack_batch(batch, device)

            augmented_keypoints = keypoint_inputs.clone()
            augmented_raw_xyz = raw_xyz.clone()
            augmented_image_features = image_features.clone()
            valid_hand = hand_detected.reshape(-1, 1, 1)

            if torch.rand(1).item() < 0.8:
                if torch.rand(1).item() < 0.5:
                    augmented_keypoints, augmented_raw_xyz = augment_spatial_pair(augmented_keypoints, augmented_raw_xyz)
                else:
                    if torch.rand(1).item() < 0.7:
                        augmented_keypoints += torch.randn_like(augmented_keypoints) * 0.5
                        augmented_raw_xyz += torch.randn_like(augmented_raw_xyz) * 0.005
                    else:
                        augmented_keypoints, augmented_raw_xyz = occlude_points_pair(
                            augmented_keypoints, augmented_raw_xyz, num_occlusions=3
                        )

                augmented_keypoints = augmented_keypoints * valid_hand
                augmented_raw_xyz = augmented_raw_xyz * valid_hand

                noise_std = 0.1
                augmented_image_features += torch.randn_like(augmented_image_features) * noise_std

            augmented_keypoints, augmented_raw_xyz, augmented_image_features, hand_detected = apply_modality_dropout(
                model,
                augmented_keypoints,
                augmented_raw_xyz,
                augmented_image_features,
                hand_detected,
                getattr(args, "modality_dropout", 0.0),
            )

            optimizer.zero_grad(set_to_none=True)
            outputs = forward_model(
                model,
                augmented_keypoints,
                augmented_image_features,
                raw_xyz=augmented_raw_xyz,
                hand_detected=hand_detected,
            )
            logits, embeddings, _, refined_xyz = parse_model_outputs(outputs)

            loss = ce_criterion(logits, labels)
            total_ce += loss.item()

            anat_loss = torch.tensor(0.0, device=device)
            anat_target = getattr(args, "anat_target", "none")
            if anat_criterion and current_anat_weight > 0 and anat_target != "none":
                if anat_target == "raw_input":
                    hand_confidences = hand_detected.unsqueeze(1).expand(-1, 21)
                    anat_loss = current_anat_weight * anat_criterion(
                        augmented_raw_xyz, labels, confidences=hand_confidences
                    )
                elif anat_target == "refined_xyz":
                    if refined_xyz is not None:
                        hand_confidences = hand_detected.unsqueeze(1).expand(-1, 21)
                        anat_loss = current_anat_weight * anat_criterion(
                            refined_xyz, labels, confidences=hand_confidences
                        )
                    elif not warned_refined_missing:
                        logging.warning("anat_target='refined_xyz' but model did not return refined_xyz; skipping anatomical loss.")
                        warned_refined_missing = True
                else:
                    raise ValueError(f"Unknown anat_target: {anat_target}")

            loss += anat_loss
            total_anat += anat_loss.item() if isinstance(anat_loss, torch.Tensor) else anat_loss

            if contrastive_criterion and embeddings is not None and contrastive_weight > 0:
                cont_loss = contrastive_weight * contrastive_criterion(embeddings, labels)
                loss += cont_loss
                total_cont += cont_loss.item()

            loss.backward()
            if args.clip_grad_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_acc, val_loss = evaluate_model(model, val_loader, device)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        logging.info(
            f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} "
            f"(CE: {total_ce / len(train_loader):.4f}, Anat: {total_anat / len(train_loader):.4f}, Cont: {total_cont / len(train_loader):.4f}) | "
            f"Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Anat W: {current_anat_weight:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"  -> New best validation accuracy: {best_val_acc:.4f}. Model saved to {model_save_path}")
        else:
            patience_counter += 1
            logging.info(f"  -> No improvement in validation accuracy for {patience_counter} epoch(s).")

        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            logging.info(f"\n--- Early stopping triggered after {patience_counter} epochs with no improvement. ---")
            logging.info(f"Best validation accuracy achieved: {best_val_acc:.4f}")
            break

    return history


def evaluate_model(model, data_loader, device):
    """(V19) Evaluates model accuracy and loss (supports multimodal)."""
    model.eval()
    all_preds, all_labels = [], []
    total_val_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            keypoint_inputs, raw_xyz, image_features, hand_detected, labels = unpack_batch(batch, device)

            outputs = forward_model(model, keypoint_inputs, image_features, raw_xyz=raw_xyz, hand_detected=hand_detected)
            logits, _, _, _ = parse_model_outputs(outputs)

            loss = criterion(logits, labels)
            total_val_loss += loss.item()

            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    avg_val_loss = total_val_loss / len(data_loader)
    all_labels_np = torch.cat(all_labels).numpy()
    all_preds_np = torch.cat(all_preds).numpy()
    acc = accuracy_score(all_labels_np, all_preds_np)

    return acc, avg_val_loss


def final_test_and_report(model_class, model_path, test_loader, device, label_encoder, args, input_dim, config,
                          output_paths, image_feature_dim=2048):
    """(V19) Loads model, evaluates, generates reports (supports multimodal)."""
    logging.info(f"\n--- Final Evaluation for {os.path.basename(model_path)} ---")
    num_classes = len(label_encoder.classes_)

    use_gated = config.get("use_gated_fusion", False)
    use_dynamic = config.get("use_dynamic_graph", False)
    current_embed_dim = config.get("embed_dim", args.embed_dim)

    # --- [ AI BUG FIX START ] ---
    # 从 config 字典中读取正确的超参数，而不是使用 args 默认值
    current_num_layers = config.get("num_layers", args.num_layers)
    current_num_heads = config.get("num_heads", args.num_heads)
    # --- [ AI BUG FIX END ] ---

    use_cross_attn = config.get("use_cross_attn", True)
    use_keypoint_refinement = config.get("use_keypoint_refinement", getattr(args, "use_keypoint_refinement", False))
    refine_scale = config.get("refine_scale", getattr(args, "refine_scale", 0.05))
    visual_token_count = config.get("visual_token_count", 49)

    # --- [ FIX ] ---
    # Default must match the training loop in main.py
    use_gcn_branch = config.get("use_gcn_branch", False)
    # --- [ FIX END ] ---

    model_kwargs = {"input_dim": input_dim, "num_classes": num_classes,
                    "num_layers": current_num_layers,  # <-- 使用修正后的变量
                    "num_heads": current_num_heads,  # <-- 使用修正后的变量
                    "embed_dim": current_embed_dim, "dropout": config.get("dropout", args.dropout),
                    "image_feature_dim": image_feature_dim}

    # Instantiate the correct model
    if model_class == MLPBaseline:
        model_kwargs.pop("image_feature_dim")
        flat_kpt_dim = 21 * input_dim
        model = model_class(input_size=flat_kpt_dim, num_classes=num_classes)
        logging.info("Instantiated MLPBaseline model for evaluation.")
    elif model_class == AdvancedAttentionModel:
        model_kwargs.pop("image_feature_dim")
        model = model_class(**model_kwargs)
        logging.info("Instantiated AdvancedAttentionModel model for evaluation.")
    elif model_class == ImageOnlyBaseline:
        model = model_class(image_feature_dim=image_feature_dim,
                            num_classes=num_classes,
                            dropout=config.get("dropout", args.dropout))
        logging.info(f"Instantiated ImageOnlyBaseline (Image Dim: {image_feature_dim}) for evaluation.")
    elif model_class == VisualTokenTransformerBaseline:
        model = model_class(
            image_feature_dim=image_feature_dim,
            num_classes=num_classes,
            visual_token_count=visual_token_count,
            num_layers=current_num_layers,
            num_heads=current_num_heads,
            embed_dim=current_embed_dim,
            dropout=config.get("dropout", args.dropout),
        )
        logging.info(f"Instantiated VisualTokenTransformerBaseline (Image Dim: {image_feature_dim}) for evaluation.")
    elif model_class == HybridAttentionModel:
        model = model_class(**model_kwargs, use_gated_fusion=use_gated, use_dynamic_graph=use_dynamic,
                            use_cross_attn=use_cross_attn, use_gcn_branch=use_gcn_branch,
                            use_keypoint_refinement=use_keypoint_refinement, refine_scale=refine_scale,
                            uniform_visual_attention=config.get("uniform_visual_attention", False),
                            visual_token_count=visual_token_count)
        gated_status = "ENABLED" if use_gated else "DISABLED"
        dynamic_status = "ENABLED" if use_dynamic else "DISABLED"
        cross_attn_status = "ENABLED" if use_cross_attn else "DISABLED (Late Fusion)"
        gcn_status = "ENABLED" if use_gcn_branch else "DISABLED"
        logging.info(
            f"Instantiated HybridAttentionModel (Multimodal Ready - Image Dim: {image_feature_dim}) "
            f"with Cross-Attn {cross_attn_status}, GCN {gcn_status}, Gated Fusion {gated_status}, Dynamic Graph {dynamic_status}.")
    else:
        logging.error(
            f"Unknown model class: {model_class.__name__ if hasattr(model_class, '__name__') else model_class}");
        return 0.0

    # Load model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"Successfully loaded model weights from {model_path}")
    except FileNotFoundError:
        logging.error(f"ERROR: Model file not found at {model_path}.");
        return 0.0
    except Exception as e:
        logging.error(f"Error loading model weights: {e}");
        return 0.0

    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            keypoint_inputs, raw_xyz, image_features, hand_detected, labels = unpack_batch(batch, device)
            outputs = forward_model(model, keypoint_inputs, image_features, raw_xyz=raw_xyz, hand_detected=hand_detected)
            logits, _, _, _ = parse_model_outputs(outputs)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    all_labels_np, all_preds_np = torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()
    if "prediction_dir" in config:
        save_matched_predictions(config, "clean", test_loader.dataset, label_encoder, all_labels_np, all_preds_np)
    logging.info(
        f"\nClassification Report:\n{classification_report(all_labels_np, all_preds_np, target_names=label_encoder.classes_, digits=4, zero_division=0)}")
    cm = confusion_matrix(all_labels_np, all_preds_np)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix for {os.path.basename(model_path)}", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    cm_filename = f"cm_{os.path.basename(model_path).replace('.pth', '.png')}"
    cm_path = os.path.join(output_paths['cm_dir'], cm_filename)
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logging.info(f"Confusion matrix saved to {cm_path}")
    return {
        "accuracy": accuracy_score(all_labels_np, all_preds_np),
        "macro_f1": f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0),
    }


def save_matched_predictions(config, condition, dataset, encoder, labels, predictions):
    path = Path(config["prediction_dir"]) / f"predictions_{condition}.csv"
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "original_index", "hand_detected", "true_label", "predicted_label"])
        writer.writerows(
            (condition, int(index), int(detected), label, prediction)
            for index, detected, label, prediction in zip(
                dataset.indices, dataset.hand_detected, encoder.inverse_transform(labels),
                encoder.inverse_transform(predictions), strict=True,
            )
        )


def test_robustness(model_class, model_path, test_dataset, device, label_encoder, args, input_dim, config,
                    image_feature_dim=2048):
    """(V19) Robustness testing (supports multimodal)."""
    logging.info(f"\n--- Robustness Test for {os.path.basename(model_path)} ---")

    # [删除] 移除了跳过 MLP 的代码
    # if model_class == MLPBaseline:
    #     logging.info("Skipping robustness test for MLP.");
    #     return None

    num_classes = len(label_encoder.classes_)
    use_gated = config.get("use_gated_fusion", False)
    use_dynamic = config.get("use_dynamic_graph", False)
    current_embed_dim = config.get("embed_dim", args.embed_dim)
    use_cross_attn = config.get("use_cross_attn", True)
    use_keypoint_refinement = config.get("use_keypoint_refinement", getattr(args, "use_keypoint_refinement", False))
    refine_scale = config.get("refine_scale", getattr(args, "refine_scale", 0.05))
    visual_token_count = config.get("visual_token_count", 49)

    # --- [ AI BUG FIX START ] ---
    # 从 config 字典中读取正确的超参数，而不是使用 args 默认值
    current_num_layers = config.get("num_layers", args.num_layers)
    current_num_heads = config.get("num_heads", args.num_heads)
    # --- [ AI BUG FIX END ] ---

    # --- [ FIX ] ---
    # Default must match the training loop in main.py
    use_gcn_branch = config.get("use_gcn_branch", False)
    # --- [ FIX END ] ---

    model_kwargs = {"input_dim": input_dim, "num_classes": num_classes,
                    "num_layers": current_num_layers,  # <-- 使用修正后的变量
                    "num_heads": current_num_heads,  # <-- 使用修正后的变量
                    "embed_dim": current_embed_dim, "dropout": config.get("dropout", args.dropout),
                    "image_feature_dim": image_feature_dim}

    # --- [ AI 补丁：添加了 MLPBaseline 的实例化逻辑 ] ---
    if model_class == MLPBaseline:
        model_kwargs.pop("image_feature_dim")
        flat_kpt_dim = 21 * input_dim
        model = model_class(input_size=flat_kpt_dim, num_classes=num_classes)
        logging.info(f"Instantiated MLPBaseline (Input Dim: {flat_kpt_dim}) for robustness test.")
    # --- [ AI 补丁结束 ] ---

    elif model_class == HybridAttentionModel:
        model = model_class(**model_kwargs, use_gated_fusion=use_gated, use_dynamic_graph=use_dynamic,
                            use_cross_attn=use_cross_attn, use_gcn_branch=use_gcn_branch,
                            use_keypoint_refinement=use_keypoint_refinement, refine_scale=refine_scale,
                            uniform_visual_attention=config.get("uniform_visual_attention", False),
                            visual_token_count=visual_token_count)
        logging.info("Instantiated HybridAttentionModel for robustness test.")

    elif model_class == ImageOnlyBaseline:
        model = model_class(image_feature_dim=image_feature_dim,
                            num_classes=num_classes,
                            dropout=config.get("dropout", args.dropout))
        logging.info(f"Instantiated ImageOnlyBaseline (Image Dim: {image_feature_dim}) for robustness test.")
    elif model_class == VisualTokenTransformerBaseline:
        model = model_class(
            image_feature_dim=image_feature_dim,
            num_classes=num_classes,
            visual_token_count=visual_token_count,
            num_layers=current_num_layers,
            num_heads=current_num_heads,
            embed_dim=current_embed_dim,
            dropout=config.get("dropout", args.dropout),
        )
        logging.info("Instantiated VisualTokenTransformerBaseline for robustness test.")

    elif model_class == AdvancedAttentionModel:
        model_kwargs.pop("image_feature_dim")
        model = model_class(**model_kwargs)
        logging.info("Instantiated AdvancedAttentionModel for robustness test.")

    else:
        logging.error(
            f"Unknown model class for robustness test: {model_class.__name__ if hasattr(model_class, '__name__') else model_class}")
        return None

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        logging.error(f"ERROR: Model file not found at {model_path} during robustness test.");
        return None
    # [新增] 捕获由参数不匹配引起的 RuntimeError
    except RuntimeError as e:
        logging.error(f"ERROR: Failed to load model weights. This is likely an architecture mismatch: {e}")
        return None

    model.to(device)
    results = {}

    noise_scenarios = {
        "gaussian_0.5": {"feature_sigma": 0.5, "raw_sigma": 0.005},
        "gaussian_1.0": {"feature_sigma": 1.0, "raw_sigma": 0.01},
        "occlusion_3": {"occlusions": 3},
        "structured_finger": {"structured": True},
        "visual_gaussian": {"visual_sigma": 0.1},
        "visual_token_dropout": {"visual_token_dropout": 0.25},
        "visual_patch_occlusion": {"visual_patch_occlusion": 0.25},
        "skeleton_visual_joint_noise": {"feature_sigma": 0.5, "raw_sigma": 0.005, "visual_sigma": 0.1},
    }

    if model_class in (ImageOnlyBaseline, VisualTokenTransformerBaseline):
        logging.info("ImageOnlyBaseline detected; running visual robustness scenarios only.")
        noise_scenarios = {
            key: value for key, value in noise_scenarios.items()
            if key.startswith("visual_") or key == "skeleton_visual_joint_noise"
        }

    # --- [ AI 补丁：添加了 MLP 的特别说明 ] ---
    if model_class == MLPBaseline:
        logging.info("MLPBaseline flattens keypoints; robustness test is running.")
    # --- [ AI 补丁结束 ] ---

    robustness_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)
    model.eval()
    for name, noise_config in noise_scenarios.items():
        if "perturbation_seed" in config:
            torch.manual_seed(config["perturbation_seed"])
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in robustness_loader:
                keypoints, raw_xyz, image_features, hand_detected, labels = unpack_batch(batch, device)
                noisy_keypoints = keypoints.clone()
                noisy_raw_xyz = raw_xyz.clone()
                noisy_image_features = image_features.clone()

                if "feature_sigma" in noise_config:
                    noisy_keypoints = noisy_keypoints + torch.randn_like(noisy_keypoints) * noise_config["feature_sigma"]
                    noisy_raw_xyz = noisy_raw_xyz + torch.randn_like(noisy_raw_xyz) * noise_config["raw_sigma"]
                if "occlusions" in noise_config:
                    noisy_keypoints, noisy_raw_xyz = occlude_points_pair(
                        noisy_keypoints, noisy_raw_xyz, num_occlusions=noise_config["occlusions"]
                    )
                if noise_config.get("structured"):
                    noisy_keypoints, noisy_raw_xyz = structured_noise_pair(noisy_keypoints, noisy_raw_xyz)

                valid_hand = hand_detected.reshape(-1, 1, 1)
                noisy_keypoints = noisy_keypoints * valid_hand
                noisy_raw_xyz = noisy_raw_xyz * valid_hand

                if "visual_sigma" in noise_config:
                    noisy_image_features = noisy_image_features + torch.randn_like(noisy_image_features) * noise_config["visual_sigma"]
                if "visual_token_dropout" in noise_config:
                    drop_prob = noise_config["visual_token_dropout"]
                    if noisy_image_features.ndim == 3:
                        keep = (torch.rand(noisy_image_features.size(0), noisy_image_features.size(1), 1,
                                           device=noisy_image_features.device) > drop_prob).to(noisy_image_features.dtype)
                    else:
                        keep = (torch.rand(noisy_image_features.size(0), 1,
                                           device=noisy_image_features.device) > drop_prob).to(noisy_image_features.dtype)
                    noisy_image_features = noisy_image_features * keep
                if "visual_patch_occlusion" in noise_config and noisy_image_features.ndim == 3:
                    drop_prob = noise_config["visual_patch_occlusion"]
                    token_count = noisy_image_features.size(1)
                    grid_size = int(round(token_count ** 0.5))
                    if grid_size * grid_size == token_count:
                        tokens_2d = noisy_image_features.reshape(noisy_image_features.size(0), grid_size, grid_size, -1)
                        patch_size = max(1, int(round((token_count * drop_prob) ** 0.5)))
                        patch_size = min(patch_size, grid_size)
                        for row in range(tokens_2d.size(0)):
                            y = torch.randint(0, grid_size - patch_size + 1, (1,), device=noisy_image_features.device).item()
                            x = torch.randint(0, grid_size - patch_size + 1, (1,), device=noisy_image_features.device).item()
                            tokens_2d[row, y:y + patch_size, x:x + patch_size] = 0
                        noisy_image_features = tokens_2d.reshape_as(noisy_image_features)
                    else:
                        drop_count = max(1, int(round(token_count * drop_prob)))
                        for row in range(noisy_image_features.size(0)):
                            start = torch.randint(0, max(1, token_count - drop_count + 1), (1,), device=noisy_image_features.device).item()
                            noisy_image_features[row, start:start + drop_count] = 0

                outputs = forward_model(
                    model,
                    noisy_keypoints,
                    noisy_image_features,
                    raw_xyz=noisy_raw_xyz,
                    hand_detected=hand_detected,
                )
                logits, _, _, _ = parse_model_outputs(outputs)
                all_preds.append(logits.argmax(dim=1).cpu())
                all_labels.append(labels.cpu())

        labels = torch.cat(all_labels).numpy()
        predictions = torch.cat(all_preds).numpy()
        if "prediction_dir" in config:
            save_matched_predictions(config, name, test_dataset, label_encoder, labels, predictions)
        acc = accuracy_score(labels, predictions)
        results[name] = acc
        logging.info(f"Accuracy with noise '{name}': {acc:.4f}")

    return results


# --- [ Helper Functions - UNCHANGED ] ---
def occlude_points(data, num_occlusions=3):
    """(V18) Simulates occlusion by randomly setting keypoints to zero."""
    for i in range(data.shape[0]):
        occluded_indices = torch.randperm(data.shape[1])[:num_occlusions]
        data[i, occluded_indices, :] = 0
    return data


def occlude_points_pair(kpts_scaled, raw_xyz, num_occlusions=3):
    """Apply identical keypoint occlusions to scaled features and raw xyz."""
    for i in range(kpts_scaled.shape[0]):
        occluded_indices = torch.randperm(kpts_scaled.shape[1], device=kpts_scaled.device)[:num_occlusions]
        kpts_scaled[i, occluded_indices, :] = 0
        raw_xyz[i, occluded_indices, :] = 0
    return kpts_scaled, raw_xyz


def structured_noise(data, noise_level=0.8):
    """(V18) Simulates structured noise (e.g., affecting one finger more)."""
    finger_indices = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
    for i in range(data.shape[0]):
        finger_to_affect = finger_indices[torch.randint(0, 5, (1,)).item()]
        noise = torch.randn(len(finger_to_affect), data.shape[2], device=data.device) * noise_level
        if 0 <= min(finger_to_affect) and max(finger_to_affect) < data.shape[1]:
            data[i, finger_to_affect, :] += noise
    return data


def structured_noise_pair(kpts_scaled, raw_xyz, feature_noise_level=0.8, raw_noise_level=0.008):
    """Apply structured finger noise to both tensors with scale-appropriate magnitudes."""
    finger_indices = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
    for i in range(kpts_scaled.shape[0]):
        finger_to_affect = finger_indices[torch.randint(0, 5, (1,), device=kpts_scaled.device).item()]
        kpt_noise = torch.randn(len(finger_to_affect), kpts_scaled.shape[2], device=kpts_scaled.device) * feature_noise_level
        raw_noise = torch.randn(len(finger_to_affect), raw_xyz.shape[2], device=raw_xyz.device) * raw_noise_level
        kpts_scaled[i, finger_to_affect, :] += kpt_noise
        raw_xyz[i, finger_to_affect, :] += raw_noise
    return kpts_scaled, raw_xyz


def augment_spatial(inputs):
    """
    (V18 - Fixed) Applies rotation and scaling to (B, N, C) coordinates.
    Only augments the first 2 dimensions (x, y).
    """
    B, N, C = inputs.shape
    device = inputs.device

    xy_coords = inputs[..., :2]
    other_features = inputs[..., 2:]

    center = torch.mean(xy_coords, dim=1, keepdim=True)
    angles = (torch.rand(B, device=device) * 30 - 15) * (math.pi / 180.0)
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    rot_mats = torch.stack([torch.stack([cos_a, -sin_a], dim=1),
                            torch.stack([sin_a, cos_a], dim=1)], dim=1)
    xy_rotated = torch.einsum('bij,bkj->bki', rot_mats, xy_coords - center) + center

    scales = (torch.rand(B, 1, 1, device=device) * 0.2 + 0.9)
    xy_scaled = (xy_rotated - center) * scales + center

    return torch.cat([xy_scaled, other_features], dim=-1)


def augment_spatial_pair(kpts_scaled, raw_xyz):
    """
    Apply the same rotation/scale parameters to scaled model features and raw xyz.
    This keeps the model input and anatomical target geometrically aligned.
    """
    B, _, _ = kpts_scaled.shape
    device = kpts_scaled.device
    angles = (torch.rand(B, device=device) * 30 - 15) * (math.pi / 180.0)
    scales = torch.rand(B, 1, 1, device=device) * 0.2 + 0.9
    kpts_scaled_aug = _augment_spatial_with_params(kpts_scaled, angles, scales)
    raw_xyz_aug = _augment_spatial_with_params(raw_xyz, angles.to(raw_xyz.device), scales.to(raw_xyz.device))
    return kpts_scaled_aug, raw_xyz_aug


def _augment_spatial_with_params(inputs, angles, scales):
    B, _, _ = inputs.shape
    device = inputs.device
    xy_coords = inputs[..., :2]
    other_features = inputs[..., 2:]
    center = torch.mean(xy_coords, dim=1, keepdim=True)
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    rot_mats = torch.stack([torch.stack([cos_a, -sin_a], dim=1),
                            torch.stack([sin_a, cos_a], dim=1)], dim=1).to(device)
    xy_rotated = torch.einsum('bij,bkj->bki', rot_mats, xy_coords - center) + center
    xy_scaled = (xy_rotated - center) * scales.to(device) + center
    return torch.cat([xy_scaled, other_features], dim=-1)


def plot_history(history, save_path):
    """(V18 - Unchanged) Plots training and validation loss/accuracy curves."""
    try:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history['val_acc'], label='Validation Accuracy', color='orange')
        plt.title('Model Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Training history plot saved to {save_path}")
    except Exception as e:
        logging.warning(f"Could not save training plot: {e}")
# --- [ Helper Functions END ] ---
