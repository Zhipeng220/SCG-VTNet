# main.py
import torch
import logging
import time
import os
import sys
import joblib
import json
import numpy as np
import argparse
import random
import numpy as np
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from config import get_args
from utils import setup_logger, load_and_preprocess_train_data, load_and_preprocess_test_data
from model import MLPBaseline, AdvancedAttentionModel, HybridAttentionModel, ImageOnlyBaseline, VisualTokenTransformerBaseline
from losses import AnatomicalLoss, HierarchicalContrastiveLoss
from train import train_model, final_test_and_report, test_robustness, plot_history
from anat_metrics import collect_model_violation_rows, write_violation_report


MODEL_CLASS_REGISTRY = {
    "MLPBaseline": MLPBaseline,
    "AdvancedAttentionModel": AdvancedAttentionModel,
    "HybridAttentionModel": HybridAttentionModel,
    "ImageOnlyBaseline": ImageOnlyBaseline,
    "VisualTokenTransformerBaseline": VisualTokenTransformerBaseline,
}


def serialize_experiment_config(name, safe_name, config, args, input_dim, image_feature_shape):
    serializable = {}
    for key, value in config.items():
        if key == "model_class":
            serializable[key] = value.__name__
        else:
            serializable[key] = value
    serializable.update(
        {
            "experiment_name": name,
            "safe_name": safe_name,
            "input_dim": input_dim,
            "image_feature_shape": list(image_feature_shape),
            "image_feature_dim": image_feature_shape[-1],
            "visual_token_count": image_feature_shape[0] if len(image_feature_shape) == 2 else 1,
            "embed_dim": config.get("embed_dim", args.embed_dim),
            "num_layers": config.get("num_layers", args.num_layers),
            "num_heads": config.get("num_heads", args.num_heads),
            "dropout": args.dropout,
            "image_features_path": config.get("image_features_path", args.image_features_path),
            "aar_descriptor": config.get("aar_descriptor", "all"),
        }
    )
    return serializable


def load_experiment_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    model_class_name = config.get("model_class")
    if model_class_name not in MODEL_CLASS_REGISTRY:
        raise ValueError(f"Unknown model_class in experiment config: {model_class_name}")
    config["model_class"] = MODEL_CLASS_REGISTRY[model_class_name]
    return config


# --- [MODIFIED BY AI] ---
def define_experiments(args):
    """
    [MODIFIED FOR HYPERPARAMETER SWEEPS]
    Defines experiment lists based on the 'study_name' argument.
    """

    experiments = {}
    keypoints_desc = "21 keypoints"  # 默认

    # --- SOTA 模型的基线配置 (来自 Exp 10) ---
    sota_baseline_config = {
        "model_class": HybridAttentionModel,  #
        "use_advanced_features": True,  #
        "use_cross_attn": True,  #
        "use_gated_fusion": True,  #
        "use_gcn_branch": False,  # (已确认)
        "use_anatomical_loss": True,  #
        "use_contrastive_loss": True,  #
        "embed_dim": args.embed_dim,  #
        "num_layers": args.num_layers,  #
        "num_heads": args.num_heads,  #
        "contrastive_temperature": args.contrastive_temperature,  #
        "use_keypoint_refinement": True,
        "refine_scale": args.refine_scale,
        "anat_target": "refined_xyz",
        "anat_schedule": "cosine",
        # --- [ BUG FIX: 统一所有 Full/SOTA 模型的解剖学损失权重 ] ---
        "anatomical_weight_final": 0.001,
        # --- [ BUG FIX END ] ---
    }

    # --- 根据 study_name 选择要运行的实验 ---

    if args.study_name == 'ablation':
        logging.warning("study_name='ablation' is treated as 'core_ablation'.")
        args.study_name = 'core_ablation'

    if args.study_name == 'sota_only':
        study_title = "SOTA Model Stability Test"
        keypoints_desc = "SOTA Model Only (Exp 10)"
        experiments['Exp 10 (Full SOTA)'] = {
            **sota_baseline_config,
        }

    elif args.study_name == 'wo_aar_only':
        study_title = "w/o AAR Only"
        keypoints_desc = "AC-GCAT without anatomy-aware representation"
        experiments["w/o AAR"] = {
            **sota_baseline_config,
            "use_advanced_features": False,
            "aar_descriptor": "none",
        }

    elif args.study_name == 'visual_modulation':
        study_title = "Configurations Exploration of Visual Modulation"
        keypoints_desc = "AAR fixed; CE-only; varying global/token visual context and fusion"

        global_path = args.global_image_features_path
        token_path = args.token_image_features_path or args.image_features_path
        if global_path is None:
            raise ValueError("--global_image_features_path is required for study_name=visual_modulation")

        base = {
            **sota_baseline_config,
            "use_advanced_features": True,
            "aar_descriptor": "all",
            "use_anatomical_loss": False,
            "use_contrastive_loss": False,
            "use_keypoint_refinement": False,
            "anat_target": "none",
        }
        experiments["A"] = {
            **base,
            "image_features_path": global_path,
            "visual_feature": "global",
            "fusion": "concat",
            "gate": "none",
            "use_cross_attn": False,
            "use_gated_fusion": False,
        }
        experiments["B"] = {
            **base,
            "image_features_path": global_path,
            "visual_feature": "global",
            "fusion": "CA",
            "gate": "none",
            "use_cross_attn": True,
            "use_gated_fusion": False,
        }
        experiments["C"] = {
            **base,
            "image_features_path": token_path,
            "visual_feature": "token",
            "fusion": "concat",
            "gate": "none",
            "use_cross_attn": False,
            "use_gated_fusion": False,
        }
        experiments["D"] = {
            **base,
            "image_features_path": token_path,
            "visual_feature": "token",
            "fusion": "CA",
            "gate": "none",
            "use_cross_attn": True,
            "use_gated_fusion": False,
        }
        experiments["E"] = {
            **base,
            "image_features_path": token_path,
            "visual_feature": "token",
            "fusion": "CA",
            "gate": "global",
            "use_cross_attn": True,
            "use_gated_fusion": True,
        }

    elif args.study_name == 'aar_construction':
        study_title = "Construction of Anatomy-Aware Representation"
        keypoints_desc = "Fixed SVM+AGG; CE-only; varying skeleton descriptors"

        token_path = args.token_image_features_path or args.image_features_path
        base = {
            **sota_baseline_config,
            "image_features_path": token_path,
            "use_cross_attn": True,
            "use_gated_fusion": True,
            "use_anatomical_loss": False,
            "use_contrastive_loss": False,
            "use_keypoint_refinement": False,
            "anat_target": "none",
        }
        experiments["3D"] = {
            **base,
            "use_advanced_features": False,
            "aar_descriptor": "none",
            "descriptor": "coord",
        }
        experiments["BV"] = {
            **base,
            "use_advanced_features": True,
            "aar_descriptor": "bone",
            "descriptor": "coord + bone",
        }
        experiments["JA"] = {
            **base,
            "use_advanced_features": True,
            "aar_descriptor": "angle",
            "descriptor": "coord + angle",
        }
        experiments["TD"] = {
            **base,
            "use_advanced_features": True,
            "aar_descriptor": "tip",
            "descriptor": "coord + tip-dist",
        }
        experiments["BV+JA"] = {
            **base,
            "use_advanced_features": True,
            "aar_descriptor": "bone_angle",
            "descriptor": "coord + bone + angle",
        }
        experiments["AAR"] = {
            **base,
            "use_advanced_features": True,
            "aar_descriptor": "all",
            "descriptor": "coord + bone + angle + tip-dist",
        }

    elif args.study_name == 'objective_construction':
        study_title = "Construction of the Constraint-Aware Objective"
        keypoints_desc = "Fixed AAR+SVM+AGG; varying objective terms"

        token_path = args.token_image_features_path or args.image_features_path
        base = {
            **sota_baseline_config,
            "image_features_path": token_path,
            "use_advanced_features": True,
            "aar_descriptor": "all",
            "use_cross_attn": True,
            "use_gated_fusion": True,
            "use_anatomical_loss": False,
            "use_contrastive_loss": False,
            "use_keypoint_refinement": False,
            "anat_target": "none",
            "anat_schedule": "constant",
            "anatomical_weight_final": 0.001,
        }
        experiments["CE"] = {
            **base,
            "objective": "CE",
        }
        experiments["SCL"] = {
            **base,
            "objective": "CE + SCL",
            "use_contrastive_loss": True,
        }
        experiments["Raw-AC"] = {
            **base,
            "objective": "CE + raw AC",
            "use_anatomical_loss": True,
            "use_keypoint_refinement": False,
            "anat_target": "raw_input",
            "anat_schedule": "constant",
        }
        experiments["Ref-AC"] = {
            **base,
            "objective": "CE + refined AC",
            "use_anatomical_loss": True,
            "use_keypoint_refinement": True,
            "anat_target": "refined_xyz",
            "anat_schedule": "constant",
        }
        experiments["Dyn-AC"] = {
            **base,
            "objective": "CE + dynamic refined AC",
            "use_anatomical_loss": True,
            "use_keypoint_refinement": True,
            "anat_target": "refined_xyz",
            "anat_schedule": "cosine",
        }
        experiments["Dyn-AC+SCL"] = {
            **base,
            "objective": "CE + dynamic refined AC + SCL",
            "use_anatomical_loss": True,
            "use_contrastive_loss": True,
            "use_keypoint_refinement": True,
            "anat_target": "refined_xyz",
            "anat_schedule": "cosine",
        }

    # 1. 核心消融实验 (来自 LaTeX a)
    elif args.study_name == 'fair_e1_e7':
        study_title = "Fair E1-E7 Comparison"
        keypoints_desc = "same split/seed/epochs; frozen visual features unless exact feature paths are provided"

        processed_dir = os.path.dirname(args.image_features_path)
        resnet_global_path = args.global_image_features_path or os.path.join(
            processed_dir, "zerofill_visual_global_from_tokens.npy"
        )
        resnet_token_path = args.token_image_features_path or args.image_features_path

        selected_method = getattr(args, "baseline_method", None) or os.environ.get("SCG_BLOCKED_BASELINE")
        efficientnet_path = args.efficientnet_features_path
        vit_path = args.vit_features_path
        missing = []
        if not efficientnet_path and selected_method in (None, "E2"):
            missing.append("--efficientnet_features_path")
        if not vit_path and selected_method in (None, "E3"):
            missing.append("--vit_features_path")
        if missing and not args.allow_feature_proxy:
            raise ValueError(
                "fair_e1_e7 needs exact feature files for E2/E3: "
                + ", ".join(missing)
                + ". Re-run with --allow_feature_proxy only for a smoke/proxy run."
            )
        efficientnet_path = efficientnet_path or resnet_global_path
        vit_path = vit_path or resnet_token_path

        base_ce = {
            "use_advanced_features": True,
            "use_anatomical_loss": False,
            "use_contrastive_loss": False,
            "use_keypoint_refinement": False,
            "anat_target": "none",
            "modality_dropout": 0.0,
        }

        experiments["E1 ResNet-50"] = {
            **base_ce,
            "model_class": ImageOnlyBaseline,
            "image_features_path": resnet_global_path,
            "method_id": "E1",
            "visual_backbone": "ResNet-50 frozen global features",
        }
        experiments["E2 EfficientNet-B0"] = {
            **base_ce,
            "model_class": ImageOnlyBaseline,
            "image_features_path": efficientnet_path,
            "method_id": "E2",
            "visual_backbone": "EfficientNet-B0 features" if args.efficientnet_features_path else "proxy: ResNet-50 global features",
        }
        experiments["E3 ViT-S/16"] = {
            **base_ce,
            "model_class": VisualTokenTransformerBaseline,
            "image_features_path": vit_path,
            "method_id": "E3",
            "visual_backbone": "ViT-S/16 features" if args.vit_features_path else "proxy: ResNet-50 layer4 tokens",
        }
        experiments["E4 HAN-static"] = {
            **base_ce,
            "model_class": AdvancedAttentionModel,
            "image_features_path": resnet_global_path,
            "method_id": "E4",
            "skeleton_encoder": "HAN/static skeleton transformer on 21 hand joints",
        }
        experiments["E5 Late Fusion"] = {
            **base_ce,
            "model_class": HybridAttentionModel,
            "image_features_path": resnet_global_path,
            "use_cross_attn": False,
            "use_gated_fusion": False,
            "use_gcn_branch": False,
            "method_id": "E5",
            "fusion": "feature concatenation",
        }
        experiments["E6 ModDrop Fusion"] = {
            **base_ce,
            "model_class": HybridAttentionModel,
            "image_features_path": resnet_global_path,
            "use_cross_attn": False,
            "use_gated_fusion": False,
            "use_gcn_branch": False,
            "modality_dropout": max(args.modality_dropout, 0.25),
            "method_id": "E6",
            "fusion": "feature concatenation + modality dropout",
        }
        experiments["E7 AC-GCAT"] = {
            **sota_baseline_config,
            "image_features_path": resnet_token_path,
            "method_id": "E7",
            "fusion": "skeleton-conditioned gated cross-attention",
        }
        if selected_method:
            experiments = {n: conf for n, conf in experiments.items() if conf.get("method_id") == selected_method}
            if len(experiments) != 1:
                raise ValueError(f"Expected one baseline for {selected_method}")

    elif args.study_name == 'asl_matched':
        if args.frozen_manifest is None or args.matched_variant is None:
            raise ValueError("ASL matched training requires a frozen manifest and one variant")
        study_title = "ASL matched component comparison"
        keypoints_desc = "AAR with 49-token visual context"
        changes = {
            "F": {},
            "A0": {"use_anatomical_loss": False, "anat_target": "none", "anatomical_weight_final": 0.0},
            "G0": {"use_gated_fusion": False},
            "Uniform": {"uniform_visual_attention": True},
        }
        experiments[args.matched_variant] = {**sota_baseline_config, **changes[args.matched_variant]}

    elif args.study_name == 'core_ablation':
        study_title = "Core Component Ablation Study"
        keypoints_desc = "w/o-style ablation of AC-GCAT"

        full_config = {
            **sota_baseline_config,
        }
        experiments["w/o AAR"] = {
            **full_config,
            "use_advanced_features": False,
            "aar_descriptor": "none",
        }
        experiments["Full AC-GCAT"] = {**full_config}
        experiments["w/o Refined anatomical loss"] = {
            **full_config,
            "use_anatomical_loss": False,
            "use_keypoint_refinement": False,
            "anat_target": "none",
        }
        experiments["w/o Contrastive loss"] = {
            **full_config,
            "use_contrastive_loss": False,
        }
        experiments["w/o Both auxiliary losses"] = {
            **full_config,
            "use_anatomical_loss": False,
            "use_contrastive_loss": False,
            "use_keypoint_refinement": False,
            "anat_target": "none",
        }
        experiments["w/o Global gate"] = {
            **full_config,
            "use_gated_fusion": False,
        }
        experiments["w/o Cross-attention"] = {
            **full_config,
            "use_cross_attn": False,
            "use_gated_fusion": False,
        }

    elif args.study_name == 'hyperparam_embed_dim':
        study_title = "Hyperparameter Search: Embedding Dimension"
        values_to_test = [128, 256, 384, 512]
        for dim in values_to_test:
            if dim % args.num_heads != 0:
                logging.warning(f"Skipping embed_dim={dim}; not divisible by num_heads={args.num_heads}")
                continue
            exp_name = f"EmbedDim_{dim}"
            experiments[exp_name] = {
                **sota_baseline_config,
                "embed_dim": dim,  # 覆盖参数
            }

    # 2. 注意力层数实验 (来自 LaTeX b)
    elif args.study_name == 'hyperparam_num_layers':
        study_title = "Hyperparameter Search: Number of Layers"
        values_to_test = [2, 4, 6, 8]
        for layers in values_to_test:
            exp_name = f"NumLayers_{layers}"
            experiments[exp_name] = {
                **sota_baseline_config,
                "num_layers": layers,  # 覆盖参数
            }

    # 3. 注意力头数实验 (来自 LaTeX c)
    elif args.study_name == 'hyperparam_num_heads':
        study_title = "Hyperparameter Search: Number of Heads"
        values_to_test = [4, 8, 16]
        for heads in values_to_test:
            if args.embed_dim % heads != 0:
                logging.warning(f"Skipping num_heads={heads}; embed_dim={args.embed_dim} not divisible by it.")
                continue
            exp_name = f"NumHeads_{heads}"
            experiments[exp_name] = {
                **sota_baseline_config,
                "num_heads": heads,  # 覆盖参数
            }

    # 4. 对比损失温度实验 (来自 LaTeX d)
    elif args.study_name == 'hyperparam_cont_temp':
        study_title = "Hyperparameter Search: Contrastive Temperature"
        values_to_test = [0.05, 0.07, 0.1, 0.2, 0.5]
        for temp in values_to_test:
            exp_name = f"ContTemp_{temp:.2f}"
            experiments[exp_name] = {
                **sota_baseline_config,
                "contrastive_temperature": temp,  # 覆盖参数
            }

    # 5. 运行完整的15个实验
    elif args.study_name == 'full_ablation':
        study_title = "Full 15-Experiment Ablation Study"
        keypoints_desc = "21 keypoints (Basic 3D / Advanced 11D)"

        # --- 10个线性实验 ---
        experiments['Exp 1 (MLP Basic)'] = {"model_class": MLPBaseline, "use_advanced_features": False,
                                            "use_anatomical_loss": False, "use_contrastive_loss": False}
        experiments['Exp 2 (MLP Adv)'] = {"model_class": MLPBaseline, "use_advanced_features": True,
                                          "use_anatomical_loss": False, "use_contrastive_loss": False}
        experiments['Exp 3 (Attn Adv)'] = {"model_class": AdvancedAttentionModel, "use_advanced_features": True,
                                           "use_anatomical_loss": False, "use_contrastive_loss": False}
        experiments['Exp 4 (Img Only)'] = {"model_class": ImageOnlyBaseline, "use_advanced_features": True,
                                           "use_anatomical_loss": False, "use_contrastive_loss": False}
        experiments['Exp 5 (Late Fusion)'] = {"model_class": HybridAttentionModel, "use_advanced_features": True,
                                              "use_cross_attn": False, "use_gated_fusion": False,
                                              "use_anatomical_loss": False, "use_contrastive_loss": False}
        experiments['Exp 6 (CrossAttn no Gate)'] = {"model_class": HybridAttentionModel, "use_advanced_features": True,
                                                    "use_cross_attn": True, "use_gated_fusion": False,
                                                    "use_anatomical_loss": False, "use_contrastive_loss": False}
        experiments['Exp 7 (SOTA Arch + CE)'] = {
            **sota_baseline_config, "use_anatomical_loss": False,
            "use_contrastive_loss": False, "use_keypoint_refinement": False,
            "anat_target": "none"
        }
        experiments['Exp 8 (SOTA Arch + CE/Anat)'] = {
            **sota_baseline_config, "use_contrastive_loss": False,
            "use_keypoint_refinement": True, "anat_target": "refined_xyz"
        }
        experiments['Exp 9 (SOTA Arch + CE/Cont)'] = {
            **sota_baseline_config, "use_anatomical_loss": False,
            "use_keypoint_refinement": False, "anat_target": "none"
        }
        experiments['Exp 10 (Full SOTA)'] = {
            **sota_baseline_config,
        }

        # --- 5个新的组合实验 ---
        experiments['New-1 (Attn Basic + CE)'] = {"model_class": AdvancedAttentionModel, "use_advanced_features": False,
                                                  "use_anatomical_loss": False, "use_contrastive_loss": False}
        experiments['New-3 (Hybrid Basic + CE)'] = {"model_class": HybridAttentionModel, "use_advanced_features": False,
                                                    "use_cross_attn": True, "use_gated_fusion": True,
                                                    "use_anatomical_loss": False, "use_contrastive_loss": False}
        experiments['New-4 (Attn Basic + CE/Cont)'] = {"model_class": AdvancedAttentionModel,
                                                       "use_advanced_features": False, "use_anatomical_loss": False,
                                                       "use_contrastive_loss": True, "anat_target": "none"}
        experiments['New-6 (Attn Adv + CE/Cont)'] = {"model_class": AdvancedAttentionModel,
                                                     "use_advanced_features": True, "use_anatomical_loss": False,
                                                     "use_contrastive_loss": True, "anat_target": "none"}
        experiments['New-7 (Hybrid Basic + Aux Losses)'] = {"model_class": HybridAttentionModel,
                                                            "use_advanced_features": False, "use_cross_attn": True,
                                                            "use_gated_fusion": True, "use_anatomical_loss": True,
                                                            "use_contrastive_loss": True,
                                                            "use_keypoint_refinement": True,
                                                            "anat_target": "refined_xyz"}

    elif args.study_name == 'hyperparam_anat':
        study_title = "Hyperparameter Search: Anatomical Loss Weight"
        keypoints_desc = "SOTA Model, varying Anat. Weight"
        values_to_test = [0.0, 0.001, 0.002, 0.005, 0.01]
        for w in values_to_test:
            exp_name = f"AnatW_{w}"
            experiments[exp_name] = {
                **sota_baseline_config, "anatomical_weight_final": w,
            }  # 覆盖

    elif args.study_name == 'hyperparam_contrastive':
        study_title = "Hyperparameter Search: Contrastive Loss Weight"
        keypoints_desc = "SOTA Model, varying Cont. Weight"
        values_to_test = [0.0, 0.05, 0.1, 0.15, 0.2]
        for w in values_to_test:
            exp_name = f"ContW_{w}"
            experiments[exp_name] = {**sota_baseline_config, "contrastive_weight": w}  # 覆盖

    elif args.study_name == 'anat_mechanism':
        study_title = "Anatomical Loss Mechanism Study"
        keypoints_desc = "SOTA Model, anatomical target ablation"
        experiments['CE Only'] = {**sota_baseline_config, "use_anatomical_loss": False,
                                  "use_contrastive_loss": False, "anat_target": "none",
                                  "use_keypoint_refinement": False}
        experiments['CE + Cont'] = {**sota_baseline_config, "use_anatomical_loss": False,
                                    "use_contrastive_loss": True, "anat_target": "none",
                                    "use_keypoint_refinement": False}
        experiments['CE + Cont + Raw-input Anat'] = {
            **sota_baseline_config, "use_anatomical_loss": True, "use_contrastive_loss": True,
            "anat_target": "raw_input", "use_keypoint_refinement": False,
            "anat_schedule": "constant", "anatomical_weight_final": 0.02
        }
        experiments['CE + Cont + Refined Anat Constant'] = {
            **sota_baseline_config, "use_anatomical_loss": True, "use_contrastive_loss": True,
            "anat_target": "refined_xyz", "use_keypoint_refinement": True,
            "anat_schedule": "constant", "anatomical_weight_final": 0.02
        }
        experiments['CE + Cont + Refined Anat Cosine'] = {
            **sota_baseline_config, "use_anatomical_loss": True, "use_contrastive_loss": True,
            "anat_target": "refined_xyz", "use_keypoint_refinement": True,
            "anat_schedule": "cosine", "anatomical_weight_final": 0.02
        }
        experiments['CE + Refined Anat'] = {
            **sota_baseline_config, "use_anatomical_loss": True, "use_contrastive_loss": False,
            "anat_target": "refined_xyz", "use_keypoint_refinement": True,
            "anat_schedule": "constant", "anatomical_weight_final": 0.02
        }

    else:
        logging.warning(f"未知的 'study_name': {args.study_name}. 运行默认的 'full_ablation'.")
        args.study_name = 'full_ablation'
        return define_experiments(args)

    logging.info(f"--- 准备运行 {len(experiments)} 个独特的实验: {study_title} ---")
    return study_title, experiments, keypoints_desc


# ... (main 函数和文件的其余部分保持不变) ...

def main(args):
    """Main function to orchestrate training or testing."""
    output_dir = args.output_dir
    paths = {
        'base': output_dir,
        'logs': os.path.join(output_dir, 'logs'),
        'weights': os.path.join(output_dir, 'weights'),
        'plots': os.path.join(output_dir, 'plots'),
        'cm': os.path.join(output_dir, 'confusion_matrix')
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    setup_logger(log_dir=paths['logs'])
    start_main_time = time.time()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    logging.info(f"Using random seed: {args.seed}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"All outputs will be saved to: {output_dir}")

    if args.mode == 'train':
        study_title, experiments, keypoints_desc = define_experiments(args)
        logging.info(f"\nStarting Study: {study_title}")

        final_results, robustness_results, anat_violation_results = {}, {}, {}
        anat_violation_rows = []
        default_scaler, default_label_encoder = None, None

        for name, config in experiments.items():
            logging.info(f"\n{'=' * 25} Running Experiment: {name} {'=' * 25}")

            safe_name = name.replace(' ', '_').replace('(', '_').replace(')', '').replace('=', '').replace(',',
                                                                                                           '').replace(
                '/', '_').replace('.', 'p')

            use_advanced_features = config.get("use_advanced_features", True)
            aar_descriptor = config.get("aar_descriptor", args.aar_descriptor)
            exp_image_features_path = config.get("image_features_path", args.image_features_path)
            logging.info(f"Config: use_advanced_features = {use_advanced_features}, aar_descriptor = {aar_descriptor}")
            logging.info(f"Config: image_features_path = {exp_image_features_path}")

            datasets, current_scaler, current_label_encoder, train_sampler, input_dim = load_and_preprocess_train_data(
                args.train_csv_path, exp_image_features_path,
                use_advanced_features=use_advanced_features,
                seed=args.seed,
                aar_descriptor=aar_descriptor,
                frozen_manifest=args.frozen_manifest,
            )

            per_exp_scaler_path = os.path.join(output_dir, f"scaler_{safe_name}.pkl")
            per_exp_encoder_path = os.path.join(output_dir, f"label_encoder_{safe_name}.pkl")
            joblib.dump(current_scaler, per_exp_scaler_path)
            joblib.dump(current_label_encoder, per_exp_encoder_path)
            logging.info(f"Saved experiment preprocessing tools to {per_exp_scaler_path} and {per_exp_encoder_path}")

            if use_advanced_features and default_scaler is None:
                default_scaler = current_scaler
                default_label_encoder = current_label_encoder

            pin_memory = device.type == 'cuda'
            train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, sampler=train_sampler,
                                      num_workers=0, pin_memory=pin_memory)
            val_loader = DataLoader(datasets['val'], batch_size=args.batch_size * 2, shuffle=False, num_workers=0,
                                    pin_memory=pin_memory)
            test_loader = DataLoader(datasets['test'], batch_size=args.batch_size * 2, shuffle=False, num_workers=0,
                                     pin_memory=pin_memory)

            ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            anatomical_criterion = AnatomicalLoss(
                device=device,
                angle_weight=config.get("anat_angle_weight", args.anat_angle_weight),
                length_weight=config.get("anat_length_weight", args.anat_length_weight),
                planarity_weight=config.get("anat_planarity_weight", args.anat_planarity_weight),
            )
            current_cont_temp = config.get("contrastive_temperature", args.contrastive_temperature)
            logging.info(f"Config: contrastive_temperature = {current_cont_temp}")
            contrastive_criterion = HierarchicalContrastiveLoss(temperature=current_cont_temp)

            model_class = config["model_class"]
            num_classes = len(current_label_encoder.classes_)

            current_embed_dim = config.get("embed_dim", args.embed_dim)
            current_num_layers = config.get("num_layers", args.num_layers)
            current_num_heads = config.get("num_heads", args.num_heads)
            logging.info(
                f"Config: embed_dim={current_embed_dim}, num_layers={current_num_layers}, num_heads={current_num_heads}")

            IMAGE_FEATURE_DIM = datasets['train'].image_feature_dim
            image_feature_shape = tuple(datasets['train'].image_feature_shape[1:])
            logging.info(f"Detected image feature shape: {image_feature_shape}")
            visual_token_count = image_feature_shape[0] if len(image_feature_shape) == 2 else 1
            config["visual_token_count"] = visual_token_count

            experiment_config = serialize_experiment_config(
                name, safe_name, config, args, input_dim, image_feature_shape
            )
            config_path = os.path.join(output_dir, f"config_{safe_name}.json")
            with open(config_path, "w", encoding="utf-8") as config_file:
                json.dump(experiment_config, config_file, indent=2, ensure_ascii=False)
            logging.info(f"Saved experiment config to {config_path}")

            model_kwargs = {"input_dim": input_dim, "num_classes": num_classes,
                            "num_layers": current_num_layers,
                            "num_heads": current_num_heads,
                            "embed_dim": current_embed_dim,
                            "dropout": args.dropout}

            if model_class == MLPBaseline:
                flat_kpt_dim = 21 * input_dim
                model = MLPBaseline(input_size=flat_kpt_dim, num_classes=num_classes)
            elif model_class == AdvancedAttentionModel:
                model = AdvancedAttentionModel(**model_kwargs)
            elif model_class == ImageOnlyBaseline:
                model = ImageOnlyBaseline(image_feature_dim=IMAGE_FEATURE_DIM, num_classes=num_classes,
                                          dropout=args.dropout)
            elif model_class == VisualTokenTransformerBaseline:
                model = VisualTokenTransformerBaseline(
                    image_feature_dim=IMAGE_FEATURE_DIM,
                    num_classes=num_classes,
                    visual_token_count=visual_token_count,
                    num_layers=current_num_layers,
                    num_heads=current_num_heads,
                    embed_dim=current_embed_dim,
                    dropout=args.dropout,
                )
            elif model_class == HybridAttentionModel:
                model = HybridAttentionModel(
                    **model_kwargs,
                    image_feature_dim=IMAGE_FEATURE_DIM,
                    use_gated_fusion=config.get("use_gated_fusion", False),
                    use_dynamic_graph=config.get("use_dynamic_graph", False),
                    use_cross_attn=config.get("use_cross_attn", True),
                    use_gcn_branch=config.get("use_gcn_branch", False),
                    use_keypoint_refinement=config.get("use_keypoint_refinement", args.use_keypoint_refinement),
                    refine_scale=config.get("refine_scale", args.refine_scale),
                    uniform_visual_attention=config.get("uniform_visual_attention", False),
                    visual_token_count=visual_token_count,
                )

            logging.info(
                f"Instantiated model: {model.__class__.__name__} with keypoint_dim: {input_dim}, embed_dim: {current_embed_dim}")

            if args.study_name == "asl_matched" or (args.study_name == "fair_e1_e7" and args.frozen_manifest is not None):
                if args.study_name == "asl_matched" and image_feature_shape != (49, 2048):
                    raise ValueError("Matched ASL configurations require 49 x 2048 visual tokens")
                import hashlib
                from pathlib import Path
                unused_params = 0
                if getattr(model, "uniform_visual_attention", False):
                    width = model.embed_dim
                    unused_params = (sum(p.numel() for p in model.img_k_proj.parameters())
                                     + sum(p.numel() for p in model.cross_attn_norm_kpt.parameters())
                                     + 2 * width * (width + 1))
                experiment_config.update({
                    "train_args": vars(args),
                    "frozen_manifest_sha256": hashlib.sha256(Path(args.frozen_manifest).read_bytes()).hexdigest(),
                    "device": str(device), "torch_version": torch.__version__,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "unused_query_key_parameters": unused_params,
                })
                with open(config_path, "w", encoding="utf-8") as config_file:
                    json.dump(experiment_config, config_file, indent=2)
                config["prediction_dir"] = output_dir
                config["perturbation_seed"] = args.perturbation_seed
            optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            total_steps = args.epochs * len(train_loader)
            if total_steps == 0:
                logging.warning("Train loader is empty! Skipping training.")
                continue
            scheduler = OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=total_steps, pct_start=0.25)

            model_path = os.path.join(paths['weights'], f"best_model_{safe_name}_21kpts.pth")
            start_time = time.time()

            current_anat_weight_final = config.get("anatomical_weight_final", args.anatomical_weight_final)
            current_contrastive_weight = config.get("contrastive_weight", args.contrastive_weight)

            train_args = argparse.Namespace(**vars(args))
            train_args.anatomical_weight_final = current_anat_weight_final
            train_args.anat_target = config.get("anat_target", args.anat_target)
            train_args.anat_schedule = config.get("anat_schedule", args.anat_schedule)
            train_args.modality_dropout = config.get("modality_dropout", args.modality_dropout)

            logging.info(
                f"Config: anat_weight_final={current_anat_weight_final}, anat_target={train_args.anat_target}, "
                f"anat_schedule={train_args.anat_schedule}, contrastive_weight={current_contrastive_weight}, "
                f"modality_dropout={train_args.modality_dropout}")

            history = train_model(model, model_path, train_loader, val_loader, optimizer, scheduler, ce_criterion,
                                  anatomical_criterion if config.get("use_anatomical_loss") else None,
                                  contrastive_criterion if config.get("use_contrastive_loss") else None,
                                  train_args, device,
                                  contrastive_weight=current_contrastive_weight)

            if args.study_name == "asl_matched" or (args.study_name == "fair_e1_e7" and args.frozen_manifest is not None):
                selected_epoch = 1 + int(np.argmax(history["val_acc"]))
                with open(os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8") as stream:
                    json.dump({**history, "selected_epoch": selected_epoch}, stream, indent=2)
            plot_path = os.path.join(paths['plots'], f"history_{safe_name}.png")
            plot_history(history, plot_path)

            test_metrics = final_test_and_report(model_class, model_path, test_loader, device, current_label_encoder, args,
                                                 input_dim, config, output_paths={'cm_dir': paths['cm']},
                                                 image_feature_dim=IMAGE_FEATURE_DIM)
            if isinstance(test_metrics, dict):
                final_results[name] = {**test_metrics, 'train_time': time.time() - start_time}
            else:
                final_results[name] = {'accuracy': test_metrics, 'macro_f1': 0.0,
                                       'train_time': time.time() - start_time}

            if model_class == HybridAttentionModel:
                try:
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    rows = collect_model_violation_rows(model, test_loader, device, name)
                    anat_violation_rows.extend(rows)
                    refined_rows = [r for r in rows if r["variant"].endswith(":refined_xyz")]
                    selected = refined_rows[0] if refined_rows else rows[0]
                    anat_violation_results[name] = selected["total_anat_violation_score"]
                except Exception as e:
                    logging.warning(f"Could not compute anatomical violation metrics for {name}: {e}")

            robust_res = test_robustness(model_class, model_path, datasets['test'], device, current_label_encoder, args,
                                         input_dim, config, image_feature_dim=IMAGE_FEATURE_DIM)
            if robust_res:
                robustness_results[name] = robust_res

        if default_scaler is not None and default_label_encoder is not None:
            joblib.dump(default_scaler, os.path.join(output_dir, 'scaler_21kpts.pkl'))
            joblib.dump(default_label_encoder, os.path.join(output_dir, 'label_encoder_21kpts.pkl'))
            logging.info(f"\nDefault advanced-feature preprocessing tools saved to: {output_dir}")

        if anat_violation_rows:
            anat_report_path = os.path.join(output_dir, "anat_violation_report.csv")
            write_violation_report(anat_violation_rows, anat_report_path)
            logging.info(f"Anatomical violation report saved to {anat_report_path}")

        summary_path = os.path.join(output_dir,
                                    f"summary_report_{args.study_name}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        with open(summary_path, 'w') as f:
            f.write(f"{'=' * (len(study_title) + 4)}\n")
            f.write(f"  {study_title} Summary ({keypoints_desc})\n")
            f.write(f"{'=' * (len(study_title) + 4)}\n\n")
            header = ("Experiment Name".ljust(40) +
                      "| Clean Acc    | Macro F1     | Gaussian 1.0 | Occlusion_3  | Structured   | Visual Gauss | Token Drop   | Patch Occ    | Joint Noise  | Anat Violation | Train Time (s) ")
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            sorted_results = sorted(final_results.items(), key=lambda item: item[1]['accuracy'], reverse=True)

            for name, results in sorted_results:
                acc_str = f"{results['accuracy'] * 100:.2f}%".ljust(12)
                f1_str = f"{results.get('macro_f1', 0.0) * 100:.2f}%".ljust(12)
                time_str = f"{results['train_time']:.2f}s".ljust(14)
                gaussian_str = "N/A".ljust(12)
                occlusion_str = "N/A".ljust(12)
                structured_str = "N/A".ljust(12)
                visual_gauss_str = "N/A".ljust(12)
                token_drop_str = "N/A".ljust(12)
                patch_occ_str = "N/A".ljust(12)
                joint_noise_str = "N/A".ljust(12)
                if name in robustness_results:
                    if 'gaussian_1.0' in robustness_results[name]:
                        gaussian_str = f"{robustness_results[name]['gaussian_1.0'] * 100:.2f}%".ljust(12)
                    if 'occlusion_3' in robustness_results[name]:
                        occlusion_str = f"{robustness_results[name]['occlusion_3'] * 100:.2f}%".ljust(12)
                    if 'structured_finger' in robustness_results[name]:
                        structured_str = f"{robustness_results[name]['structured_finger'] * 100:.2f}%".ljust(12)
                    if 'visual_gaussian' in robustness_results[name]:
                        visual_gauss_str = f"{robustness_results[name]['visual_gaussian'] * 100:.2f}%".ljust(12)
                    if 'visual_token_dropout' in robustness_results[name]:
                        token_drop_str = f"{robustness_results[name]['visual_token_dropout'] * 100:.2f}%".ljust(12)
                    if 'visual_patch_occlusion' in robustness_results[name]:
                        patch_occ_str = f"{robustness_results[name]['visual_patch_occlusion'] * 100:.2f}%".ljust(12)
                    if 'skeleton_visual_joint_noise' in robustness_results[name]:
                        joint_noise_str = f"{robustness_results[name]['skeleton_visual_joint_noise'] * 100:.2f}%".ljust(12)
                anat_str = "N/A".ljust(14)
                if name in anat_violation_results:
                    anat_str = f"{anat_violation_results[name]:.6f}".ljust(14)
                f.write(
                    f"{name.ljust(40)}| {acc_str} | {f1_str} | {gaussian_str} | "
                    f"{occlusion_str} | {structured_str} | {visual_gauss_str} | "
                    f"{token_drop_str} | {patch_occ_str} | {joint_noise_str} | {anat_str} | {time_str}\n"
                )
            f.write("=" * len(header) + "\n")
        logging.info(f"Summary report saved to {summary_path}")

    elif args.mode == 'test':
        if not all([args.test_csv_path, args.image_features_path, args.model_path, args.output_dir]):
            logging.error(
                "FATAL: For 'test' mode, --test_csv_path, --image_features_path, --model_path, and --output_dir are required.");
            sys.exit(1)

        experiment_config = None
        if args.experiment_config:
            try:
                experiment_config = load_experiment_config(args.experiment_config)
                logging.info(f"Loaded experiment config from {args.experiment_config}")
            except Exception as e:
                logging.error(f"FATAL: Could not load experiment config {args.experiment_config}: {e}")
                sys.exit(1)

        try:
            if experiment_config and experiment_config.get("safe_name"):
                scaler_path = os.path.join(args.output_dir, f"scaler_{experiment_config['safe_name']}.pkl")
                encoder_path = os.path.join(args.output_dir, f"label_encoder_{experiment_config['safe_name']}.pkl")
            else:
                scaler_path = os.path.join(args.output_dir, 'scaler_21kpts.pkl')
                encoder_path = os.path.join(args.output_dir, 'label_encoder_21kpts.pkl')
            scaler = joblib.load(scaler_path)
            label_encoder = joblib.load(encoder_path)
            logging.info("Loaded pre-fitted scaler and label encoder.")
        except FileNotFoundError:
            logging.error(
                f"FATAL: scaler ({scaler_path}) or label_encoder ({encoder_path}) not found in {args.output_dir}. Run in 'train' mode first.");
            sys.exit(1)

        use_advanced_features = experiment_config.get("use_advanced_features", True) if experiment_config else True
        aar_descriptor = experiment_config.get("aar_descriptor", args.aar_descriptor) if experiment_config else args.aar_descriptor
        exp_image_features_path = experiment_config.get("image_features_path", args.image_features_path) if experiment_config else args.image_features_path
        if not experiment_config:
            logging.warning("No --experiment_config supplied; assuming ADVANCED features and legacy model mapping.")
        test_dataset = load_and_preprocess_test_data(
            args.test_csv_path, exp_image_features_path, scaler, label_encoder,
            use_advanced_features=use_advanced_features,
            aar_descriptor=aar_descriptor,
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)

        if experiment_config:
            model_class = experiment_config["model_class"]
            config = experiment_config
        else:
            model_mapping = {
                'MLPBaseline': (MLPBaseline, {}),
                'AdvancedAttentionModel': (AdvancedAttentionModel, {}),
                'HybridAttentionModel': (HybridAttentionModel, {
                    'use_gated_fusion': True, 'use_dynamic_graph': False,
                    'use_cross_attn': True, 'use_gcn_branch': False,
                    'use_keypoint_refinement': args.use_keypoint_refinement,
                    'refine_scale': args.refine_scale,
                    'visual_token_count': test_dataset.image_feature_shape[1] if len(test_dataset.image_feature_shape) == 3 else 1,
                }),
                'ImageOnlyBaseline': (ImageOnlyBaseline, {})
            }
            if args.model_name not in model_mapping:
                logging.error(f"FATAL: Unknown model name '{args.model_name}'. Choose from {list(model_mapping.keys())} or pass --experiment_config.");
                sys.exit(1)
            model_class, config = model_mapping[args.model_name]

        cm_dir = os.path.join(args.output_dir, 'confusion_matrix')
        os.makedirs(cm_dir, exist_ok=True)
        input_dim = experiment_config.get("input_dim", test_dataset.tensors[0].shape[-1]) if experiment_config else test_dataset.tensors[0].shape[-1]

        IMAGE_FEATURE_DIM = test_loader.dataset.image_feature_dim
        logging.info(f"Detected image feature shape for test: {tuple(test_loader.dataset.image_feature_shape[1:])}")

        final_test_and_report(model_class, args.model_path, test_loader, device, label_encoder, args, input_dim, config,
                              output_paths={'cm_dir': cm_dir}, image_feature_dim=IMAGE_FEATURE_DIM)

    logging.info(f"\nTotal execution time: {(time.time() - start_main_time) / 60:.2f} minutes.")


if __name__ == "__main__":
    args = get_args()
    logging.info(f"Running study: {args.study_name}")
    main(args)

    # !/bin/bash
    # run_all_datasets.sh

    # 设置为在第一个错误处停止
    """
    set - e

    echo
    "--- [ 1/4 ] 正在运行: NUS Hand Posture Dataset ---"
    python
    main.py - -study_name
    sota_only - -epochs
    50 \
    - -train_csv_path
    "data/NUS_Hand_Posture/all_landmarks_multimodal.csv" \
    - -image_features_path
    "data/NUS_Hand_Posture/all_image_features_resnet50.npy" \
    - -output_dir
    "./results_NUS_Hand_Posture"

    echo
    "--- [ 2/4 ] 正在运行: Indian Sign Language Dataset ---"
    python
    main.py - -study_name
    sota_only - -epochs
    50 \
    - -train_csv_path
    "data/Indian Sign Language/all_landmarks_multimodal.csv" \
    - -image_features_path
    "data/Indian Sign Language/all_image_features_resnet50.npy" \
    - -output_dir
    "./results_Indian_Sign_Language"

    echo
    "--- [ 3/4 ] 正在运行: asl_dataset (新) ---"
    python
    main.py - -study_name
    sota_only - -epochs
    50 \
    - -train_csv_path
    "data/asl_dataset/all_landmarks_multimodal.csv" \
    - -image_features_path
    "data/asl_dataset/all_image_features_resnet50.npy" \
    - -output_dir
    "./results_asl_dataset"

    echo
    "--- [ 4/4 ] 正在运行: ASL Alphabet (旧) ---"
    # 注意：这些路径来自你 config.py 的默认值
    python
    main.py - -study_name
    sota_only - -epochs
    50 \
    - -train_csv_path
    "data/ASL_Alphabet/all_landmarks_multimodal.csv" \
    - -image_features_path
    "data/ASL_Alphabet/all_image_features_resnet50.npy" \
    - -output_dir
    "./results_ASL_Alphabet_Old"

    echo
    "--- 所有 4 个数据集的实验均已完成 ---"
    """
