# config.py
import argparse
import sys


def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate hand gesture recognition models (Refactored - 21 keypoints, Train/Test modes).")

    # --- 实验控制参数 ---
    parser.add_argument('--study_name', type=str, default='core_ablation',
                        choices=['ablation', 'full_ablation',
                                 'hyperparam_embed_dim', 'hyperparam_num_layers',
                                 'hyperparam_num_heads', 'hyperparam_cont_temp',
                                 'hyperparam_anat', 'hyperparam_contrastive',
                                 'sota_only', 'anat_mechanism', 'core_ablation',
                                 'wo_aar_only',
                                 'visual_modulation',
                                 'aar_construction',
                                 'objective_construction',
                                 'fair_e1_e7', 'asl_matched'],
                        help="Specify the study to run: 'ablation' for component analysis, "
                             "'hyperparam_embed_dim' for embedding dimension, "
                             "'hyperparam_num_layers' for attention layers, "
                             "'hyperparam_num_heads' for attention heads, "
                             "'hyperparam_cont_temp' for contrastive temperature, "
                             "'hyperparam_anat'/'hyperparam_contrastive' for loss weights, "
                             "'full_ablation' for the complete ablation study,"
                             "'sota_only' for running only the SOTA model.")  # <-- [新增]
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help="Set script to 'train' or 'test' mode.")

    # --- [ 新增: 随机数种子 ] ---
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility (e.g., set to $i in a loop for stability tests).")
    # --- [ 新增结束 ] ---

    # ... (文件的其余部分保持不变) ...

    parser.add_argument('--frozen_manifest', help="Explicit frozen split CSV.")
    parser.add_argument('--matched_variant', choices=['F', 'A0', 'G0', 'Uniform'])
    parser.add_argument('--baseline_method', type=str, default=None,
                        choices=['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7'],
                        help="Optional method_id filter for fair_e1_e7 (e.g., E2 or E5).")
    parser.add_argument('--include_skeleton_zeroing', action='store_true', default=False,
                        help="Include fixed-visual skeleton zeroing in robustness evaluation.")
    parser.add_argument('--perturbation_seed', type=int, default=20261120)
    parser.add_argument('--train_csv_path', type=str,
                        # [修改] 指向您处理后的 CSV
                        default="data/asl_alphabet/processed_train/zerofill_landmarks_mask.csv",
                        help="Path to the training CSV file (should include 'image_path' column).")
    parser.add_argument('--test_csv_path', type=str,
                        # [修改] 指向测试 CSV (如果 test 模式需要)
                        default="data/asl_alphabet/processed_test/test_landmarks_multimodal.csv",
                        help="Path to the external test CSV file for test mode.")
    parser.add_argument('--image_features_path', type=str,
                        # [修改] 指向您生成的 NPY 文件
                        default="data/asl_alphabet/processed_train/zerofill_visual_tokens_layer4_compact.npy",
                        help="Path to the pre-extracted .npy file for image features.")

    parser.add_argument('--global_image_features_path', type=str, default=None,
                        help="Path to global visual features, e.g., N x 2048 ResNet GAP features.")
    parser.add_argument('--token_image_features_path', type=str, default=None,
                        help="Path to token-level visual features, e.g., N x 49 x 2048 visual tokens.")
    parser.add_argument('--efficientnet_features_path', type=str, default=None,
                        help="Path to pre-extracted EfficientNet-B0 RGB features for fair_e1_e7.")
    parser.add_argument('--vit_features_path', type=str, default=None,
                        help="Path to pre-extracted ViT-S/16 token/global RGB features for fair_e1_e7.")
    parser.add_argument('--allow_feature_proxy', action='store_true',
                        help="Allow fair_e1_e7 to use the ResNet-50 feature file for E2/E3 when exact features are missing.")
    parser.add_argument('--aar_descriptor', type=str, default='all',
                        choices=['none', 'bone', 'angle', 'tip', 'bone_angle', 'all'],
                        help="Descriptor mode for AAR construction experiments.")
    parser.add_argument('--model_path', type=str, help="Path to the pre-trained model .pth file for testing.")
    parser.add_argument('--model_name', type=str,
                        help="Name of the model to test (e.g., 'Hybrid_GCN_Attention(Gated)').")
    parser.add_argument('--experiment_config', type=str,
                        help="Path to config_<experiment>.json saved during training for exact test-mode reconstruction.")
    parser.add_argument('--output_dir', type=str, default="./results",
                        help="Directory to save all outputs (logs, weights, plots, scalers, etc.)")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--debug', action='store_true',
                        help="Use a short debug run; sets epochs to 2 unless --epochs is explicitly provided.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=3e-4, help="Max learning rate for OneCycleLR.")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="Weight decay for AdamW.")
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help="Max norm for gradient clipping (0 for no clipping).")
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help="Number of epochs to wait for validation accuracy improvement before stopping (0 to disable).")
    parser.add_argument('--anatomical_weight_initial', type=float, default=0.0,
                        help="Initial weight for anatomical loss.")
    parser.add_argument('--anatomical_weight_final', type=float, default=0.02,
                        help="Default final weight for anatomical loss.")
    parser.add_argument('--anat_angle_weight', type=float, default=1.0,
                        help="Internal weight for anatomical angle loss.")
    parser.add_argument('--anat_length_weight', type=float, default=2.0,
                        help="Internal weight for anatomical bone-length loss.")
    parser.add_argument('--anat_planarity_weight', type=float, default=0.5,
                        help="Internal weight for anatomical palm-planarity loss.")
    parser.add_argument('--anat_target', type=str, default='none',
                        choices=['none', 'raw_input', 'refined_xyz'],
                        help="Target for anatomical loss. raw_input is a diagnostic baseline only.")
    parser.add_argument('--anat_schedule', type=str, default='cosine',
                        choices=['cosine', 'constant'],
                        help="Schedule for anatomical loss weight.")
    parser.add_argument('--use_keypoint_refinement', action='store_true', default=False,
                        help="Enable learnable keypoint refinement head for anatomical loss.")
    parser.add_argument('--refine_scale', type=float, default=0.05,
                        help="Bounded residual scale for the keypoint refinement head.")
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                        help="Weight for Hierarchical Contrastive Loss.")
    parser.add_argument('--contrastive_temperature', type=float, default=0.1,
                        help="Temperature for Hierarchical Contrastive Loss.")
    parser.add_argument('--num_layers', type=int, default=4, help="Number of attention blocks.")
    parser.add_argument('--num_heads', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--embed_dim', type=int, default=256, help="Embedding dimension.")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate.")
    parser.add_argument('--modality_dropout', type=float, default=0.0,
                        help="Probability of dropping one modality during multimodal training.")

    args, unknown = parser.parse_known_args()
    has_explicit_epochs = any(arg == '--epochs' or arg.startswith('--epochs=') for arg in sys.argv)
    if args.debug and not has_explicit_epochs:
        args.epochs = 2
    return args
