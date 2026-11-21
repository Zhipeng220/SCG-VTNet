# utils.py
import logging
import os
import time
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from scg_vtnet.datasets.landmark_features import get_feature_columns, calculate_angle, get_hand_detected_mask, add_advanced_features

from scg_vtnet.datasets.splits import write_split_manifest


class ImageFeatureReference:
    def __init__(self, image_features, indices):
        self.image_features = image_features
        self.indices = np.asarray(indices)
        self.shape = (len(self.indices), *image_features.shape[1:])
        self.feature_shape = image_features.shape[1:]
        self.feature_dim = image_features.shape[-1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.image_features[self.indices[item]]


class IndexedMultimodalDataset(Dataset):
    def __init__(self, keypoints, raw_xyz, image_features, indices, hand_detected, labels):
        self.keypoints = torch.tensor(keypoints, dtype=torch.float32)
        self.raw_xyz = torch.tensor(raw_xyz, dtype=torch.float32)
        self.image_features = image_features
        self.indices = np.asarray(indices)
        self.hand_detected = torch.tensor(hand_detected, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.image_feature_shape = (len(self.indices), *image_features.shape[1:])
        self.image_feature_dim = image_features.shape[-1]
        self.tensors = (
            self.keypoints,
            self.raw_xyz,
            ImageFeatureReference(image_features, self.indices),
            self.hand_detected,
            self.labels,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        source_idx = int(self.indices[item])
        image_feature = torch.from_numpy(np.array(self.image_features[source_idx], copy=True))
        return (
            self.keypoints[item],
            self.raw_xyz[item],
            image_feature,
            self.hand_detected[item],
            self.labels[item],
        )




def setup_logger(log_dir="."):
    """Configures the logger to output to console and a file."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f"experiment_log_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_filepath)  # Corrected typo
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    logging.info(f"Log will be saved to: {log_filepath}")
    return logger










def load_and_preprocess_train_data(csv_path, image_features_path, use_advanced_features=True, seed=42,
                                   aar_descriptor="all"):

    """
    (V19 - Scheme B)
    Loads landmarks from CSV, pre-extracted features from NPY, applies feature
    engineering to landmarks, scales landmarks, splits, and returns TensorDatasets.
    """
    logging.info(f"Loading landmarks from {csv_path}...")
    try:
        data = pd.read_csv(csv_path)
        if 'image_path' not in data.columns:
            logging.warning(f"CSV {csv_path} does not contain 'image_path' column needed for multimodal alignment.")
    except FileNotFoundError:
        logging.error(f"FATAL: Landmark CSV file not found: {csv_path}");
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading landmark CSV: {e}");
        sys.exit(1)

    # Load image features
    logging.info(f"Loading pre-extracted image features from {image_features_path}...")
    try:
        image_features_np = np.load(image_features_path, mmap_mode='r')
    except FileNotFoundError:
        logging.error(f"FATAL: Image features file not found: {image_features_path}");
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading image features NPY: {e}");
        sys.exit(1)

    # Validate alignment
    if len(data) != len(image_features_np):
        logging.error(f"FATAL: Mismatch! CSV has {len(data)} rows but NPY file has {len(image_features_np)} features.");
        sys.exit(1)
    logging.info(f"Successfully loaded {len(image_features_np)} image features.")

    # Feature Engineering for Landmarks (V18 logic)
    feature_cols = get_feature_columns(data)
    features, labels = data[feature_cols].values, data['label'].values
    hand_detected = get_hand_detected_mask(data)

    num_feature_columns = features.shape[1]
    if num_feature_columns % 21 != 0:
        logging.error(f"FATAL: Landmark feature count ({num_feature_columns}) is not a multiple of 21.");
        sys.exit(1)
    input_dim_original = num_feature_columns // 21  # Original dim (e.g., 3)

    X_reshaped = features.reshape(len(data), 21, input_dim_original)
    X_raw_xyz = X_reshaped[..., :3].astype(np.float32)
    X_raw_xyz[hand_detected == 0] = 0.0

    if use_advanced_features:
        X_engineered_reshaped = add_advanced_features(X_reshaped, hand_detected, descriptor_mode=aar_descriptor)  # Apply selected AAR feature engineering
    else:
        X_engineered_reshaped = X_reshaped  # 使用原始 (N, 21, 3) 特征
        X_engineered_reshaped[hand_detected == 0] = 0.0
        logging.info("Skipping advanced feature engineering. Using basic (3D) keypoints.")
    # --- [MODIFIED END] ---

    input_dim_engineered = X_engineered_reshaped.shape[2]  # New keypoint dim (e.g., 3 or 11)
    X_engineered_flat = X_engineered_reshaped.reshape(len(data), 21 * input_dim_engineered)

    # Encode labels before splitting. The scaler is fit only on the train split.
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    indices = np.arange(len(data))
    idx_train_val, idx_test, y_train_val, y_test = train_test_split(
        indices, encoded_labels,
        test_size=0.2, random_state=seed, stratify=encoded_labels
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_train_val, y_train_val,
        test_size=0.25, random_state=seed, stratify=y_train_val
    )

    X_train, X_val, X_test = X_engineered_flat[idx_train], X_engineered_flat[idx_val], X_engineered_flat[idx_test]
    raw_train, raw_val, raw_test = X_raw_xyz[idx_train], X_raw_xyz[idx_val], X_raw_xyz[idx_test]
    hand_train, hand_val, hand_test = hand_detected[idx_train], hand_detected[idx_val], hand_detected[idx_test]

    scaler = StandardScaler()
    scaler_fit_mask = hand_train == 1
    if np.any(scaler_fit_mask):
        scaler.fit(X_train[scaler_fit_mask])
        logging.info("Fitted StandardScaler on detected-hand samples from the train split.")
    else:
        scaler.fit(X_train)
        logging.warning("No detected-hand train samples found; fitted StandardScaler on all train samples.")
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_train[hand_train == 0] = 0.0
    X_val[hand_val == 0] = 0.0
    X_test[hand_test == 0] = 0.0

    manifest_path = os.path.join(os.path.dirname(csv_path) or ".", f"split_manifest_seed{seed}.csv")
    write_split_manifest(manifest_path, data, idx_train, "train", seed, append=False)
    write_split_manifest(manifest_path, data, idx_val, "val", seed, append=True)
    write_split_manifest(manifest_path, data, idx_test, "test", seed, append=True)
    logging.info(f"Split manifest saved to {manifest_path}")

    # Weighted Sampler for training set imbalance
    class_counts = np.bincount(y_train)
    weights = 1. / (class_counts + 1e-8)
    # Give more weight to extremely rare classes if any (optional)
    # weights[class_counts < 10] *= 10
    sample_weights = weights[y_train]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y_train), replacement=True)

    # Create TensorDatasets with scaled model inputs plus raw xyz for anatomical losses/metrics.
    data_dict = {}
    for name, X_flat, raw_xyz, hand_mask, split_indices, y in [
        ('train', X_train, raw_train, hand_train, idx_train, y_train),
        ('val', X_val, raw_val, hand_val, idx_val, y_val),
        ('test', X_test, raw_test, hand_test, idx_test, y_test)
    ]:
        data_dict[name] = IndexedMultimodalDataset(
            X_flat.reshape(-1, 21, input_dim_engineered),
            raw_xyz,
            image_features_np,
            split_indices,
            hand_mask,
            y,
        )

    logging.info(
        f"Data split: {len(data_dict['train'])} train, {len(data_dict['val'])} validation, {len(data_dict['test'])} test samples.")
    # Return the engineered keypoint dimension
    return data_dict, scaler, label_encoder, sampler, input_dim_engineered


def load_and_preprocess_test_data(csv_path, image_features_path, scaler, label_encoder,
                                  use_advanced_features=True, aar_descriptor="all"):

    """
    (V19 - Scheme B)
    Loads landmarks, pre-extracted features for an external test set.
    Uses pre-fitted scaler and label encoder.
    """
    logging.info(f"Loading external test landmarks from {csv_path}...")
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        logging.error(f"FATAL: Landmark CSV file not found: {csv_path}");
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading landmark CSV: {e}");
        sys.exit(1)

    # Load image features
    logging.info(f"Loading pre-extracted image features from {image_features_path}...")
    try:
        image_features_np = np.load(image_features_path, mmap_mode='r')
    except FileNotFoundError:
        logging.error(f"FATAL: Image features file not found: {image_features_path}");
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading image features NPY: {e}");
        sys.exit(1)

    # Validate alignment
    if len(data) != len(image_features_np):
        logging.error(f"FATAL: Mismatch! CSV has {len(data)} rows but NPY file has {len(image_features_np)} features.");
        sys.exit(1)

    # Feature Engineering for Landmarks (V18 logic)
    feature_cols = get_feature_columns(data)
    features, labels = data[feature_cols].values, data['label'].values
    hand_detected = get_hand_detected_mask(data)

    num_feature_columns = features.shape[1]
    if num_feature_columns % 21 != 0:
        logging.error(f"FATAL: Landmark feature count ({num_feature_columns}) is not a multiple of 21.");
        sys.exit(1)
    input_dim_original = num_feature_columns // 21

    X_reshaped = features.reshape(len(data), 21, input_dim_original)
    X_raw_xyz = X_reshaped[..., :3].astype(np.float32)
    X_raw_xyz[hand_detected == 0] = 0.0

    if use_advanced_features:
        X_engineered_reshaped = add_advanced_features(X_reshaped, hand_detected, descriptor_mode=aar_descriptor)
    else:
        X_engineered_reshaped = X_reshaped
        X_engineered_reshaped[hand_detected == 0] = 0.0
        logging.info("Skipping advanced feature engineering for test set.")
    # --- [MODIFIED END] ---

    input_dim_engineered = X_engineered_reshaped.shape[2]
    X_engineered_flat = X_engineered_reshaped.reshape(len(data), 21 * input_dim_engineered)

    # Scale landmarks and encode labels using pre-fitted tools
    try:
        X_scaled_flat = scaler.transform(X_engineered_flat)
        X_scaled_flat[hand_detected == 0] = 0.0
        encoded_labels = label_encoder.transform(labels)
    except Exception as e:
        logging.error(f"Error transforming test data: {e}");
        sys.exit(1)

    return IndexedMultimodalDataset(
        X_scaled_flat.reshape(-1, 21, input_dim_engineered),
        X_raw_xyz,
        image_features_np,
        np.arange(len(data)),
        hand_detected,
        encoded_labels,
    )
