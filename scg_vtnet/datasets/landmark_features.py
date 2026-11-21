import numpy as np


def get_feature_columns(df):
    """Identifies feature columns in the dataframe."""
    # Prioritize columns starting with x_, y_, z_
    coord_cols = [col for col in df.columns if str(col).lower().startswith(('x_', 'y_', 'z_'))]
    if coord_cols: return coord_cols

    # Fallback: dynamically find columns excluding known non-feature ones
    logging.warning("Falling back to positional feature selection (excluding label and image_path).")
    non_feature_cols = ['label', 'image_path', 'hand_detected']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    if feature_cols: return feature_cols

    logging.error("FATAL: Could not determine feature columns.")
    sys.exit(1)


def calculate_angle(p1, p2, p3):
    """
    Vectorized calculation of the angle (radians) at p2.
    p1, p2, p3 shape: (num_samples, 3)
    """
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.sum(v1 * v2, axis=1)
    norm_v1 = np.linalg.norm(v1, axis=1)
    norm_v2 = np.linalg.norm(v2, axis=1)
    cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-8)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return angle


def get_hand_detected_mask(df):
    if 'hand_detected' not in df.columns:
        return np.ones(len(df), dtype=np.float32)
    return df['hand_detected'].fillna(0).astype(np.float32).values


def add_advanced_features(X_reshaped, hand_detected=None, descriptor_mode="all"):
    """
    (V18 Feature Engineering)
    Adds bone vectors, joint angles, and fingertip distances to keypoint data.
    Input X_reshaped shape: (num_samples, 21, input_dim_original)
    """
    num_samples, num_keypoints, input_dim_original = X_reshaped.shape

    # Always use only the first 3 coordinates (x, y, z) for geometric features
    coords = X_reshaped[..., :3]
    if hand_detected is None:
        valid_samples = np.ones(num_samples, dtype=bool)
    else:
        valid_samples = np.asarray(hand_detected).astype(bool)

    # 1. Bone Vectors (3 features)
    parent_map = {
        0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 0, 6: 5, 7: 6, 8: 7,
        9: 0, 10: 9, 11: 10, 12: 11, 13: 0, 14: 13, 15: 14, 16: 15,
        17: 0, 18: 17, 19: 18, 20: 19
    }
    bone_vectors = np.zeros((num_samples, num_keypoints, 3))
    for i in range(1, num_keypoints):
        parent_idx = parent_map[i]
        bone_vectors[:, i, :] = coords[:, i, :] - coords[:, parent_idx, :]

    # 2. Joint Angles (1 feature)
    angle_features = np.zeros((num_samples, num_keypoints, 1))
    angle_triplets = [
        [1, 2, 3], [2, 3, 4],  # Thumb
        [5, 6, 7], [6, 7, 8],  # Index
        [9, 10, 11], [10, 11, 12],  # Middle
        [13, 14, 15], [14, 15, 16],  # Ring
        [17, 18, 19], [18, 19, 20]  # Pinky
    ]
    for p1_idx, p2_idx, p3_idx in angle_triplets:
        p1, p2, p3 = coords[:, p1_idx], coords[:, p2_idx], coords[:, p3_idx]
        angles = calculate_angle(p1, p2, p3)
        angle_features[:, p2_idx, 0] = angles  # Store angle at the middle joint

    # 3. Fingertip Distances (4 features)
    fingertip_dist_features = np.zeros((num_samples, num_keypoints, 4))
    p_thumb_tip = coords[:, 4]
    other_tip_indices = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky

    for i, tip_idx in enumerate(other_tip_indices):
        p_tip = coords[:, tip_idx]
        dist = np.linalg.norm(p_thumb_tip - p_tip, axis=1)
        # Broadcast this distance to all 21 keypoints
        fingertip_dist_features[..., i] = dist[:, np.newaxis]

    # 4. Concatenate selected descriptor groups.
    # Descriptor modes:
    #   bone       = coord + bone vectors
    #   angle      = coord + joint angles
    #   tip        = coord + fingertip distances
    #   bone_angle = coord + bone vectors + joint angles
    #   all        = coord + bone vectors + joint angles + fingertip distances
    descriptor_mode = str(descriptor_mode or "all").lower()
    if descriptor_mode in ["aar"]:
        descriptor_mode = "all"
    if descriptor_mode in ["td", "tip_dist", "fingertip", "fingertip_distance"]:
        descriptor_mode = "tip"
    if descriptor_mode in ["bv", "bone_vector"]:
        descriptor_mode = "bone"
    if descriptor_mode in ["ja", "joint_angle"]:
        descriptor_mode = "angle"
    if descriptor_mode in ["bv_ja", "bone+angle", "bone_angle"]:
        descriptor_mode = "bone_angle"

    selected_features = [X_reshaped]
    if descriptor_mode in ["bone", "bone_angle", "all"]:
        selected_features.append(bone_vectors)
    if descriptor_mode in ["angle", "bone_angle", "all"]:
        selected_features.append(angle_features)
    if descriptor_mode in ["tip", "all"]:
        selected_features.append(fingertip_dist_features)
    if descriptor_mode not in ["bone", "angle", "tip", "bone_angle", "all"]:
        raise ValueError(f"Unknown AAR descriptor_mode: {descriptor_mode}")

    X_engineered = np.concatenate(selected_features, axis=2)
    X_engineered[~valid_samples] = 0.0

    logging.info(
        f"Advanced feature engineering mode={descriptor_mode}: "
        f"Input dimension increased from {input_dim_original} to {X_engineered.shape[2]}")
    return X_engineered
