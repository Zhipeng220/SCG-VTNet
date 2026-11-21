# process_full_dataset_resnet50.py
import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
import shutil
import torch
import torchvision.models as models
# --- [ 修改 1: 导入 ResNet50 权重 ] ---
from torchvision.models import ResNet50_Weights  # <-- 新增
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sys


def process_full_dataset(
        root_dir,
        output_csv_path,
        output_npy_path,  # Path will reflect ResNet50
        output_img_copy_dir
):
    """
    (合并且不拆分版 - ResNet50)
    遍历图像目录，一次性提取 MediaPipe 关键点和 ResNet50 视觉特征。
    ... (rest of docstring is the same)
    """

    # --- 1. 初始化所有模型 ---

    # MediaPipe
    print("Initializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands

    # --- [ 修改 2: 更改 MediaPipe 初始化参数 ] ---
    hands = mp_hands.Hands(
        static_image_mode=True,  # <-- 更改：对于静态图片，必须设为 True
        max_num_hands=1,
        min_detection_confidence=0.3  # <-- 更改：适当降低阈值以检测更多手
        # min_tracking_confidence 在 static_image_mode=True 时无效
    )
    # --- [ 修改结束 ] ---

    # --- [ 修改: 使用 ResNet50 ] ---
    print("Initializing ResNet50...")
    device = torch.device("cuda" if torch.cuda.is_available() else (
        "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # --- [ 修改 3: 更新 ResNet50 加载方式 ] ---
    # 加载预训练的 ResNet50 (使用新的 'weights' API)
    model_resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)  # <-- 更改
    # --- [ 修改结束 ] ---

    # ResNet50 的全连接层输入维度是 2048
    # 我们仍然移除它以获取特征
    model_resnet.fc = torch.nn.Identity()
    model_resnet.to(device)
    model_resnet.eval()

    # ResNet 的标准预处理 (保持不变)
    preprocess_resnet = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # --- [ 修改结束 ] ---

    # --- 2. 遍历所有图像并提取两种特征 ---
    all_landmark_data = []
    all_image_features = []

    print(f"Starting data scan from directory: {root_dir}")
    if not os.path.isdir(root_dir):
        print(f"Error: Directory not found -> {root_dir}")
        return

    label_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    total_files = sum([len(files) for r, d, files in os.walk(root_dir)])

    with tqdm(total=total_files, desc="Processing Images") as pbar:
        for label in label_dirs:
            label_path = os.path.join(root_dir, label)
            image_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for image_name in image_files:
                pbar.update(1)
                relative_img_path = os.path.join(label, image_name)
                original_full_path = os.path.join(label_path, image_name)

                try:
                    # MediaPipe 提取 (保持不变)
                    image_cv = cv2.imread(original_full_path)
                    if image_cv is None:
                        tqdm.write(f"SKIPPING (CV2 read fail): {original_full_path}")
                        continue
                    image_rgb_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                    results = hands.process(image_rgb_cv)

                    if results.multi_hand_landmarks:
                        # 1. 保存关键点数据 (保持不变)
                        hand_landmarks = results.multi_hand_landmarks[0]
                        row_data = [label, relative_img_path]
                        for landmark in hand_landmarks.landmark:
                            row_data.extend([landmark.x, landmark.y, landmark.z])
                        all_landmark_data.append(row_data)

                        # 2. ResNet50 提取 (模型已改为 ResNet50)
                        pil_image = Image.fromarray(image_rgb_cv)
                        image_tensor = preprocess_resnet(pil_image).unsqueeze(0).to(device)

                        with torch.no_grad():
                            features = model_resnet(image_tensor)  # 现在输出 (1, 2048)

                        all_image_features.append(features.cpu().numpy())

                        # 3. 复制有效图像 (保持不变)
                        dest_label_dir = os.path.join(output_img_copy_dir, label)
                        os.makedirs(dest_label_dir, exist_ok=True)
                        dest_file_path = os.path.join(dest_label_dir, image_name)
                        shutil.copy(original_full_path, dest_file_path)

                    else:
                        tqdm.write(f"SKIPPING (No hand): {original_full_path}")

                except Exception as e:
                    print(f"An error occurred while processing {original_full_path}: {e}")

    hands.close()

    if not all_landmark_data:
        print("No landmark data was extracted. Halting process.")
        return

    print("\nImage processing complete. Creating full dataset...")

    # --- 3. 创建 DataFrame 和 Numpy Array ---

    # DataFrame (保持不变)
    headers = ['label', 'image_path']
    for i in range(21):
        headers.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    full_df = pd.DataFrame(all_landmark_data, columns=headers)

    # Numpy Array (现在包含 2048 维特征)
    full_image_features_np = np.concatenate(all_image_features, axis=0)

    # 验证 (保持不变)
    if len(full_df) != len(full_image_features_np):
        print(f"FATAL ERROR: Mismatch! Landmarks: {len(full_df)}, Img Features: {len(full_image_features_np)}")
        sys.exit(1)

    print(f"Full, synchronized dataset created with {len(full_df)} samples.")
    print(f"Image feature dimension: {full_image_features_np.shape[1]}")  # Will print 2048

    # --- 4. 保存所有数据 ---
    print("\nSaving landmark CSV and image feature .npy file...")

    # 保存 CSV (保持不变)
    full_df.to_csv(output_csv_path, index=False)

    # 保存 NPY (文件名已在 __main__ 中更新)
    np.save(output_npy_path, full_image_features_np)

    print("\n" + "=" * 50)
    print("Process Finished Successfully!")
    print(f"Total landmarks (CSV) saved to: {output_csv_path} ({len(full_df)} rows)")
    print(f"Total image features (NPY) saved to: {output_npy_path} ({full_image_features_np.shape})")
    print(f"Processed images (copied) saved to: {output_img_copy_dir}")
    print("=" * 50)


if __name__ == '__main__':
    # ==================== Parameters to Modify ====================
    # 1. 原始数据集的根目录
    root_dataset_dir = '/Users/gzp/Desktop/静态手势识别/data/ASL Alphabet/asl_alphabet_train/asl_alphabet_train'

    # 2. 输出文件的基础目录
    output_base_dir = '/Users/gzp/Desktop/静态手势识别/data/ASL_Alphabet_Processed_Full_ResNet50'  # Suggest new dir

    os.makedirs(output_base_dir, exist_ok=True)

    # 3. 输出文件路径
    csv_output = os.path.join(output_base_dir, 'all_landmarks_multimodal.csv')  # CSV name can stay same

    # --- [ 修改: 更新 NPY 文件名 ] ---
    npy_output = os.path.join(output_base_dir, 'all_image_features_resnet50.npy')  # Changed filename
    # --- [ 修改结束 ] ---

    # 4. 输出图像目录
    img_output_dir = os.path.join(output_base_dir, 'asl_images_processed')

    # ==============================================================

    process_full_dataset(
        root_dataset_dir,
        csv_output,
        npy_output,  # Pass updated path
        img_output_dir
    )