import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm


class CsvImageDataset(Dataset):
    def __init__(self, csv_path, image_root, transform):
        self.data = pd.read_csv(csv_path)
        if "image_path" not in self.data.columns:
            raise ValueError(f"{csv_path} does not contain an image_path column.")
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rel_path = str(self.data.iloc[index]["image_path"]).replace("\\", "/")
        image_path = self.image_root / rel_path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.transform(image), index


def build_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_efficientnet_b0(device):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model, weights.transforms(), (1280,)


def build_vit_s16(device):
    try:
        import timm
        from timm.data import create_transform, resolve_model_data_config
    except ImportError as exc:
        raise ImportError(
            "Exact ViT-S/16 feature extraction requires timm. Install it in the ACGCAT "
            "environment, then rerun this script: python -m pip install timm"
        ) from exc

    model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
    config = resolve_model_data_config(model)
    transform = create_transform(**config, is_training=False)
    model.to(device)
    model.eval()

    embed_dim = getattr(model, "embed_dim", 384)
    patch_count = getattr(model.patch_embed, "num_patches", 196)
    return model, transform, (patch_count, embed_dim)


def forward_features(model, images, backbone):
    with torch.no_grad():
        if backbone == "efficientnet_b0":
            return model(images)

        tokens = model.forward_features(images)
        if isinstance(tokens, dict):
            tokens = tokens.get("x") or tokens.get("tokens")
        if tokens.ndim != 3:
            raise ValueError(f"Expected ViT tokens with shape B,T,C, got {tuple(tokens.shape)}")
        return tokens[:, 1:, :]


def extract(args):
    device = build_device(args.device)
    if args.backbone == "efficientnet_b0":
        model, transform, feature_shape = build_efficientnet_b0(device)
    elif args.backbone == "vit_s16":
        model, transform, feature_shape = build_vit_s16(device)
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")

    dataset = CsvImageDataset(args.csv_path, args.image_root, transform)
    if args.resume_from < 0 or args.resume_from >= len(dataset):
        raise ValueError(f"--resume_from must be in [0, {len(dataset) - 1}], got {args.resume_from}")
    run_dataset = dataset if args.resume_from == 0 else Subset(dataset, range(args.resume_from, len(dataset)))
    loader = DataLoader(
        run_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if args.save_dtype == "float16" else np.float32
    mode = "r+" if args.resume_from > 0 and output_path.exists() else "w+"
    features = np.lib.format.open_memmap(
        output_path,
        mode=mode,
        dtype=dtype,
        shape=(len(dataset), *feature_shape),
    )

    for images, indices in tqdm(loader, desc=f"Extracting {args.backbone}"):
        images = images.to(device, non_blocking=True)
        batch_features = forward_features(model, images, args.backbone).detach().cpu().numpy()
        features[np.asarray(indices, dtype=np.int64)] = batch_features.astype(dtype)

    features.flush()
    print(f"Saved {args.backbone} features to {output_path} with shape {(len(dataset), *feature_shape)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract exact RGB backbone features for fair_e1_e7.")
    parser.add_argument("--backbone", choices=["efficientnet_b0", "vit_s16"], required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--resume_from", type=int, default=0,
                        help="Resume from this CSV row index. Use with an existing output file.")
    return parser.parse_args()


if __name__ == "__main__":
    extract(parse_args())
