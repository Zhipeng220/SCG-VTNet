import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


DEGRADATIONS = {
    "clean": ("clean", 0),
    "dark_light": ("dark", 0.75),
    "dark_heavy": ("dark", 0.45),
    "blur_light": ("blur", 1.5),
    "blur_heavy": ("blur", 4.0),
    "occlude_light": ("occlude", 0.08),
    "occlude_heavy": ("occlude", 0.20),
}
PERTURBATION_SEED = 20261120


def degrade(image, kind, value, index):
    params = {"kind": kind, "value": value}
    if kind == "dark":
        image = ImageEnhance.Brightness(image).enhance(value)
    elif kind == "blur":
        image = image.filter(ImageFilter.GaussianBlur(radius=value))
    elif kind == "occlude":
        rng = np.random.default_rng([PERTURBATION_SEED, index])
        width = max(1, round(image.width * np.sqrt(value)))
        height = max(1, round(image.height * np.sqrt(value)))
        x = int(rng.integers(0, image.width - width + 1))
        y = int(rng.integers(0, image.height - height + 1))
        image = image.copy()
        image.paste((0, 0, 0), (x, y, x + width, y + height))
        params.update(rect=[x, y, width, height], fill=[0, 0, 0])
    return image, params


def letterbox(image):
    scale = 224 / max(image.size)
    width, height = (round(side * scale) for side in image.size)
    padded = Image.new("RGB", (224, 224))
    padded.paste(image.resize((width, height), Image.Resampling.BILINEAR),
                 ((224 - width) // 2, (224 - height) // 2))
    return padded
