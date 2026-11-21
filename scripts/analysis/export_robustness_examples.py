"""Export the source and paired degraded panels used in Fig. 4."""

import csv
import hashlib
import json

from PIL import Image

from scripts.experiments.evaluate_input_robustness import DEGRADATIONS, RAW, ROOT, degrade


METHODS = ("E7", "E2", "E5")
PAIRED = ROOT / "runs/asl_paired_degradation/predictions.csv"
FULL = ROOT / "runs/asl_baselines/seed42_predictions"
OUTPUT = ROOT / "runs/figure_panels"


def read_csv(path):
    with path.open(encoding="utf8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    paired = {(int(row["original_index"]), row["condition"], row["method"]): row
              for row in read_csv(PAIRED)}
    selections = (("a", 1843, "blur_heavy"), ("b", 180, "blur_heavy"),
                  ("c", 11152, "occlude_heavy"))
    records = []
    OUTPUT.mkdir(exist_ok=True)
    for panel, index, condition in selections:
        row = paired[index, condition, "E7"]
        path = row["image_path"]
        with Image.open(RAW / path) as image:
            original = image.convert("RGB")
        transformed, params = degrade(original, *DEGRADATIONS[condition], index)
        if (hashlib.sha256(original.tobytes()).hexdigest() != paired[index, "clean", "E7"]["image_sha256"]
                or hashlib.sha256(transformed.tobytes()).hexdigest() != row["image_sha256"]
                or params != json.loads(row["params"])):
            raise ValueError(f"Displayed pixels differ from the evaluated pair: {index}")
        original.save(OUTPUT / f"{panel}_clean.png")
        transformed.save(OUTPUT / f"{panel}_degraded.png")
        records.append({
            "panel": panel, "original_index": index, "source": path,
            "label": row["true_label"], "parameters": params,
            "detection": [row["clean_detected"], row["condition_detected"]],
            "predictions": {method: paired[index, condition, method]["prediction"] for method in METHODS},
        })

    full = {method: {int(row["original_index"]): row for row in read_csv(FULL / f"{method}_seed_42.csv")}
            for method in METHODS}
    index = min(i for i, row in full["E7"].items()
                if row["true_label"] == "X" and row["predicted_label"] == "R")
    row = full["E7"][index]
    source = row["image_path"].replace(chr(92), "/")
    with Image.open(RAW / source) as image:
        image.convert("RGB").save(OUTPUT / "d_clean.png")
    records.append({
        "panel": "d", "original_index": index, "source": source, "label": "X",
        "parameters": None, "detection": [row["hand_detected"]],
        "predictions": {method: full[method][index]["predicted_label"] for method in METHODS},
    })
    result = json.dumps(records, indent=2) + "\n"
    (OUTPUT / "provenance.json").write_text(result, encoding="utf8")
    print(result)


if __name__ == "__main__":
    main()

