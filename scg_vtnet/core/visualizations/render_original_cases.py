"""Render the screened ASL Alphabet difficult-case plate without modifying source images."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from PIL import Image

TITLES = {
    "challenging_light": "Difficult lighting",
    "oblique_view": "Oblique view",
    "partial_out_of_frame": "Partial framing",
    "finger_self_occlusion": "Finger overlap",
}


def render(cases_dir, output_dir):
    cases_dir = cases_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with (cases_dir / "figure_cases.csv").open(encoding="utf-8-sig", newline="") as stream:
        panels = {(r["source_pool"], r["category"]): r for r in csv.DictReader(stream)}

    plt.rcParams.update({"font.family": "Times New Roman", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.08, 4.54), facecolor="white")
    centers = [0.134, 0.378, 0.622, 0.866]
    width = 0.186
    height = width * 7.08 / 4.54

    for col, (_, label) in enumerate(TITLES.items()):
        fig.text(centers[col], 0.965, label, fontsize=10.5, weight="semibold", ha="center", va="top")

    for pool, kicker, top in (
        ("letter-stratified screen", "Letter-stratified screen", 0.84),
        ("error-enriched screen", "Error-enriched screen", 0.397),
    ):
        fig.text(0.026, top + 0.045, kicker, fontsize=9.7, weight="semibold", va="center")
        for col, key in enumerate(TITLES):
            x = centers[col]
            row = panels[(pool, key)]
            original = next((cases_dir / "originals").glob(row["case_id"] + ".*"))
            with Image.open(original) as image:
                ax = fig.add_axes((x - width / 2, top - height, width, height))
                ax.imshow(image.convert("RGB"), interpolation="nearest")
                ax.axis("off")
            detection = "Yes" if row["hand_detected"] == "1" else "No"
            fig.text(
                x, top - height - 0.019,
                f"ID {row['original_index']}   True {row['true_label']}   MP {detection}",
                fontsize=8.15, ha="center", va="top",
            )
            fig.text(
                x, top - height - 0.057,
                "    ".join(f"{m} {row[m]}" for m in ("E7", "E2", "E5")),
                fontsize=8.4, ha="center", va="top",
            )

    pdf_path = output_dir / "real_challenging_cases.pdf"
    png_path = output_dir / "real_challenging_cases.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=255)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")


def parse_args():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path,
                        default=root / "results/reviewer11_14/r11_original_cases",
                        help="Directory containing figure_cases.csv and originals/.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for real_challenging_cases.pdf/.png (defaults to --cases-dir).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render(args.cases_dir, args.output_dir or args.cases_dir)
