import pandas as pd


def write_split_manifest(csv_path, data, split_indices, split_name, seed, append=False):
    mode = "a" if append else "w"
    header = not append
    rows = []
    for original_index in split_indices:
        row = data.iloc[int(original_index)]
        rows.append(
            {
                "image_path": row.get("image_path", ""),
                "label": row["label"],
                "split": split_name,
                "seed": seed,
                "original_index": int(original_index),
                "hand_detected": row.get("hand_detected", 1),
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, mode=mode, header=header, index=False)
