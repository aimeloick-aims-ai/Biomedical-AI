from pathlib import Path
import pandas as pd
import numpy as np

INPUT_DIR = Path("data/nucls/NuCLS_dataset/csv")
OUTPUT_FILE = Path("processed/nucls/cells.csv")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

all_cells = []

for csv_file in INPUT_DIR.glob("*.csv"):

    df = pd.read_csv(csv_file)

    # Identifiant de l'image
    image_id = csv_file.stem

    df["image_id"] = image_id

    # Identifiant local de cellule
    df["cell_id"] = np.arange(len(df))

    # Centre de la bounding box
    df["center_x"] = (df["xmin"] + df["xmax"]) / 2.0
    df["center_y"] = (df["ymin"] + df["ymax"]) / 2.0

    # Dimensions de la bounding box
    df["bbox_width"] = df["xmax"] - df["xmin"]
    df["bbox_height"] = df["ymax"] - df["ymin"]

    # Aire approximative bounding box
    df["bbox_area"] = df["bbox_width"] * df["bbox_height"]

    keep_columns = [
        "image_id",
        "cell_id",
        "raw_classification",
        "main_classification",
        "super_classification",
        "type",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "center_x",
        "center_y",
        "bbox_width",
        "bbox_height",
        "bbox_area",
        "coords_x",
        "coords_y",
    ]

    df = df[keep_columns]

    all_cells.append(df)

cells = pd.concat(all_cells, ignore_index=True)

cells.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")
print(f"Total cells: {len(cells):,}")
print(f"Total images: {cells['image_id'].nunique():,}")

print("\nMain classes:")
print(cells["main_classification"].value_counts())

print("\nSuper classes:")
print(cells["super_classification"].value_counts())