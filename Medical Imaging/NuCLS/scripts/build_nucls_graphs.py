from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# ============================================================
# Paths
# ============================================================

INPUT_FILE = Path("processed/nucls/cells.csv")
OUTPUT_DIR = Path("processed/nucls/graphs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Graph parameters
# ============================================================

K = 5
MAX_DISTANCE = 120.0

# ============================================================
# Load cells
# ============================================================

df = pd.read_csv(INPUT_FILE)

# Keep all cells in the graph,
# but mark which ones have reliable labels
# for supervised concept computation
df["usable_for_concepts"] = ~df["main_classification"].isin(
    ["AMBIGUOUS", "unlabeled"]
)

print(f"Total cells: {len(df):,}")
print(f"Total images: {df['image_id'].nunique():,}")
print(
    f"Usable for concepts: "
    f"{df['usable_for_concepts'].sum():,}"
)

# ============================================================
# Build one graph per image
# ============================================================

for idx, (image_id, group) in enumerate(df.groupby("image_id")):

    group = group.reset_index(drop=True)

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    positions = group[
        ["center_x", "center_y"]
    ].to_numpy(dtype=np.float32)

    # Basic numerical node features
    node_features = group[
        [
            "bbox_width",
            "bbox_height",
            "bbox_area",
        ]
    ].to_numpy(dtype=np.float32)

    n_nodes = len(group)

    # Skip images with fewer than 2 cells
    if n_nodes < 2:
        print(
            f"Skipping {image_id}: "
            f"only {n_nodes} node"
        )
        continue

    # --------------------------------------------------------
    # KNN graph
    # --------------------------------------------------------

    # +1 because the first nearest neighbour is the node itself
    n_neighbors = min(K + 1, n_nodes)

    knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean"
    )

    knn.fit(positions)

    distances, indices = knn.kneighbors(positions)

    edges = []
    edge_distances = []

    for source in range(n_nodes):

        for j in range(1, n_neighbors):

            target = indices[source, j]
            distance = distances[source, j]

            # Keep only spatially local neighbours
            if distance <= MAX_DISTANCE:

                edges.append(
                    [source, target]
                )

                edge_distances.append(
                    distance
                )

    # --------------------------------------------------------
    # Handle graphs with zero edges
    # --------------------------------------------------------

    if len(edges) == 0:

        edge_index = np.empty(
            (2, 0),
            dtype=np.int64
        )

        edge_distance = np.empty(
            (0,),
            dtype=np.float32
        )

    else:

        edge_index = np.array(
            edges,
            dtype=np.int64
        ).T

        edge_distance = np.array(
            edge_distances,
            dtype=np.float32
        )

    # --------------------------------------------------------
    # Labels / metadata
    # --------------------------------------------------------

    cell_types = group[
        "main_classification"
    ].astype(str).to_numpy()

    usable_for_concepts = group[
        "usable_for_concepts"
    ].to_numpy(dtype=bool)

    # --------------------------------------------------------
    # Save graph
    # --------------------------------------------------------

    output_file = OUTPUT_DIR / f"{image_id}.npz"

    np.savez_compressed(

        output_file,

        # Node features
        x=node_features,

        # Spatial coordinates
        pos=positions,

        # Graph connectivity
        edge_index=edge_index,

        # Edge length
        edge_distance=edge_distance,

        # Cell labels
        cell_type=cell_types,

        # Reliable labels for concepts
        usable_for_concepts=usable_for_concepts,

        # Image identifier
        image_id=image_id,
    )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if idx % 100 == 0:

        print(
            f"{idx} graphs processed | "
            f"{image_id} | "
            f"{n_nodes} nodes | "
            f"{edge_index.shape[1]} edges"
        )

# ============================================================
# Done
# ============================================================

print("\nDone.")
print(f"Graphs saved in: {OUTPUT_DIR}")