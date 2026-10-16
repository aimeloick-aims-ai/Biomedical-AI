from pathlib import Path
import numpy as np
import pandas as pd

GRAPH_DIR = Path("processed/nucls/graphs")
OUTPUT_FILE = Path("processed/nucls/edge_distance_stats.csv")

rows = []
all_distances = []

n_graphs = 0
n_zero_edge_graphs = 0

for graph_file in GRAPH_DIR.glob("*.npz"):

    g = np.load(graph_file, allow_pickle=True)

    image_id = g["image_id"].item()
    edge_index = g["edge_index"]
    distances = g["edge_distance"]

    n_nodes = g["x"].shape[0]

    if edge_index.ndim == 2:
        n_edges = edge_index.shape[1]
    else:
        n_edges = 0

    n_graphs += 1

    # ----------------------------------------------------
    # Graphs with zero edges
    # ----------------------------------------------------

    if len(distances) == 0:

        n_zero_edge_graphs += 1

        rows.append({
            "image_id": image_id,
            "n_nodes": n_nodes,
            "n_edges": 0,
            "mean_edge_distance": np.nan,
            "median_edge_distance": np.nan,
            "max_edge_distance": np.nan,
            "p95_edge_distance": np.nan,
        })

        continue

    # ----------------------------------------------------
    # Graphs with edges
    # ----------------------------------------------------

    all_distances.extend(distances.tolist())

    rows.append({
        "image_id": image_id,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "mean_edge_distance": float(distances.mean()),
        "median_edge_distance": float(np.median(distances)),
        "max_edge_distance": float(distances.max()),
        "p95_edge_distance": float(np.percentile(distances, 95)),
    })

# --------------------------------------------------------
# Save per-graph statistics
# --------------------------------------------------------

stats = pd.DataFrame(rows)
stats.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------------
# Global statistics
# --------------------------------------------------------

all_distances = np.array(
    all_distances,
    dtype=np.float32
)

print("\n=== GRAPH SUMMARY ===")
print(f"Total graphs      : {n_graphs:,}")
print(f"Zero-edge graphs  : {n_zero_edge_graphs:,}")

if n_graphs > 0:
    print(
        f"Zero-edge percent : "
        f"{100 * n_zero_edge_graphs / n_graphs:.2f}%"
    )

if len(all_distances) > 0:

    print("\n=== GLOBAL EDGE DISTANCES ===")
    print(f"Number of edges : {len(all_distances):,}")
    print(f"Mean            : {all_distances.mean():.2f}")
    print(f"Median          : {np.median(all_distances):.2f}")
    print(f"75th percentile : {np.percentile(all_distances, 75):.2f}")
    print(f"90th percentile : {np.percentile(all_distances, 90):.2f}")
    print(f"95th percentile : {np.percentile(all_distances, 95):.2f}")
    print(f"99th percentile : {np.percentile(all_distances, 99):.2f}")
    print(f"Maximum         : {all_distances.max():.2f}")

else:

    print("\nNo edges found in any graph.")

print(f"\nSaved: {OUTPUT_FILE}")