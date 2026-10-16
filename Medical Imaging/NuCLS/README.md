# NuCLS cell classification

Research workflow for predicting cell types from histology images and spatial cell graphs.

## Workflow

1. Merge NuCLS annotations into a cell table.
2. Build spatial graphs and inspect their connectivity.
3. Extract pretrained ResNet-18 features from cell image crops.
4. Train an MLP, GraphSAGE, or graph transformer and evaluate on separate patients.

## Setup

Use Python 3.11 or later and install dependencies in a virtual environment:

```sh
pip install torch torchvision torch-geometric pandas numpy scikit-learn matplotlib pillow tqdm jupyterlab
```

Place NuCLS annotations in `data/nucls/NuCLS_dataset/csv/` and tissue images in `data/nucls/NuCLS_dataset/rgb/`. Run from the project root:

```sh
python scripts/build_nucls_cells.py
jupyter lab notebooks/nucls_image_embeddings_gnn.ipynb
```

The notebook downloads pretrained weights on first use, caches cell embeddings, and saves metrics and checkpoints in `results/nucls_image_graph/`. It classifies annotated cells; it does not detect new cells. Its patient split is an internal experiment, not the official NuCLS benchmark split.

## Files

- `scripts/`: cell-table preparation, graph construction, distance checks, and visualization.
- `notebooks/nucls_image_embeddings_gnn.ipynb`: image features, models, evaluation, and results commentary.
- `data/`: source NuCLS and Graph-BRACS datasets.
- `processed/`: generated tables, graphs, and feature caches.
- `results/`: experiment outputs.

Graph-BRACS is available as a separate graph-classification dataset; the current notebook uses NuCLS. Datasets and generated outputs are excluded from Git.
