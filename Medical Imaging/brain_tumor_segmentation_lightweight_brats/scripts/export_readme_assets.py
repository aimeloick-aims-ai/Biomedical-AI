"""Export the saved notebook figures for the GitHub README (standard library only).

Run from any directory: python scripts/export_readme_assets.py
This reads saved outputs; it never executes notebooks or trains a model.
"""

import base64
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
FIGURES = [
    ("01_brats_africa_preparation.ipynb", 38, 0, "data/mri_sequences.png", "Four MRI sequences, reference mask, and overlay"),
    ("01_brats_africa_preparation.ipynb", 41, 0, "data/three_planes.png", "Reference annotation in three anatomical planes"),
    ("01_brats_africa_preparation.ipynb", 43, 0, "data/tumor_regions.png", "Each annotated tumor class shown separately"),
    ("01_brats_africa_preparation.ipynb", 45, 0, "data/lesion_slices.png", "Several slices through the annotated lesion"),
    ("01_brats_africa_preparation.ipynb", 66, 0, "data/preprocessed_case.png", "Example after intensity scaling and cropping"),
    ("02_unet3d_pytorch_brain_segmentation.ipynb", 38, 0, "training/learning_curves.png", "Saved training and validation curves"),
    ("02_unet3d_pytorch_brain_segmentation.ipynb", 45, 0, "training/test_prediction.png", "Whole-volume inference displayed on a test slice"),
    ("03_explain_unet_pytorch.ipynb", 9, 0, "xai/reference_annotation.png", "MRI and reference tumor annotations"),
    ("03_explain_unet_pytorch.ipynb", 11, 0, "xai/segmentation_errors.png", "Selected-class correct, missed, and extra regions"),
    ("03_explain_unet_pytorch.ipynb", 13, 0, "xai/target_region.png", "Fixed output ROI and class probability"),
    ("03_explain_unet_pytorch.ipynb", 17, 0, "xai/occlusion_example.png", "One 3D input block masked: before and after"),
    ("03_explain_unet_pytorch.ipynb", 19, 0, "xai/occlusion_map.png", "Spatial occlusion effects in percentage points"),
    ("03_explain_unet_pytorch.ipynb", 21, 0, "xai/input_modalities.png", "Four input modalities for the XAI example"),
    ("03_explain_unet_pytorch.ipynb", 21, 1, "xai/modality_ablation.png", "Fixed-region score after removing a whole modality"),
    ("03_explain_unet_pytorch.ipynb", 25, 0, "xai/gradient_sensitivity.png", "Gradient x input and SmoothGrad magnitude"),
    ("03_explain_unet_pytorch.ipynb", 25, 1, "xai/signed_gradients.png", "Signed gradient explanations per MRI modality"),
    ("04_lightweight_architecture_personal.ipynb", 15, 0, "architecture/parameter_counts.png", "Parameter counts for five U-Net variants"),
    ("04_lightweight_architecture_personal.ipynb", 28, 0, "architecture/size_vs_dice.png", "Parameter count versus short-run validation Dice"),
]


def main():
    # Validate every requested output before replacing any exported image.
    cache = {}
    exports = []
    for name, index, image_index, relative, description in FIGURES:
        if name not in cache:
            path = ROOT / "notebooks" / name
            raw = path.read_bytes()
            cache[name] = (json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest())
        notebook, notebook_sha = cache[name]
        cell = notebook["cells"][index]
        images = [o["data"]["image/png"] for o in cell.get("outputs", [])
                  if "image/png" in o.get("data", {})]
        if image_index >= len(images):
            raise ValueError(f"Missing PNG: {name}, cell {index}, image {image_index}. Save executed notebook outputs first.")
        encoded = images[image_index]
        if isinstance(encoded, list):
            encoded = "".join(encoded)
        content = base64.b64decode(encoded, validate=False)
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Invalid PNG output: {name}, cell {index}")
        exports.append((relative, content, {
            "file": relative,
            "description": description,
            "source_notebook": f"notebooks/{name}",
            "source_notebook_sha256": notebook_sha,
            "cell_index_zero_based": index,
            "cell_id": cell.get("id"),
            "image_index_in_cell": image_index,
            "execution_count": cell.get("execution_count"),
            "sha256": hashlib.sha256(content).hexdigest(),
        }))

    ASSETS.mkdir(parents=True, exist_ok=True)
    manifest = []
    for relative, content, entry in exports:
        target = ASSETS / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest.append(entry)
    (ASSETS / "manifest.json").write_text(json.dumps({
        "origin": "Saved notebook outputs; no notebook execution or image synthesis",
        "figures": manifest,
        "diagram": {"file": "pipeline.svg", "origin": "Hand-authored overview of the four local notebooks"},
    }, indent=2) + "\n", encoding="utf-8")

    gallery = [
        "# Figure gallery and provenance",
        "",
        "These 18 PNG figures are exported directly from saved notebook outputs. "
        "They are local experiment results, not generated illustrations. "
        "The pipeline SVG is a schematic of the workflow.",
        "",
        "MRI examples use BraTS-Africa annotations. See the [project README](../README.md) "
        "for dataset and tutorial credits, result interpretation, and limitations.",
        "",
        "Regenerate PNG files and their [provenance manifest](manifest.json) from the project root:",
        "",
        "```bash", "python scripts/export_readme_assets.py", "```",
        "",
        "The script uses Python's standard library and does not execute the notebooks. "
        "Save notebook outputs before exporting. Cell indices in the manifest are zero-based; "
        "update the mapping if cells are inserted, removed, or reordered.",
        "",
        "## Workflow", "", "![Four-notebook workflow](pipeline.svg)",
    ]
    for entry in manifest:
        gallery.extend([
            "", f"## {entry['description']}", "",
            f"![{entry['description']}]({entry['file']})", "",
            f"Source: [{Path(entry['source_notebook']).name}](../{entry['source_notebook']}), "
            f"cell {entry['cell_index_zero_based']} (zero-based), image {entry['image_index_in_cell']}.",
        ])
    (ASSETS / "README.md").write_text("\n".join(gallery) + "\n", encoding="utf-8")
    print(f"Exported {len(exports)} notebook figures to {ASSETS}")


if __name__ == "__main__":
    main()
