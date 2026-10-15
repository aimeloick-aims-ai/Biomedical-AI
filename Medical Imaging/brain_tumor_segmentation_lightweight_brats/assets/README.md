# Figure gallery and provenance

These 18 PNG figures are exported directly from saved notebook outputs. They are local experiment results, not generated illustrations. The pipeline SVG is a schematic of the workflow.

MRI examples use BraTS-Africa annotations. See the [project README](../README.md) for dataset and tutorial credits, result interpretation, and limitations.

Regenerate PNG files and their [provenance manifest](manifest.json) from the project root:

```bash
python scripts/export_readme_assets.py
```

The script uses Python's standard library and does not execute the notebooks. Save notebook outputs before exporting. Cell indices in the manifest are zero-based; update the mapping if cells are inserted, removed, or reordered.

## Workflow

![Four-notebook workflow](pipeline.svg)

## Four MRI sequences, reference mask, and overlay

![Four MRI sequences, reference mask, and overlay](data/mri_sequences.png)

Source: [01_brats_africa_preparation.ipynb](../notebooks/01_brats_africa_preparation.ipynb), cell 38 (zero-based), image 0.

## Reference annotation in three anatomical planes

![Reference annotation in three anatomical planes](data/three_planes.png)

Source: [01_brats_africa_preparation.ipynb](../notebooks/01_brats_africa_preparation.ipynb), cell 41 (zero-based), image 0.

## Each annotated tumor class shown separately

![Each annotated tumor class shown separately](data/tumor_regions.png)

Source: [01_brats_africa_preparation.ipynb](../notebooks/01_brats_africa_preparation.ipynb), cell 43 (zero-based), image 0.

## Several slices through the annotated lesion

![Several slices through the annotated lesion](data/lesion_slices.png)

Source: [01_brats_africa_preparation.ipynb](../notebooks/01_brats_africa_preparation.ipynb), cell 45 (zero-based), image 0.

## Example after intensity scaling and cropping

![Example after intensity scaling and cropping](data/preprocessed_case.png)

Source: [01_brats_africa_preparation.ipynb](../notebooks/01_brats_africa_preparation.ipynb), cell 66 (zero-based), image 0.

## Saved training and validation curves

![Saved training and validation curves](training/learning_curves.png)

Source: [02_unet3d_pytorch_brain_segmentation.ipynb](../notebooks/02_unet3d_pytorch_brain_segmentation.ipynb), cell 38 (zero-based), image 0.

## Whole-volume inference displayed on a test slice

![Whole-volume inference displayed on a test slice](training/test_prediction.png)

Source: [02_unet3d_pytorch_brain_segmentation.ipynb](../notebooks/02_unet3d_pytorch_brain_segmentation.ipynb), cell 45 (zero-based), image 0.

## MRI and reference tumor annotations

![MRI and reference tumor annotations](xai/reference_annotation.png)

Source: [03_explain_unet_pytorch.ipynb](../notebooks/03_explain_unet_pytorch.ipynb), cell 9 (zero-based), image 0.

## Selected-class correct, missed, and extra regions

![Selected-class correct, missed, and extra regions](xai/segmentation_errors.png)

Source: [03_explain_unet_pytorch.ipynb](../notebooks/03_explain_unet_pytorch.ipynb), cell 11 (zero-based), image 0.

## Fixed output ROI and class probability

![Fixed output ROI and class probability](xai/target_region.png)

Source: [03_explain_unet_pytorch.ipynb](../notebooks/03_explain_unet_pytorch.ipynb), cell 13 (zero-based), image 0.

## One 3D input block masked: before and after

![One 3D input block masked: before and after](xai/occlusion_example.png)

Source: [03_explain_unet_pytorch.ipynb](../notebooks/03_explain_unet_pytorch.ipynb), cell 17 (zero-based), image 0.

## Spatial occlusion effects in percentage points

![Spatial occlusion effects in percentage points](xai/occlusion_map.png)

Source: [03_explain_unet_pytorch.ipynb](../notebooks/03_explain_unet_pytorch.ipynb), cell 19 (zero-based), image 0.

## Four input modalities for the XAI example

![Four input modalities for the XAI example](xai/input_modalities.png)

Source: [03_explain_unet_pytorch.ipynb](../notebooks/03_explain_unet_pytorch.ipynb), cell 21 (zero-based), image 0.

## Fixed-region score after removing a whole modality

![Fixed-region score after removing a whole modality](xai/modality_ablation.png)

Source: [03_explain_unet_pytorch.ipynb](../notebooks/03_explain_unet_pytorch.ipynb), cell 21 (zero-based), image 1.

## Gradient x input and SmoothGrad magnitude

![Gradient x input and SmoothGrad magnitude](xai/gradient_sensitivity.png)

Source: [03_explain_unet_pytorch.ipynb](../notebooks/03_explain_unet_pytorch.ipynb), cell 25 (zero-based), image 0.

## Signed gradient explanations per MRI modality

![Signed gradient explanations per MRI modality](xai/signed_gradients.png)

Source: [03_explain_unet_pytorch.ipynb](../notebooks/03_explain_unet_pytorch.ipynb), cell 25 (zero-based), image 1.

## Parameter counts for five U-Net variants

![Parameter counts for five U-Net variants](architecture/parameter_counts.png)

Source: [04_lightweight_architecture_personal.ipynb](../notebooks/04_lightweight_architecture_personal.ipynb), cell 15 (zero-based), image 0.

## Parameter count versus short-run validation Dice

![Parameter count versus short-run validation Dice](architecture/size_vs_dice.png)

Source: [04_lightweight_architecture_personal.ipynb](../notebooks/04_lightweight_architecture_personal.ipynb), cell 28 (zero-based), image 0.
