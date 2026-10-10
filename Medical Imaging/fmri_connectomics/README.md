<p align="center">
  <img src="assets/connectomics_banner_notebook.png" alt="fMRI connectomics notebook results" width="100%" />
</p>

# fMRI Connectomics: from regional BOLD signals to predictive networks

This notebook follows the full path from **fMRI time series** to a **functional connectome**, then asks a simple but useful question: can patterns of functional connectivity help distinguish **children from adults**?

What makes the notebook interesting is that it does not stop at one connectivity matrix. It moves through several representations of the same data: regional BOLD signals, correlation and partial-correlation matrices, thresholded graphs, tangent-space features, and finally a linear SVM whose strongest weighted connections can be mapped back onto the brain.

The figures below are the **actual outputs produced by the notebook**.

## What the fMRI signal represents

Functional MRI does not directly record neuronal firing. It measures the **BOLD (blood-oxygen-level-dependent) signal**, which reflects changes in blood oxygenation associated with neural activity. Each scan is therefore a sequence of 3D brain volumes collected over time.

Instead of modelling every voxel independently, this notebook uses the **Schaefer 2018 atlas with 100 cortical regions and 7 networks**. For each subject, the voxel-level fMRI data are reduced to one cleaned time series per region. The resulting subject representation is a matrix of:

```text
number of time points × 100 regions
```

Those regional trajectories are then compared pairwise to build a functional network.

## 1. Brain parcellation

<p align="center">
  <img src="assets/schaefer_atlas.png" alt="Schaefer 2018 atlas with 100 ROIs" width="70%" />
</p>

The atlas gives every subject the same set of **100 cortical nodes**. This common parcellation is essential because all later connectivity matrices must refer to the same regions in the same order.

The notebook uses **50 subjects** from Nilearn's development fMRI dataset:

- **11 adults**
- **39 children**

The classification target is encoded as `0 = Adult` and `1 = Child`.

## 2. From time series to connectivity matrices

After confound regression and regional signal extraction with `NiftiLabelsMasker`, the notebook computes two forms of connectivity.

<p align="center">
  <img src="assets/connectivity_matrices.png" alt="Correlation and partial correlation matrices from subject 0" width="82%" />
</p>

**Correlation** captures how strongly two regional BOLD signals fluctuate together. The matrix is relatively dense because many regions share common temporal variation.

**Partial correlation** asks a more conditional question: how strongly are two regions associated after accounting for the remaining regional signals? The resulting matrix is much sparser, which can be seen directly in the notebook output.

For 100 ROIs, each matrix is `100 × 100`. Rather than reading thousands of entries individually, the same matrix can be projected back onto brain coordinates.

## 3. Looking at the connectome anatomically

<p align="center">
  <img src="assets/subject_connectome.png" alt="Subject 0 connectome showing top 5 percent of edges" width="78%" />
</p>

The figure keeps only the **top 5% of edges** for visualization. This is not a new model; it is a way of seeing where the strongest functional relationships of one subject are located across sagittal, coronal, and axial views.

The colored nodes correspond to atlas regions, while the lines represent functional connections. Positive and negative connections are encoded by the connectome color scale.

## 4. Treating the connectome as a graph

The notebook also turns the first subject's correlation matrix into a binary graph by removing self-connections and retaining edges satisfying:

```text
|correlation| > 0.4
```

<p align="center">
  <img src="assets/thresholded_graph.png" alt="Thresholded functional connectivity graph for subject 0" width="48%" />
</p>

This graph representation makes it possible to inspect network organization rather than only pairwise correlations. For this subject, the notebook reports:

| Graph quantity | Result |
|---|---:|
| Average clustering coefficient | **0.513** |
| Highest-degree nodes | **58 (26), 74 (24), 87 (23)** |
| Highest-betweenness nodes | **33, 73, 64** |
| Modularity | **0.485** |
| Detected communities | **5** |

The spring-layout drawing is only a graph visualization; node positions in that panel are not anatomical coordinates.

## 5. Classification in tangent space

For prediction, the notebook switches from ordinary correlation matrices to **tangent-space connectivity**. Tangent connectivity is useful because covariance-like matrices live on a curved geometric space; the tangent representation maps them into a vector space where standard machine-learning models can be applied more naturally.

With 100 ROIs and the diagonal retained by the current `ConnectivityMeasure(vectorize=True)` configuration, the fitted representation contains **5,050 features**. Univariate selection then keeps the **100 most informative features** before fitting a linear SVM.

The complete strict pipeline is:

```text
regional BOLD time series
        ↓
Tangent ConnectivityMeasure
        ↓
StandardScaler
        ↓
SelectKBest(f_classif, k=100)
        ↓
Linear SVM
```

## 6. Why the validation design matters

One of the most important parts of the notebook is the comparison between two ways of evaluating the classifier.

In the first version, tangent connectivity is fitted on **all subjects** and feature selection also sees **all labels** before cross-validation. This lets information from the future test folds influence the representation and selected features.

The strict version instead places the complete sequence inside a scikit-learn `Pipeline`. Each fold therefore learns the tangent reference, scaling, feature selection, and classifier using only its training subjects.

The difference is visible in the results:

| Evaluation | AUROC | AUPRC |
|---|---:|---:|
| Global preprocessing / leakage | **0.990 ± 0.017** | **0.996 ± 0.006** |
| Strict fold-wise pipeline | **0.952 ± 0.083** | **0.975 ± 0.044** |

The strict scores are still high, but the comparison shows why apparently small preprocessing decisions can make performance look more optimistic than it really is.

The intervals shown here are the notebook's approximate 95% intervals computed from the five cross-validation scores.

## 7. Which connections drive the linear classifier?

After validation, the strict pipeline is fitted to the complete dataset to inspect the coefficients of the linear SVM. The 100 selected coefficients are restored to the full tangent-connectivity feature space, and the strongest **1% of model weights** are displayed as a connectome.

<p align="center">
  <img src="assets/svm_discriminative_edges.png" alt="Top 1 percent of SVM connectivity weights" width="78%" />
</p>

This is the notebook's most direct link between machine learning and connectomics: the classifier is no longer only a score, but a set of weighted connections distributed across the cortical network.

These edges should be interpreted as **features contributing strongly to the fitted decision boundary**, not as causal developmental connections. Their sign and magnitude describe the linear model after tangent transformation, scaling, and feature selection.

## The notebook at a glance

| Step | What happens | Main output |
|---|---|---|
| 1 | Fetch Schaefer 2018 atlas | 100 cortical ROIs |
| 2 | Load 50 development fMRI subjects | 11 adults, 39 children |
| 3 | Extract and clean regional BOLD signals | one time series per ROI |
| 4 | Estimate functional connectivity | correlation + partial correlation matrices |
| 5 | Visualize strongest subject-level connections | top-5% anatomical connectome |
| 6 | Threshold the matrix and build a graph | clustering, hubs, betweenness, modularity |
| 7 | Encode connectomes in tangent space | 5,050 connectivity features |
| 8 | Select features and train linear SVM | 100 selected features |
| 9 | Compare leaky and strict CV workflows | AUROC + AUPRC |
| 10 | Map fitted SVM weights back to the network | top-1% discriminative edges |

## Why this small notebook is useful

The dataset is deliberately small, so this is not intended as a definitive developmental-neuroscience study. Its value is in putting several ideas that are often taught separately into one reproducible workflow:

**fMRI → parcellation → BOLD time series → functional connectivity → graph analysis → tangent-space machine learning → leakage-aware validation → connectome interpretation.**

It also shows an important practical lesson: in connectomics, preprocessing is part of the model. A transformation such as tangent connectivity may look like a harmless feature-construction step, but because its group reference is estimated from data, it belongs inside cross-validation when performance is being evaluated.

## Running the notebook

The main Python packages are:

```bash
pip install nilearn numpy pandas matplotlib seaborn networkx scikit-learn
```

Then run:

```text
AIMS_09_fmri_connectomics.ipynb
```

Nilearn downloads the Schaefer atlas and development fMRI dataset on first use. Regional signal extraction is the slowest stage, so caching is useful when re-running the notebook.

## Repository structure

```text
.
├── AIMS_09_fmri_connectomics.ipynb
├── README.md
└── assets/
    ├── connectomics_banner_notebook.png
    ├── schaefer_atlas.png
    ├── connectivity_matrices.png
    ├── subject_connectome.png
    ├── thresholded_graph.png
    └── svm_discriminative_edges.png
```

---

All scientific figures shown in this README come directly from the notebook outputs rather than from generic stock or generated neuroscience illustrations.
