<p align="center">
  <img src="complete_assets/connectomics_abide_banner.png" alt="fMRI connectomics and ABIDE GNN notebook outputs" width="100%" />
</p>

# fMRI Connectomics: from regional BOLD signals to predictive networks and graph neural networks

This repository develops the same core idea at two different scales: **turn resting-state fMRI into a functional network, then ask what can be learned from the network without losing sight of how it was constructed**.

The first notebook is a compact developmental-connectomics study built around 50 subjects, the Schaefer atlas, graph analysis, tangent-space connectivity, and leakage-aware linear classification. The second notebook moves to **500 quality-checked ABIDE subjects**, uses the **116-region AAL atlas**, and compares a conventional SVM with a **Graph Attention Network (GAT)** for ASD-versus-control classification.

The two notebooks are complementary rather than interchangeable:

| Notebook | Cohort / task | Representation | Main model | Main lesson |
|---|---|---|---|---|
| `AIMS_09_fmri_connectomics.ipynb` | 50 development-fMRI subjects; child vs adult | Schaefer-100 connectivity, graph metrics, tangent space | Linear SVM | Data-dependent connectome preprocessing must stay inside validation |
| `fmri_gnns_abide.ipynb` | 500 ABIDE subjects; ASD vs control | AAL-116 partial-correlation connectomes | RBF-SVM + GAT | Explicit graph structure does not guarantee better prediction |

---

## Part I — Developmental fMRI connectomics

### From BOLD signals to a common cortical network

Functional MRI measures the **BOLD (blood-oxygen-level-dependent) signal**, not neuronal firing directly. The first notebook reduces each subject's fMRI sequence to regional time series using the **Schaefer 2018 atlas with 100 cortical regions and 7 networks**. Each subject is therefore represented as:

```text
number of time points × 100 regions
```

The notebook uses **50 subjects** from Nilearn's development fMRI dataset:

- 11 adults
- 39 children
- target encoding: `0 = Adult`, `1 = Child`

It then estimates both **correlation** and **partial correlation** connectivity matrices. A thresholded correlation graph (`|r| > 0.4`) is used for descriptive network analysis.

For the example graph shown in the original notebook, the reported graph statistics are:

| Quantity | Result |
|---|---:|
| Average clustering coefficient | **0.513** |
| Highest-degree nodes | **58 (26), 74 (24), 87 (23)** |
| Highest-betweenness nodes | **33, 73, 64** |
| Modularity | **0.485** |
| Detected communities | **5** |

The spring-layout graph is a network abstraction; its node positions are not anatomical coordinates.

### Tangent-space classification and leakage

For prediction, the notebook uses **tangent-space connectivity**. With 100 ROIs and the diagonal retained by the current vectorised configuration, the representation contains **5,050 features**. `SelectKBest(f_classif)` keeps 100 features before a linear SVM is fitted.

The strict pipeline is:

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

The notebook deliberately contrasts a globally preprocessed workflow with fold-wise preprocessing:

| Evaluation | AUROC | AUPRC |
|---|---:|---:|
| Global preprocessing / leakage | **0.990 ± 0.017** | **0.996 ± 0.006** |
| Strict fold-wise pipeline | **0.952 ± 0.083** | **0.975 ± 0.044** |

The point is methodological: the tangent reference and feature selection are learned from data. If they are estimated before cross-validation, held-out subjects can influence the representation. In the strict workflow, the complete transformation sequence is refitted inside each training fold.

The final linear-SVM weights can be mapped back to the connectome to show the strongest model-weighted edges. Those edges are **model features**, not causal developmental connections.

---

# Part II — ABIDE: SVM versus Graph Attention Network

## 1. Cohort and regional time series

The second notebook uses the **Autism Brain Imaging Data Exchange (ABIDE)** through Nilearn's preprocessed ABIDE PCP interface. It requests **500 quality-checked subjects** with AAL regional time series.

The downloaded cohort contains:

- **242 ASD participants**
- **258 controls**
- **410 males**
- **90 females**

The notebook encodes `0 = ASD` and `1 = Control`. The first subject has 196 time points across **116 AAL regions**.

<p align="center">
  <img src="complete_assets/abide_cohort_distribution.png" alt="ABIDE diagnosis distribution and diagnosis by sex" width="88%" />
</p>

The split is stratified jointly by diagnosis and sex:

| Split | Subjects | ASD | Control | Male | Female |
|---|---:|---:|---:|---:|---:|
| Train | **374** | 181 | 193 | 307 | 67 |
| Validation | **76** | 37 | 39 | 62 | 14 |
| Test | **50** | 24 | 26 | 41 | 9 |

The test set is held out first, and the same subject indices are reused by the SVM and GAT experiments.

## 2. Partial-correlation connectomes

`ConnectivityMeasure(kind='partial correlation')` produces one **116 × 116** matrix per subject, giving an array of shape:

```text
(500, 116, 116)
```

Partial correlation is used here to describe the association between two regional signals after accounting for the remaining regions.

<p align="center">
  <img src="complete_assets/abide_subject_partial_connectivity.png" alt="Partial-correlation matrices for one ASD and one control subject" width="92%" />
</p>

A group-level descriptive comparison averages matrices separately for ASD and control participants and computes `Control − ASD`.

<p align="center">
  <img src="complete_assets/abide_group_connectivity_comparison.png" alt="Mean ASD connectivity, mean control connectivity, and control-minus-ASD difference" width="100%" />
</p>

The notebook reports:

- maximum absolute group-mean difference: **0.0359**
- mean absolute group-mean difference: **0.0040**

These maps are descriptive. They are not edge-wise significance tests and should not be interpreted as diagnostic biomarkers by themselves.

## 3. Classical baseline: vectorised connectome + RBF-SVM

Because each connectivity matrix is symmetric, the SVM uses only the upper triangular entries excluding the diagonal:

```text
116 × 115 / 2 = 6,670 unique connections
```

The resulting tabular matrix has shape **500 × 6,670**. A `StandardScaler` and RBF-kernel `SVC(C=1.0, probability=True)` are fitted on the training set.

### SVM results

| Split | Accuracy | Balanced accuracy | ROC-AUC |
|---|---:|---:|---:|
| Validation | **0.5263** | **0.5184** | **0.5925** |
| Test | **0.6400** | **0.6266** | **0.7420** |

The notebook also reports F1 = 0.6400 on validation and 0.7353 on test. Because label `1` is **Control**, scikit-learn's default binary F1 here is the **control-class F1**, not the ASD-class F1.

<p align="center">
  <img src="complete_assets/abide_svm_test_confusion_matrix.png" alt="SVM confusion matrix on the held-out ABIDE test set" width="46%" />
</p>

The held-out ROC-AUC is clearly better than chance in this particular split, but the gap between validation and test performance is a reminder that a single split can be variable, especially in a heterogeneous multi-site dataset.

## 4. Turning each connectome into a graph

The GNN branch keeps network structure explicit. Each subject becomes a PyTorch Geometric `Data` object with:

- **116 nodes** — AAL brain regions;
- **edges** — retained partial-correlation connections;
- **edge attributes** — signed partial-correlation values;
- **graph label** — ASD or control;
- **7 node features** per region.

The seven node features are:

```text
3 anatomical coordinates (normalised MNI x, y, z)
+ 4 subject-specific connectivity descriptors
  - absolute node strength
  - thresholded degree
  - mean connectivity
  - connectivity standard deviation
```

The subject-specific features are z-scored within each subject. The coordinates are min-max normalised across atlas regions.

### Important implementation detail: graph threshold

The explanatory markdown in the notebook mentions `0.3`, but the executed graph construction actually sets:

```python
THRESHOLD = 0.05
```

The README follows the **executed code**. At this threshold the notebook creates 500 graphs and reports:

| Graph statistic | Value |
|---|---:|
| Mean retained directed edges | **4,533** |
| Minimum | **342** |
| Maximum | **6,324** |
| Mean density | **0.340** |

The example first graph contains 116 nodes, 7 features per node, and 5,166 retained directed edges.

## 5. Graph Attention Network

The graph classifier is a compact two-layer GAT:

```text
116-node subject graph
        ↓
GATConv(7 → 8, 4 attention heads)
        ↓
ELU + dropout
        ↓
GATConv(32 → 8, 1 head)
        ↓
global mean pooling
        ↓
Linear(8 → 2)
```

The model contains only **698 trainable parameters**. Edge weights are supplied to the attention layers as one-dimensional edge attributes. Training uses Adam (`lr=1e-3`, `weight_decay=1e-3`) and cross-entropy loss for 100 epochs; the checkpoint with the lowest validation loss is retained.

<p align="center">
  <img src="complete_assets/abide_gat_training_curves.png" alt="GAT training and validation loss and accuracy across 100 epochs" width="88%" />
</p>

The best validation loss is **0.6940**. The trajectories remain close to chance-level behaviour rather than showing successful separation of the classes.

## 6. Held-out GAT evaluation

The GAT does **not** outperform the classical baseline in this run.

| Metric | GAT test result |
|---|---:|
| Accuracy | **0.5200** |
| Balanced accuracy | **0.5000** |
| ROC-AUC | **0.4696** |
| Reported binary F1 | **0.6842** |

The apparently moderate F1 is misleading if read as ASD performance: because `1 = Control`, it is the **control-class F1**. The classification report shows that the GAT predicts every held-out subject as Control:

- ASD recall: **0.00**
- Control recall: **1.00**
- ASD F1: **0.00**
- Control F1: **0.68**

<p align="center">
  <img src="complete_assets/abide_gat_test_confusion_matrix.png" alt="GAT confusion matrix on the held-out ABIDE test set" width="46%" />
</p>

This is a useful negative result. Representing a connectome as a graph and adding attention does not automatically create a stronger classifier. The model, graph construction, threshold, node features, training objective, cohort heterogeneity, and validation design all matter.

## 7. SVM vs GAT on the same test subjects

| Model | Accuracy | Balanced accuracy | ROC-AUC | Behaviour |
|---|---:|---:|---:|---|
| RBF-SVM | **0.6400** | **0.6266** | **0.7420** | Separates the classes to a useful degree on this split |
| GAT | **0.5200** | **0.5000** | **0.4696** | Collapses to the Control class |

The comparison is fair in one important sense: both models use exactly the same train, validation, and test subject indices. It is **not** a definitive architecture benchmark, because there is no repeated cross-validation, hyperparameter search, site-aware split, or uncertainty estimate around the test metrics.

## Methodological boundaries

These notebooks are learning and research-prototyping workflows, not clinical diagnostic systems. Several limitations should remain visible when interpreting the ABIDE experiment:

- ABIDE is **multi-site**, but the split is stratified only by diagnosis and sex; acquisition site is not explicitly controlled in the partitioning.
- The reported ABIDE results come from **one train/validation/test split**.
- The graph threshold (`0.05`) is an analysis choice rather than a biologically established cut-off.
- The GAT uses a small manually defined set of node descriptors and a very compact architecture.
- Group-average connectivity differences are descriptive and are not corrected inferential tests.
- Model attention weights, if inspected later, would be model-internal quantities rather than causal neurobiological explanations.

These constraints do not make the notebook uninformative. They make the main conclusion more precise: **connectome construction and evaluation design matter at least as much as model complexity**.

## Two workflows at a glance

```text
Notebook 1 — Developmental connectomics
fMRI volumes
   ↓
Schaefer-100 regional BOLD signals
   ↓
correlation / partial correlation
   ↓
graph analysis + tangent connectivity
   ↓
feature selection + linear SVM
   ↓
leakage-aware cross-validation + model-weighted edges

Notebook 2 — ABIDE ASD classification
preprocessed AAL regional time series
   ↓
116 × 116 partial-correlation matrices
   ├───────────────┐
   ↓               ↓
6,670 edges       thresholded subject graph
   ↓               ↓
RBF-SVM           7 node features + weighted edges
                   ↓
                  GAT
   └───────────────┘
          ↓
   common held-out test set
```

## Running the notebooks

Main dependencies across both workflows:

```bash
pip install nilearn numpy pandas matplotlib seaborn networkx scikit-learn torch torch_geometric
```

Run the notebooks independently:

```text
AIMS_09_fmri_connectomics.ipynb
fmri_gnns_abide.ipynb
```

The ABIDE notebook downloads regional derivatives on first execution. Its GAT stage is written to use CUDA when available, although the network itself is small.

## Repository structure

```text
.
├── AIMS_09_fmri_connectomics.ipynb
├── fmri_gnns_abide.ipynb
├── README.md
└── complete_assets/
    ├── connectomics_abide_banner.png
    ├── abide_cohort_distribution.png
    ├── abide_subject_partial_connectivity.png
    ├── abide_group_connectivity_comparison.png
    ├── abide_svm_test_confusion_matrix.png
    ├── abide_gat_training_curves.png
    └── abide_gat_test_confusion_matrix.png
```

---



All scientific figures shown in this README come directly from the notebook outputs rather than from generic stock or generated neuroscience illustrations.
