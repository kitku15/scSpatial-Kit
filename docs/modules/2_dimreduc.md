# Module 2: Dimension Reduction & Clustering

!!! abstract "Overview"
    This module performs dimensionality reduction and cell clustering. Users can choose to process their data using standard Principal Component Analysis (PCA), or use deep learning models (scVI and scVIVA) to correct for batch effects and account for the spatial tissue structure. 
    
    Following dimensionality reduction, the module groups cells using Leiden clustering across multiple user-defined resolutions and generates UMAP plots. Additionally, it provides an option to map the dataset onto a pre-annotated reference atlas using scANVI.


<mark>Add / check defaults for below </mark>

## Parameters

### General & Clustering Settings {#hide-me}

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `run_pca` | `Boolean` | `true` | If `true`, runs standard PCA before clustering. |
| `n_comps` | `Integer` or `None` | `50` | Number of Principal Components to compute. If set to `null` or omitted, the pipeline will utilize a mathematical elbow-detection heuristic to automatically calculate the optimal number of PCs based on variance ratio decay. |
| `n_neighbors` | `List[Int]` | `[15, 30]` | Defines KNN sizes for UMAP/Leiden (e.g., `[15, 30]`). The pipeline will iterate and generate outputs for *every* value in this list. |
| `resolution` | `List[Float]` | `[0.5, 1.0]` | Defines Leiden clustering resolutions (e.g., `[0.5, 1.0]`). Iterates over every combination of `n_neighbors` and `resolution`. |
| `cluster_name` | `String` | `"leiden"` | Prefix for the resulting cluster column in `adata.obs` (e.g., `"leiden"` -> `"leiden_n15_r0.5"`). |
| `umap_latent` | `String` | `"X_scVIVA"` | The latent space used to build the UMAP/graph (e.g., `"X_pca"`, `"X_scVI"`, `"X_scVIVA"`, or `"X_scANVI"`). |
| `dot_size` | `Float` | `5.0` | Size of the dots in the Scanpy UMAP and spatial scatter plots (default: `5.0`). |
| `exclude_genes_file` | `String` or `None` | `""` | *(Optional)* Path to a plain-text file containing a list of genes to exclude from highly variable gene (HVG) selection before computing PCA (one gene per line). |

### Deep Learning Model Settings (scVI / scVIVA) {#hide-me}

*Note: Hardware acceleration is handled automatically. The pipeline uses `torch.cuda.is_available()` to detect GPU presence. If a GPU is allocated on your HPC node, scVI and scVIVA will utilize it; otherwise, they will safely fall back to the CPU.*

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_scviva` | `Boolean` | `false` | If `true`, runs the scVIVA spatially-aware VAE pipeline. If `false`, bypasses deep learning models entirely. |
| `scviva_layer` | `String` | `"counts"` | The AnnData layer containing *raw integer counts* required by scVI (e.g., `"counts"`). |
| `scviva_batch_key` | `String` | `"slide_id"` | Column name in `adata.obs` defining batches/slides for batch effect correction. |
| `scvi_epochs` | `Integer` | `400` | Training epochs for the baseline spatially-unaware scVI model. |
| `scviva_epochs` | `Integer` | `400` | Training epochs for the spatial scVIVA model. |
| `scviva_spatial_knn` | `Integer` | `10` | Number of spatial neighbors used by scVIVA to model the tissue microenvironment graph (default: `10`). Note: Only builds neighbors within the same sample. |
| `scviva_batch_size` | `Integer` | `512` | Batch size for training scVIVA (default: `512`). Lower if running out of GPU memory. |
| `n_latent` | `Integer` | `30` | Number of latent dimensions for scVI/scVIVA to learn (default: `30`). |
| `n_layers` | `Integer` | `2` | Number of hidden layers in the scVI/scVIVA neural network (default: `2`). |
| `pre_cluster_res` | `Float` | `1.0` | Leiden resolution used to generate the pre-clustering labels on scVI latent space that scVIVA requires (default: `1.0`). |

### Reference Mapping Settings (scANVI) {#hide-me}

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_scanvi` | `Boolean` | `false` | If `true`, enables scANVI reference mapping. |
| `scanvi_mode` | `String` | `"scarches"` | `"scarches"` (fast adapter training) or `"joint"` (*de novo* joint training with reference). |
| `reference_adata_path` | `String` | `""` | Path to the reference dataset `.h5ad` or local scVI model directory. |
| `reference_label_key` | `String` | `"cell_type"` | The column in the reference containing the ground truth cell types. |
| `reference_batch_key` | `String` | `"batch"` | The batch column expected by the reference model (maps your data into the model's batch covariate). |
| `scanvi_epochs` | `Integer` | `200` | Training epochs for the scANVI model (default: `200`). |
| `reference_covariates`| `Dictionary` | `{}` | *(Optional)* Maps extra categorical covariates required by external pre-trained reference models. The pipeline inspects `_scvi_required_metadata.json` automatically. If the model expects a column (e.g., `"tissue_in_vivo"`) that your data lacks, you can inject it here: `{"tissue_in_vivo": "liver", "assay": "CosMx"}`. If omitted, missing requirements fallback to `"unspecified"`. |

## Example Config

```toml
[modules.DimensionReduction]
# General & Clustering
run_pca = true
n_comps = 50
n_neighbors = [5, 10, 30, 50]
resolution = [0.5, 1.0, 1.5, 2.0]
cluster_name = "leiden"
umap_latent = "X_scVIVA"
dot_size = 5.0
exclude_genes_file = "exclude_genes.txt"

# Deep Learning (scVI / scVIVA)
use_scviva = true
scviva_layer = "counts"
scviva_batch_key = "slide_id"
scvi_epochs = 400
scviva_epochs = 400
scviva_spatial_knn = 10
scviva_batch_size = 512
n_latent = 30
n_layers = 2
pre_cluster_res = 1.0

# Reference Mapping (scANVI)
use_scanvi = false
scanvi_mode = "scarches"
reference_adata_path = "path/to/reference/model"
reference_label_key = "cell_type"
reference_batch_key = "batch"
scanvi_epochs = 200
reference_covariates = {}
```

## Outputs

Outputs are saved in the `{analysis_name}_analysis/2_DimensionReduction/` directory:

```text
2_DimensionReduction/
├── adata.h5ad                                                    # Updated AnnData object with new embeddings and clusters
├── n{n_neighbours}_r{resolution}/
│   ├── {cluster_name}_n{n_neighbours}_r{resolution}_spatial.png  # Spatial tissue scatter plot colored by cluster
│   ├── umap_n{n_neighbours}_{umap_latent}_{cluster_name}_n{n_neighbours}_r{resolution}.png # UMAP scatter plot colored by cluster
│   └── ...
├── pca_variance_ratioPCA.png  # PCA elbow plot
```

### AnnData Updates {#hide-me}
Running this module adds the following data to `adata.h5ad`:

* **`adata.obsm`**:
    * `'X_pca'`: Principal component matrix (if `run_pca = true`).
    * `'X_scVI'`: Latent dimensions from the spatially-unaware scVI model (if `use_scviva = true`).
    * `'X_scVIVA'`: Latent dimensions from the spatially-aware scVIVA model (if `use_scviva = true`).
    * `'X_scANVI'`: Latent dimensions from label transfer (if `use_scanvi = true`).
    * `'X_umap'`: 2D UMAP coordinates generated from `umap_latent`.
* **`adata.obs`**:
    * `'{cluster_name}_n{neighbors}_r{resolution}'`: Cluster assignments for every tested parameter combination (e.g., `leiden_n10_r1.0`).
    * `'scviva_precluster'`: Preliminary clustering labels generated for scVIVA training (if `use_scviva = true`).
* **`adata.uns`**:
    * `'neighbors'`: Connectivities and distance matrices calculated from the chosen latent space.

## Core Libraries & Functions Used
* **[scvi-tools](https://docs.scvi-tools.org/en/stable/)**:
    * [`scvi.model.SCVI()`](https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html): Spatially-unaware baseline model for transcriptomic representation and batch-correction.
    * [`scvi.model.SCANVI.load_query_data()`](https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCANVI.html): Performs scArches reference mapping.
* **[scVIVA](https://github.com/chenhcs/scVIVA)**:
    * `SCVIVA()`: Spatially-aware Variational Autoencoder that incorporates local tissue niches into the latent embedding.
* **[Scanpy](https://scanpy.readthedocs.io/en/stable/)**:
    * [`sc.pp.highly_variable_genes()`](https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html): Isolates the top 2,000 highly variable genes using the `seurat_v3` flavor on raw counts prior to dimension reduction.
    * [`sc.pp.scale()`](https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.scale.html): Scales expression data to a maximum value of 10.
    * [`sc.pp.pca()`](https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.pca.html): Computes standard principal component analysis on the scaled HVGs.
    * [`sc.pp.neighbors()`](https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.neighbors.html): Computes the neighborhood graph based on the deep learning or PCA embeddings.
    * [`sc.tl.umap()`](https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.umap.html): Non-linear dimension reduction for visualization.
    * [`sc.tl.leiden()`](https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.leiden.html): Graph-based clustering (using the `igraph` flavor).