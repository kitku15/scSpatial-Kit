# Module 9: Condition-Specific Pseudobulk Differential Expression

!!! abstract "Overview"
    While Module 3 identifies marker genes that distinguish different cell types from one another, **Module 9** performs targeted differential expression analysis *within* individual cell types across experimental or clinical conditions (e.g., *Healthy Macrophages* vs. *Inflamed Macrophages*). 
    
    To avoid false-positive inflation common in single-cell hypothesis testing, this module operates on the sample-level **pseudobulk** count matrices generated in Module 7 and runs **PyDESeq2**, applying negative binomial generalized linear models with empirical Bayes dispersion estimation and log2 fold change shrinkage.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `celltype_col` | `String` | `"Final_Annotation"` | The cell type annotation column in `adata.obs`. Comparisons are evaluated independently within each population. |
| `treatment_col` | `String` | `"TreatmentResponse"` | Metadata column in `adata.obs` defining biological conditions or treatment arms. |
| `comparisons` | `List[List[String]]` | `[]` | Contrasts to compute, formatted as `["Test_Group", "Reference_Group"]`. Positive $\log_2\text{FC}$ values indicate upregulation in the `Test_Group`. |
| `pca_n_top_genes` | `Integer` | `2000` | Number of highly variable genes to compute for pseudobulk PCA prior to DE. |
| `pca_n_comps` | `Integer` | `50` | Number of principal components to calculate for pseudobulk PCA prior to DE. |
| `exclude_genes_file` | `String` or `None` | `""` | *(Optional)* Path to a plain-text file containing a list of genes to exclude before computing pseudobulk PCA (one gene per line). |

## Example Config

```toml
[modules.DEAnalysis]
celltype_col = "Final_Annotation"
treatment_col = "TreatmentResponse"
comparisons = [
    ["IFX_NR", "IFX_R"],
    ["CPIc", "Healthy"]
]
pca_n_top_genes = 2000 
pca_n_comps = 50 
exclude_genes_file = "exclude_genes.txt"
```

## Outputs

Outputs are structured per cell type within the `{analysis_name}_analysis/9_DEAnalysis/` directory:

```text
9_DEAnalysis/
└── run_{hash}/                                           # Hashed parameter directory (e.g., run_a1b2c3)
    ├── parameters.json                                   # JSON manifest of the settings used for this run
    └── {cell_type}/                                      # Per-cell-type results directory
        ├── {Test}_vs_{Ref}_all_genes.csv                 # Complete PyDESeq2 statistics table (baseMean, logfoldchanges, lfcSE, stat, pvals, pvals_adj)
        └── {Test}_vs_{Ref}_SIGNIFICANT_only.csv          # Filtered candidate table (pvals_adj < 0.05 and |logfoldchanges| > 0.5)
```

## Core Libraries & Functions Used
* **[PyDESeq2](https://pydeseq2.readthedocs.io/en/latest/)**:
    * `pydeseq2.dds.DeseqDataSet`: Builds the dataset container, computes size factors, and estimates gene-wise dispersions using negative binomial models.
    * `pydeseq2.ds.DeseqStats`: Computes Wald test statistics and evaluates Benjamini-Hochberg adjusted p-values ($p_{adj}$).