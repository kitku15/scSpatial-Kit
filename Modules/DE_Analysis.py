"""DE Analysis for specific conditions module using PyDESeq2."""

import re
import warnings
from logging import getLogger
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt

# Import PyDESeq2
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

warnings.filterwarnings("ignore")
logger = getLogger(__name__)


def _run_pydeseq2_comparison(
    adata_sub: AnnData,
    treatment_col: str,
    test_group: str,
    ref_group: str,
    output_dir: Path,
) -> None:
    """Runs PyDESeq2 for a specific pairwise comparison and exports results."""
    comparison_name = f"{test_group}_vs_{ref_group}"

    sc.pp.filter_genes(adata_sub, min_counts=1)
    if adata_sub.n_obs < 2:
        logger.info("      Not enough samples for this comparison.")
        return

    try:
        # Run PyDESeq2
        dds = DeseqDataSet(
            adata=adata_sub, design_factors=treatment_col, refit_cooks=True, n_cpus=4
        )
        dds.deseq2()

        stat_res = DeseqStats(
            dds, contrast=[treatment_col, test_group, ref_group], n_cpus=4
        )
        stat_res.summary()

        # Format Results
        res_df = stat_res.results_df
        markers = res_df.reset_index().rename(
            columns={
                "index": "names",
                "log2FoldChange": "logfoldchanges",
                "pvalue": "pvals",
                "padj": "pvals_adj",
            }
        )

        # Handle NaNs from PyDESeq2
        markers["pvals_adj"] = markers["pvals_adj"].fillna(1.0)
        markers["pvals"] = markers["pvals"].fillna(1.0)
        markers["logfoldchanges"] = markers["logfoldchanges"].fillna(0.0)
        markers = markers.sort_values(by="logfoldchanges", ascending=False)

        # Save outputs
        markers.to_csv(output_dir / f"{comparison_name}_all_genes.csv", index=False)

        sig_markers = markers[
            (markers["pvals_adj"] < 0.05) & (markers["logfoldchanges"].abs() > 0.5)
        ]
        sig_markers.to_csv(
            output_dir / f"{comparison_name}_SIGNIFICANT_only.csv", index=False
        )

        # Generate MA Plot
        try:
            plt.figure(figsize=(8, 6))
            # Filter out 0 or NaN baseMeans to safely use log-scale on X-axis
            plot_df = markers[markers["baseMean"] > 0].copy()

            sig_up = (plot_df["pvals_adj"] < 0.05) & (plot_df["logfoldchanges"] > 0.5)
            sig_down = (plot_df["pvals_adj"] < 0.05) & (
                plot_df["logfoldchanges"] < -0.5
            )
            ns = ~(sig_up | sig_down)

            # Plot NS first so it sits in the background
            plt.scatter(
                plot_df.loc[ns, "baseMean"],
                plot_df.loc[ns, "logfoldchanges"],
                color="lightgray",
                s=10,
                alpha=0.5,
                label="Not Sig",
            )
            # Plot Down
            plt.scatter(
                plot_df.loc[sig_down, "baseMean"],
                plot_df.loc[sig_down, "logfoldchanges"],
                color="steelblue",
                s=15,
                alpha=0.8,
                label="Down",
            )
            # Plot Up
            plt.scatter(
                plot_df.loc[sig_up, "baseMean"],
                plot_df.loc[sig_up, "logfoldchanges"],
                color="firebrick",
                s=15,
                alpha=0.8,
                label="Up",
            )

            plt.xscale("log")
            plt.axhline(0, color="black", linewidth=1, linestyle="--")

            plt.xlabel("Mean Expression (baseMean)", fontsize=12)
            plt.ylabel("Log2 Fold Change", fontsize=12)
            plt.title(
                f"MA Plot: {comparison_name.replace('_vs_', ' vs ')}", fontsize=14
            )
            plt.legend(loc="upper right")
            plt.tight_layout()

            plt.savefig(
                output_dir / f"{comparison_name}_MA_plot.pdf", bbox_inches="tight"
            )
            plt.close("all")
        except Exception as plot_e:
            logger.error(f"      Failed to generate MA plot: {plot_e}")

        logger.info(f"      Found {len(sig_markers)} significant DE genes.")

    except Exception as e:
        logger.error(f"      PyDESeq2 Error: {e}")


def targeted_pairwise_DE(
    pseudobulk_adata_path: str | Path,
    celltype_col: str,
    treatment_col: str,
    comparisons: List[Tuple[str, str]],
    module_dir: str | Path,
    batch_key: str = None,
    sample_key: str = None,
    n_top_genes: int = 2000,
    n_comps: int = 50,
    exclude_genes_file: str | Path = None,
) -> None:
    """Performs PCA on pseudobulk, then specific targeted pairwise DE comparisons within each cell type using PyDESeq2."""
    adata = sc.read_h5ad(pseudobulk_adata_path)
    module_dir = Path(module_dir)

    # 1. Pseudobulk PCA
    logger.info("Running PCA on pseudobulk data before DE analysis...")
    try:
        adata_pca = adata.copy()

        if exclude_genes_file and Path(exclude_genes_file).is_file():
            with open(exclude_genes_file, "r") as f:
                excluded_genes = {line.strip() for line in f if line.strip()}
            keep_genes = [g for g in adata_pca.var_names if g not in excluded_genes]
            adata_pca = adata_pca[:, keep_genes].copy()
            logger.info(f"Excluded {len(excluded_genes)} genes before pseudobulk PCA.")

        # Ensure non-negative integers for seurat_v3 HVG calculations
        adata_pca.X = np.round(adata_pca.X).astype(int)

        # Filter out genes with zero counts across all pseudobulk samples
        sc.pp.filter_genes(adata_pca, min_counts=1)

        actual_n_comps = min(n_comps, adata_pca.n_obs - 1)
        actual_n_top_genes = min(n_top_genes, adata_pca.n_vars)

        if actual_n_comps >= 2 and actual_n_top_genes > 2:
            sc.pp.highly_variable_genes(
                adata_pca, flavor="seurat_v3", n_top_genes=actual_n_top_genes
            )

            # Standard practice: normalize and log before scaling
            sc.pp.normalize_total(adata_pca, target_sum=1e4)
            sc.pp.log1p(adata_pca)
            sc.pp.scale(adata_pca, max_value=10)

            sc.tl.pca(adata_pca, n_comps=actual_n_comps)

            # Set up plotting directory
            sc.settings.figdir = str(module_dir)

            # A) Variance Ratio (Elbow Plot)
            sc.pl.pca_variance_ratio(
                adata_pca,
                log=True,
                n_pcs=actual_n_comps,
                show=False,
                save="_variance_ratio.pdf",
            )

            # B) PCA scatter plot
            color_cols = [
                c for c in [treatment_col, batch_key, sample_key, celltype_col] if c
            ]
            color_cols = list(
                dict.fromkeys(color_cols)
            )  # Maintain order, remove duplicates safely
            valid_colors = [c for c in color_cols if c in adata_pca.obs.columns]

            if valid_colors:
                sc.pl.pca(
                    adata_pca,
                    color=valid_colors,
                    show=False,
                    save="_pseudobulk_samples.pdf",
                )
            else:
                sc.pl.pca(adata_pca, show=False, save="_pseudobulk_samples.pdf")

            plt.close("all")  # Clear figures to prevent memory leaks

            # Transfer PCA representation to original object for web visualization
            # Note: Observation rows are un-altered, making coordinate transfer 1-to-1 safe.
            if adata_pca.n_obs == adata.n_obs:
                adata.obsm["X_pca"] = adata_pca.obsm["X_pca"]

                # Map HVG stats back to main pseudobulk object for transparency
                for col in ["highly_variable", "means", "variances", "variances_norm"]:
                    if col in adata_pca.var.columns:
                        adata.var[col] = adata.var_names.map(adata_pca.var[col])
                if "highly_variable" in adata.var.columns:
                    adata.var["highly_variable"] = (
                        adata.var["highly_variable"].fillna(False).astype(bool)
                    )

                adata.write_h5ad(pseudobulk_adata_path)
                logger.info(
                    f"PCA completed successfully. 'X_pca' and HVG stats added to {Path(pseudobulk_adata_path).name}"
                )
            else:
                logger.warning(
                    "Observation dimension mismatch. Skipping X_pca transfer."
                )
        else:
            logger.warning(
                f"Not enough valid observations ({adata_pca.n_obs}) or genes ({adata_pca.n_vars}) "
                f"for PCA. Minimum 3 obs and 3 genes required."
            )
    except Exception as e:
        logger.error(f"Pseudobulk PCA encountered an error: {e}")

    # 2. PyDESeq2 Cell Type Targeted Comparisons
    # Ensure raw integer counts for PyDESeq2
    adata.X = np.round(adata.X).astype(int)
    cell_types = adata.obs[celltype_col].dropna().unique()

    for ct in cell_types:
        logger.info(f"\n{'=' * 60}\nRunning PyDESeq2 for cell type: {ct}\n{'=' * 60}")

        adata_ct = adata[adata.obs[celltype_col] == ct].copy()
        safe_ct = re.sub(r"[^\w\s-]", "", str(ct)).replace(" ", "_")
        output_dir = module_dir / safe_ct
        output_dir.mkdir(parents=True, exist_ok=True)

        for test_group, ref_group in comparisons:
            groups_present = adata_ct.obs[treatment_col].unique()
            if test_group not in groups_present or ref_group not in groups_present:
                logger.info(f"  [Skip] {test_group} or {ref_group} missing in {ct}")
                continue

            logger.info(
                f"\n  --> Comparing: {test_group} (Test) vs {ref_group} (Reference)"
            )

            adata_sub = adata_ct[
                adata_ct.obs[treatment_col].isin([test_group, ref_group])
            ].copy()
            _run_pydeseq2_comparison(
                adata_sub, treatment_col, test_group, ref_group, output_dir
            )

    logger.info("\nAll cell types and targeted comparisons processed successfully!")
