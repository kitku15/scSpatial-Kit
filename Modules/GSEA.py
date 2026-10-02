"""Module 9b: Gene Set Enrichment Analysis using GSEApy."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import gseapy as gp
import networkx as nx
from gseapy.scipalette import SciPalette

logger = logging.getLogger(__name__)


def create_dual_gsea_plot(df_subset: pd.DataFrame, out_path: Path, is_up: bool = True):
    if df_subset.empty:
        return

    # Take Top 10 by absolute NES
    df_plot = df_subset.copy().sort_values(by="NES", key=abs, ascending=False).head(10)

    # Reverse order so the highest NES is at the top of the plot
    df_plot = df_plot.iloc[::-1]

    # Clean pathway names (e.g., "HALLMARK_INFLAMMATORY_RESPONSE" -> "Inflammatory response")
    clean_terms = df_plot["Term"].str.replace("HALLMARK_", "", case=False)
    clean_terms = clean_terms.str.replace("KEGG_", "", case=False)
    clean_terms = clean_terms.str.replace("REACTOME_", "", case=False)
    clean_terms = clean_terms.str.replace("_", " ").str.capitalize()

    # Setup the figure canvas
    fig, ax1 = plt.subplots(figsize=(10, min(10, max(5, len(df_plot) * 0.8))))

    # Pick colors: Yellowish-green for UP, Steel Blue for DOWN
    color = "#ded328" if is_up else "#4682B4"

    # 1. Bar Plot for NES
    ax1.barh(clean_terms, df_plot["NES"], color=color, align="center")
    ax1.set_xlabel("Normalized enrichment score", fontsize=24)
    ax1.tick_params(axis="x", labelsize=16)
    ax1.tick_params(axis="y", labelsize=22)

    # Replicate the X-axis crop from the reference image
    if is_up:
        ax1.set_xlim(max(0, df_plot["NES"].min() - 0.2), df_plot["NES"].max() + 0.2)
    else:
        ax1.set_xlim(min(0, df_plot["NES"].max() + 0.2), df_plot["NES"].min() - 0.2)

    # 2. Twin Axis for Adjusted P-value
    ax2 = ax1.twiny()
    ax2.plot(
        df_plot["FDR q-val"],
        clean_terms,
        color="black",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=14,
        linewidth=1.5,
    )

    # Log scale & reverse it (so highly significant/small p-values go to the right)
    ax2.set_xscale("log")
    ax2.invert_xaxis()

    # Enforce P-value limits
    min_p = df_plot["FDR q-val"].min()
    ax2.set_xlim(1.0, min(1e-10, min_p * 0.01))

    ax2.set_xlabel("Adjusted P value", fontsize=24)
    ax2.tick_params(axis="x", labelsize=16)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_gsea_on_de_results(
    module_9_dir: Path,
    module_9b_dir: Path,
    databases: list[str],
    gene_col: str = "names",
    score_col: str = "logfoldchanges",
    nes_threshold: float = 2.0,
    padj_threshold: float = 0.05,
    min_size: int = 5,
    max_size: int = 1000,
    threads: int = 4,
) -> None:
    """
    Scans the DE Analysis (Module 9) output directory for results,
    ranks genes, and runs GSEApy per configured database, generating all plots.
    """
    de_files = list(module_9_dir.rglob("*_all_genes.csv"))

    if not de_files:
        logger.warning(
            f"No DE result files found in {module_9_dir}. Did Module 9 run successfully?"
        )
        return

    logger.info(f"Targeting {len(databases)} databases: {', '.join(databases)}")

    # Initialize SciPalette for combined plots
    sci = SciPalette()
    NbDr = sci.create_colormap()

    for de_file in de_files:
        relative_path = de_file.relative_to(module_9_dir)
        output_dir = module_9b_dir / relative_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        comparison_name = de_file.name.replace("_all_genes.csv", "")
        logger.info(
            f"Running GSEA for: {relative_path.parent.name} -> {comparison_name}"
        )

        try:
            # 1. Load and prep ranking data
            ranks_df = pd.read_csv(de_file)
            ranks_df = ranks_df.dropna(subset=[gene_col, score_col])

            rnk = ranks_df[[gene_col, score_col]].copy()
            rnk[gene_col] = rnk[gene_col].astype(str).str.upper()
            rnk = rnk.sort_values(by=score_col, ascending=False)

            # 2. Iterate through requested databases
            for db in databases:
                logger.info(f"  -> Querying database: {db}")
                db_outdir = output_dir / db
                db_outdir.mkdir(parents=True, exist_ok=True)

                # Run GSEA Prerank
                pre_res = gp.prerank(
                    rnk=rnk,
                    gene_sets=db,
                    threads=threads,
                    min_size=min_size,
                    max_size=max_size,
                    permutation_num=1000,
                    outdir=None,
                    seed=42,
                    verbose=False,
                )

                res_df = pre_res.res2d
                if res_df.empty:
                    continue

                res_df.to_csv(
                    db_outdir / f"{comparison_name}_{db}_ALL.csv", index=False
                )

                # 3. Filter for significance
                sig_df = res_df[
                    (res_df["FDR q-val"] < padj_threshold)
                    & (res_df["NES"].abs() >= nes_threshold)
                ].copy()

                if sig_df.empty:
                    logger.info(
                        f"     No significant pathways met thresholds for {db}."
                    )
                    continue

                sig_df.to_csv(
                    db_outdir / f"{comparison_name}_{db}_SIGNIFICANT.csv", index=False
                )

                # Prevent matplotlib colormap crash when all FDRs are exactly 0
                sig_df["FDR q-val"] = sig_df["FDR q-val"].replace(0, 1e-10)
                up_df = sig_df[sig_df["NES"] > 0].copy()
                down_df = sig_df[sig_df["NES"] < 0].copy()

                min_width = 7
                min_height = 5
                height_per_pathway = 0.5

                # A. Dotplots
                if not up_df.empty:
                    gp.dotplot(
                        up_df,
                        column="FDR q-val",
                        title=f"UP: {db}",
                        cmap=plt.cm.autumn_r,
                        size=6,
                        figsize=(
                            min_width,
                            max(min_height, len(up_df) * height_per_pathway),
                        ),
                        show_ring=True,
                        ofname=str(db_outdir / f"{comparison_name}_UP_dotplot.pdf"),
                    )

                if not down_df.empty:
                    gp.dotplot(
                        down_df,
                        column="FDR q-val",
                        title=f"DOWN: {db}",
                        cmap=plt.cm.winter_r,
                        size=6,
                        figsize=(
                            min_width,
                            max(min_height, len(down_df) * height_per_pathway),
                        ),
                        show_ring=True,
                        ofname=str(db_outdir / f"{comparison_name}_DOWN_dotplot.pdf"),
                    )

                # B. Combined Barplots & Dotplots
                if not up_df.empty and not down_df.empty:
                    enr_res = pd.concat([up_df, down_df])
                    enr_res = enr_res.rename(columns={"FDR q-val": "Adjusted P-value"})

                    gp.barplot(
                        enr_res,
                        group="UP_DW",
                        column="Adjusted P-value",
                        title=db,
                        color=["r", "b"],
                        figsize=(
                            min_width,
                            max(min_height, len(enr_res) * height_per_pathway),
                        ),
                        ofname=str(
                            db_outdir / f"{comparison_name}_combined_barplot.pdf"
                        ),
                    )

                    gp.dotplot(
                        enr_res,
                        x="UP_DW",
                        column="Adjusted P-value",
                        x_order=["UP", "DOWN"],
                        title=db,
                        cmap=NbDr.reversed(),
                        size=6,
                        figsize=(
                            min_width,
                            max(min_height, len(enr_res) * height_per_pathway),
                        ),
                        show_ring=True,
                        ofname=str(
                            db_outdir / f"{comparison_name}_combined_dotplot.pdf"
                        ),
                    )

                # C. Custom Dual Axis Plot (Your requested format!)
                if not up_df.empty:
                    create_dual_gsea_plot(
                        up_df,
                        out_path=db_outdir / f"{comparison_name}_UP_dualplot.pdf",
                        is_up=True,
                    )

                if not down_df.empty:
                    create_dual_gsea_plot(
                        down_df,
                        out_path=db_outdir / f"{comparison_name}_DOWN_dualplot.pdf",
                        is_up=False,
                    )

                # D. GSEA Enrichment Score Curves (Top 5)
                top_terms = (
                    sig_df.sort_values(by="NES", key=abs, ascending=False)["Term"]
                    .head(5)
                    .tolist()
                )
                if top_terms:
                    pre_res.plot(
                        terms=top_terms,
                        figsize=(7, 6),
                        ofname=str(db_outdir / f"{comparison_name}_GSEA_curves.pdf"),
                    )

                # E. Network Visualization
                try:
                    nodes, edges = gp.enrichment_map(sig_df)
                    if not nodes.empty and not edges.empty:
                        G = nx.from_pandas_edgelist(
                            edges,
                            source="src_idx",
                            target="targ_idx",
                            edge_attr=["jaccard_coef", "overlap_coef", "overlap_genes"],
                        )

                        for node in nodes.index:
                            if node not in G.nodes():
                                G.add_node(node)

                        fig, ax = plt.subplots(figsize=(10, 10))
                        pos = nx.layout.spring_layout(G, k=0.5)

                        nodes_nes = list(nodes.NES)
                        max_abs_nes = max([abs(x) for x in nodes_nes] + [2.0])

                        nx.draw_networkx_nodes(
                            G,
                            pos=pos,
                            cmap=plt.cm.RdBu_r,
                            node_color=nodes_nes,
                            vmin=-max_abs_nes,
                            vmax=max_abs_nes,
                            node_size=list(nodes.Hits_ratio * 2000),
                        )
                        nx.draw_networkx_labels(
                            G, pos=pos, labels=nodes.Term.to_dict(), font_size=10
                        )

                        edge_weight = nx.get_edge_attributes(G, "jaccard_coef").values()
                        nx.draw_networkx_edges(
                            G,
                            pos=pos,
                            width=list(map(lambda x: x * 10, edge_weight)),
                            edge_color="#CDDBD4",
                        )

                        plt.title(f"Enrichment Map: {db}", fontsize=14)
                        plt.tight_layout()
                        plt.savefig(
                            str(db_outdir / f"{comparison_name}_network_map.pdf"),
                            bbox_inches="tight",
                        )
                        plt.close("all")
                except Exception as e:
                    logger.warning(f"     Could not generate network map for {db}: {e}")

                logger.info(f"     Successfully generated all plots for {db}.")

        except Exception as e:
            logger.error(f"  -> Failed to run GSEA on {de_file.name}: {e}")

    logger.info("Module 9b: GSEA Processing Complete.")
