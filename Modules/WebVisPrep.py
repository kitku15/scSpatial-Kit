"""Module 10: Prepares Zarr and Auxiliary Data for the FastAPI Backend."""

import gc
import json
import re
import shutil
import tarfile
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

warnings.filterwarnings("ignore")
logger = getLogger(__name__)


def export_cpdb_for_web_vis(
    cpdb_out_dir: Path,
    adata_path: Path,
    celltype_key: str,
    microenv_key: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    matched_files = list(cpdb_out_dir.glob("*significant_means*.txt"))

    if not matched_files:
        logger.info(
            f"Error: No files matching '*significant_means*.txt' found in {cpdb_out_dir}"
        )
        return

    target_file = matched_files[0]
    sig_means = pd.read_csv(target_file, sep="\t")
    logger.info(f"Successfully loaded edges from: {target_file.name}")

    pair_cols = [c for c in sig_means.columns if "|" in c]
    edges = []

    for _, row in sig_means.iterrows():
        interaction = row["interacting_pair"]
        for pair in pair_cols:
            val = row[pair]
            if pd.notna(val) and val > 0:
                source, target = pair.split("|")
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "interaction": interaction,
                        "value": float(val),
                    }
                )

    with open(out_dir / "cpdb_edges.json", "w") as f:
        json.dump(edges, f)

    logger.info(
        f"Loading AnnData (backed) to calculate cell counts: {adata_path.name}..."
    )
    # SWE Upgrade: Use backed mode to save RAM
    adata = sc.read_h5ad(adata_path, backed="r")
    counts = (
        adata.obs.groupby([microenv_key, celltype_key]).size().reset_index(name="count")
    )

    micro_map = {}
    for _, row in counts.iterrows():
        env, cell, cnt = (
            str(row[microenv_key]),
            str(row[celltype_key]),
            int(row["count"]),
        )
        if cnt > 0:
            micro_map.setdefault(env, {})[cell] = cnt

    with open(out_dir / "cpdb_microenvs.json", "w") as f:
        json.dump(micro_map, f)


def export_spatial_stats_for_web(mod5_dir: Path, mod6_dir: Path, out_dir: Path) -> None:
    target_stats_dir = out_dir / "spatial_stats"
    target_stats_dir.mkdir(parents=True, exist_ok=True)

    if mod5_dir.exists():
        for sample_dir in [
            d
            for d in mod5_dir.iterdir()
            if d.is_dir() and d.name != "Aggregated_Results"
        ]:
            sample_out = target_stats_dir / sample_dir.name
            sample_out.mkdir(parents=True, exist_ok=True)

            files_to_copy = [
                f"centrality_scores_{sample_dir.name}.json",
                f"co_occurrence_{sample_dir.name}.json",
                f"nhood_enrichment_{sample_dir.name}.json",
                f"moranI_results_{sample_dir.name}.csv",
            ]
            for f in files_to_copy:
                src = sample_dir / f
                if src.exists():
                    shutil.copy(src, sample_out / f)

    if mod6_dir.exists():
        for sample_dir in [
            d
            for d in mod6_dir.iterdir()
            if d.is_dir() and d.name != "Aggregated_Results"
        ]:
            sample_out = target_stats_dir / sample_dir.name
            sample_out.mkdir(parents=True, exist_ok=True)

            pcf_files = list(sample_dir.rglob("cross_pcf_all.json"))
            morph_files = list(sample_dir.rglob("morphometrics.csv"))

            if pcf_files:
                shutil.copy(pcf_files[0], sample_out / "cross_pcf_all.json")
            if morph_files:
                shutil.copy(morph_files[0], sample_out / "morphometrics.csv")


def export_qc_for_web(mod1_dir: Path, out_dir: Path) -> None:
    target_qc_dir = out_dir / "qc"
    target_qc_dir.mkdir(parents=True, exist_ok=True)

    qc_csvs = list(mod1_dir.rglob("qc_metrics.csv"))
    qc_jsons = list(mod1_dir.rglob("qc_thresholds.json"))

    histograms_by_slide, thresholds_by_slide = {}, {}
    all_dfs = []

    for json_path in qc_jsons:
        slide_name = json_path.parent.name
        if slide_name == mod1_dir.name:
            slide_name = "Slide_1"
        with open(json_path, "r") as f:
            thresholds_by_slide[slide_name] = json.load(f)

    for csv_path in qc_csvs:
        slide_name = csv_path.parent.name
        if slide_name == mod1_dir.name:
            slide_name = "Slide_1"

        df = pd.read_csv(csv_path)
        all_dfs.append(df)
        histograms_by_slide[slide_name] = {}
        for col in ["total_counts", "n_genes_by_counts", "area", "nucleus_signal"]:
            if col in df.columns:
                data = df[col].dropna()
                if len(data) > 0:
                    counts, edges = np.histogram(data, bins=100)
                    histograms_by_slide[slide_name][col] = {
                        "counts": counts.tolist(),
                        "edges": np.round(edges, 2).tolist(),
                    }

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        histograms_by_slide["All"] = {}
        for col in ["total_counts", "n_genes_by_counts", "area", "nucleus_signal"]:
            if col in combined_df.columns:
                data = combined_df[col].dropna()
                if len(data) > 0:
                    counts, edges = np.histogram(data, bins=100)
                    histograms_by_slide["All"][col] = {
                        "counts": counts.tolist(),
                        "edges": np.round(edges, 2).tolist(),
                    }
        if thresholds_by_slide:
            thresholds_by_slide["All"] = list(thresholds_by_slide.values())[0]

    with open(target_qc_dir / "qc_histograms.json", "w") as f:
        json.dump(histograms_by_slide, f)
    with open(target_qc_dir / "qc_thresholds.json", "w") as f:
        json.dump(thresholds_by_slide, f)


def _extract_proseg_polygons(
    proseg_zarr_dir: str, adata_obs: pd.DataFrame, slide_id: str
) -> Dict[str, list]:
    import spatialdata

    slide_polys = {}
    logger.info(f"Slide {slide_id}: Extracting polygons from Proseg Zarr...")
    try:
        proseg_sdata = spatialdata.read_zarr(proseg_zarr_dir)
        proseg_gdf = proseg_sdata.shapes["cell_boundaries"]
        if hasattr(proseg_gdf, "compute"):
            proseg_gdf = proseg_gdf.compute()

        proseg_obs = proseg_sdata.tables["table"].obs
        proseg_id_map = {}

        if "fov" in proseg_obs.columns and "cell_ID" in proseg_obs.columns:
            for idx, row in proseg_obs.iterrows():
                proseg_id_map[
                    (str(row["fov"]), str(row["cell_ID"]).replace(".0", ""))
                ] = idx
        elif "original_cell_id" in proseg_obs.columns:
            for idx, row in proseg_obs.iterrows():
                parts = str(row["original_cell_id"]).split("_")
                if len(parts) >= 2:
                    proseg_id_map[(parts[-2], parts[-1].replace(".0", ""))] = idx

        for obs_name, row in adata_obs.iterrows():
            if "fov" in row and "cell_ID" in row:
                f, c = str(row["fov"]), str(row["cell_ID"]).replace(".0", "")
                proseg_idx = proseg_id_map.get((f, c))

                if proseg_idx is not None:
                    try:
                        geom = proseg_gdf.loc[int(proseg_idx)].geometry
                    except KeyError:
                        geom = proseg_gdf.loc[str(proseg_idx)].geometry

                    if geom.geom_type == "MultiPolygon":
                        geom = max(geom.geoms, key=lambda a: a.area)
                    if geom.geom_type != "Polygon":
                        continue

                    coords = np.array(geom.exterior.coords)[:, :2].tolist()
                    slide_polys[obs_name] = coords
    except Exception as e:
        logger.error(f"Failed to parse Proseg Zarr: {e}")
    return slide_polys


def _extract_cosmx_polygons(
    raw_dataset_dir: str,
    adata_obs: pd.DataFrame,
    spatial_coords: np.ndarray,
    slide_id: str,
) -> Dict[str, list]:
    slide_polys = {}
    poly_files = list(Path(raw_dataset_dir).glob("*polygons*.csv"))
    if poly_files:
        logger.info(
            f"Slide {slide_id}: Extracting, scaling (0.12 µm/px), and aligning CosMx polygons..."
        )
        df_poly = pd.read_csv(poly_files[0])
        if "cellID" in df_poly.columns:
            df_poly.rename(columns={"cellID": "cell_ID"}, inplace=True)

        df_poly["fov"] = df_poly["fov"].astype(str)
        df_poly["cell_ID"] = (
            df_poly["cell_ID"].astype(str).str.replace(".0", "", regex=False)
        )

        df_adata = pd.DataFrame(
            {
                "obs_name": adata_obs.index,
                "fov": adata_obs["fov"].astype(str),
                "cell_ID": adata_obs["cell_ID"]
                .astype(str)
                .str.replace(".0", "", regex=False),
                "pt_x": spatial_coords[:, 0],
                "pt_y": spatial_coords[:, 1],
            }
        )

        df_centroids = (
            df_poly.groupby(["fov", "cell_ID"])
            .agg(poly_x=("x_global_px", "mean"), poly_y=("y_global_px", "mean"))
            .reset_index()
        )

        df_mapping = pd.merge(
            df_centroids, df_adata, on=["fov", "cell_ID"], how="inner"
        )
        df_poly_mapped = pd.merge(
            df_poly,
            df_mapping[
                ["fov", "cell_ID", "obs_name", "pt_x", "pt_y", "poly_x", "poly_y"]
            ],
            on=["fov", "cell_ID"],
            how="inner",
        )

        PIXEL_SIZE = 1
        df_poly_mapped["final_x"] = df_poly_mapped["pt_x"] + (
            (df_poly_mapped["x_global_px"] - df_poly_mapped["poly_x"]) * PIXEL_SIZE
        )
        df_poly_mapped["final_y"] = df_poly_mapped["pt_y"] - (
            (df_poly_mapped["y_global_px"] - df_poly_mapped["poly_y"]) * PIXEL_SIZE
        )

        for obs_name, group in df_poly_mapped.groupby("obs_name"):
            slide_polys[obs_name] = group[["final_x", "final_y"]].values.tolist()

    return slide_polys


def _extract_xenium_polygons(
    raw_dataset_dir: str, adata_obs: pd.DataFrame, slide_id: str
) -> Dict[str, list]:
    slide_polys = {}
    bound_csv = Path(raw_dataset_dir) / "cell_boundaries.csv.gz"
    bound_pq = Path(raw_dataset_dir) / "cell_boundaries.parquet"

    df_poly = None
    if bound_pq.exists():
        df_poly = pd.read_parquet(bound_pq)
    elif bound_csv.exists():
        df_poly = pd.read_csv(bound_csv)

    if df_poly is not None:
        lookup = {
            str(row.get("cell_id", obs_name)): obs_name
            for obs_name, row in adata_obs.iterrows()
        }
        for cell_id, group in df_poly.groupby("cell_id"):
            obs_name = lookup.get(str(cell_id))
            if obs_name:
                slide_polys[obs_name] = group[["vertex_x", "vertex_y"]].values.tolist()

    return slide_polys


def export_segmentations_for_web(
    adata_path: Path, data_type: str, settings: dict, out_dir: Path
) -> None:
    logger.info("\n--- Extracting Cell Segmentations ---")

    # SWE Upgrade: Backed mode for memory optimization! We only need `.obs` and `.obsm`.
    adata = sc.read_h5ad(adata_path, backed="r")
    spatial_key = settings.get("project", {}).get(
        "spatial_key", "global" if data_type == "CosMx" else "spatial"
    )

    seg_all, seg_by_slide, seg_by_sample, seg_by_microenv = {}, {}, {}, {}

    obs_meta = {
        obs_name: {
            "slide": str(row.get("slide_id", "All")),
            "sample": str(row.get("sample_id", "All")),
            "microenv": str(row.get("spatial_microenvironment", "All"))
            if "spatial_microenvironment" in row
            else None,
        }
        for obs_name, row in adata.obs.iterrows()
    }

    slide_ids = (
        adata.obs["slide_id"].dropna().unique()
        if "slide_id" in adata.obs.columns
        else [None]
    )

    for slide_id in slide_ids:
        slide_settings = (
            settings.get("io", {}).get("raw_data", {}).get(slide_id, {})
            if slide_id
            else settings.get("io", {})
        )
        raw_dataset_dir = slide_settings.get("dataset_dir")
        proseg_zarr_dir = slide_settings.get("proseg_zarr_dir")

        adata_obs_sub = (
            adata.obs[adata.obs["slide_id"] == slide_id] if slide_id else adata.obs
        )
        global_indices = adata.obs_names.get_indexer(adata_obs_sub.index)
        spatial_coords = adata.obsm[spatial_key][global_indices]

        slide_polys = {}

        if proseg_zarr_dir and Path(proseg_zarr_dir).exists():
            slide_polys = _extract_proseg_polygons(
                proseg_zarr_dir, adata_obs_sub, slide_id
            )
        elif (
            data_type == "CosMx" and raw_dataset_dir and Path(raw_dataset_dir).exists()
        ):
            slide_polys = _extract_cosmx_polygons(
                raw_dataset_dir, adata_obs_sub, spatial_coords, slide_id
            )
        elif (
            data_type == "Xenium" and raw_dataset_dir and Path(raw_dataset_dir).exists()
        ):
            slide_polys = _extract_xenium_polygons(
                raw_dataset_dir, adata_obs_sub, slide_id
            )

        for obs_name, coords in slide_polys.items():
            seg_all[obs_name] = coords
            m = obs_meta.get(obs_name)
            if m:
                if m["slide"] != "All":
                    seg_by_slide.setdefault(m["slide"], {})[obs_name] = coords
                if m["sample"] != "All":
                    seg_by_sample.setdefault(m["sample"], {})[obs_name] = coords
                if m["microenv"]:
                    seg_by_microenv.setdefault(m["microenv"], {})[obs_name] = coords

    seg_dir = out_dir / "segmentations"
    seg_dir.mkdir(parents=True, exist_ok=True)

    def save_seg(data_dict, filename):
        rounded_data = {k: np.round(v, 1).tolist() for k, v in data_dict.items()}
        with open(seg_dir / filename, "w") as f:
            json.dump(rounded_data, f, separators=(",", ":"))

    save_seg(seg_all, "segmentations.json")
    for k, v in seg_by_slide.items():
        save_seg(v, f"segmentations_{k}.json")
    for k, v in seg_by_sample.items():
        save_seg(v, f"segmentations_{k}.json")
    for k, v in seg_by_microenv.items():
        save_seg(v, f"segmentations_microenv_{k}.json")

    logger.info(
        f"Successfully exported {len(seg_all)} cell boundaries across all subsets."
    )


def export_de_analysis_for_web(mod3_dir: Path, out_dir: Path) -> None:
    target_de_dir = out_dir / "de_analysis"
    target_de_dir.mkdir(parents=True, exist_ok=True)
    de_metadata = {}

    top_de_files = list(mod3_dir.rglob("top_DEgenes_*.csv"))

    for top_file in top_de_files:
        annotation_col = top_file.name.replace("top_DEgenes_", "").replace(".csv", "")
        shutil.copy(top_file, target_de_dir / top_file.name)

        cluster_files = list(top_file.parent.glob("DEgenes/cluster_*_data.csv"))
        clusters = []
        for c_file in cluster_files:
            cluster_name = c_file.name.replace("cluster_", "").replace("_data.csv", "")
            clusters.append(cluster_name)

            df = pd.read_csv(c_file)
            clean_data = {
                "names": df["names"].tolist(),
                "logfc": np.round(df["logfoldchanges"].fillna(0), 3).tolist(),
                "pvals": df["pvals_adj"].fillna(1.0).tolist(),
            }
            with open(
                target_de_dir / f"{annotation_col}_cluster_{cluster_name}.json", "w"
            ) as f:
                json.dump(clean_data, f)

        de_metadata[annotation_col] = sorted(clusters)

    with open(target_de_dir / "de_metadata.json", "w") as f:
        json.dump(de_metadata, f)


def export_conditions_de_for_web(
    targeted_de_dir: Path, out_dir: Path, celltype_col: str, treatment_col: str
) -> None:
    if not targeted_de_dir.exists():
        return

    target_de_dir = out_dir / "conditions_de_analysis"
    target_de_dir.mkdir(parents=True, exist_ok=True)

    de_metadata = {
        "config": {"celltype_col": celltype_col, "treatment_col": treatment_col},
        "comparisons": {},
    }

    for ct_path in [d for d in targeted_de_dir.iterdir() if d.is_dir()]:
        celltype_folder = ct_path.name
        comparisons, summary_rows = [], []

        for c_file in ct_path.glob("*_all_genes.csv"):
            comparison_name = c_file.name.replace("_all_genes.csv", "")
            comparisons.append(comparison_name)

            df = pd.read_csv(c_file)
            df["pvals_adj"] = df["pvals_adj"].fillna(1.0).clip(lower=1e-300)

            plot_df = df[(df["pvals_adj"] < 0.1) | (abs(df["logfoldchanges"]) > 0.5)]
            clean_data = {
                "names": plot_df["names"].tolist(),
                "logfc": np.round(plot_df["logfoldchanges"].fillna(0), 3).tolist(),
                "pvals": plot_df["pvals_adj"].tolist(),
                "baseMean": np.round(plot_df["baseMean"].fillna(0), 3).tolist()
                if "baseMean" in plot_df.columns
                else [],
            }

            with open(
                target_de_dir / f"{celltype_folder}_comparison_{comparison_name}.json",
                "w",
            ) as f:
                json.dump(clean_data, f)

            sig_df = df[df["pvals_adj"] < 0.05]
            top_up = (
                sig_df[sig_df["logfoldchanges"] > 0]
                .sort_values(by="logfoldchanges", ascending=False)["names"]
                .head(5)
                .tolist()
            )
            top_down = (
                sig_df[sig_df["logfoldchanges"] < 0]
                .sort_values(by="logfoldchanges", ascending=True)["names"]
                .head(5)
                .tolist()
            )
            summary_rows.append(
                {
                    "Comparison": comparison_name,
                    "Top Upregulated": top_up,
                    "Top Downregulated": top_down,
                }
            )

        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(
                target_de_dir / f"summary_{celltype_folder}.csv", index=False
            )
        de_metadata["comparisons"][celltype_folder] = sorted(comparisons)

    with open(target_de_dir / "conditions_de_metadata.json", "w") as f:
        json.dump(de_metadata, f)


def export_gsea_for_web(mod9_dir: Path, mod9b_dir: Path, out_dir: Path) -> None:
    if not mod9_dir.exists():
        return

    gsea_dir = out_dir / "gsea"
    rankings_dir = gsea_dir / "rankings"
    precomputed_dir = gsea_dir / "precomputed"

    rankings_dir.mkdir(parents=True, exist_ok=True)
    precomputed_dir.mkdir(parents=True, exist_ok=True)

    gsea_metadata = {"celltypes": [], "comparisons": {}, "databases": set()}

    # 1. Process Rankings from Mod 9 (Compress to Parquet for On-the-fly Web Compute)
    for ct_path in [d for d in mod9_dir.iterdir() if d.is_dir()]:
        ct_name = ct_path.name
        if ct_name not in gsea_metadata["celltypes"]:
            gsea_metadata["celltypes"].append(ct_name)

        for c_file in ct_path.glob("*_all_genes.csv"):
            comparison_name = c_file.name.replace("_all_genes.csv", "")

            if ct_name not in gsea_metadata["comparisons"]:
                gsea_metadata["comparisons"][ct_name] = []
            gsea_metadata["comparisons"][ct_name].append(comparison_name)

            df = pd.read_csv(c_file)
            cols_to_keep = ["names", "logfoldchanges", "pvals_adj"]
            df = df[[c for c in cols_to_keep if c in df.columns]]
            df = df.dropna(subset=["names", "logfoldchanges"])

            out_pq = rankings_dir / f"{ct_name}_{comparison_name}.parquet"
            df.to_parquet(out_pq, index=False)

    # 2. Process Precomputed GSEA from Mod 9b (Convert to JSON for Instant UI Load)
    if mod9b_dir and mod9b_dir.exists():
        for ct_path in [d for d in mod9b_dir.iterdir() if d.is_dir()]:
            ct_name = ct_path.name

            for db_path in [d for d in ct_path.iterdir() if d.is_dir()]:
                db_name = db_path.name
                gsea_metadata["databases"].add(db_name)

                for sig_file in db_path.glob("*_SIGNIFICANT.csv"):
                    comparison_name = sig_file.name.replace(
                        f"_{db_name}_SIGNIFICANT.csv", ""
                    )
                    df = pd.read_csv(sig_file)

                    net_nodes, net_edges = [], []
                    try:
                        import gseapy as gp

                        nodes, edges = gp.enrichment_map(df)
                        if not nodes.empty and not edges.empty:
                            for idx, row in nodes.iterrows():
                                net_nodes.append(
                                    {
                                        "id": str(idx),
                                        "name": row["Term"],
                                        "nes": float(row["NES"]),
                                        "hits": float(row.get("Hits_ratio", 0.5)),
                                    }
                                )
                            for idx, row in edges.iterrows():
                                net_edges.append(
                                    {
                                        "source": str(row["src_idx"]),
                                        "target": str(row["targ_idx"]),
                                        "weight": float(row["jaccard_coef"]),
                                    }
                                )
                    except Exception:
                        pass

                    out_json = (
                        precomputed_dir / f"{ct_name}_{comparison_name}_{db_name}.json"
                    )
                    with open(out_json, "w") as f:
                        json.dump(
                            {
                                "table": df.to_dict(orient="records"),
                                "network": {"nodes": net_nodes, "edges": net_edges},
                            },
                            f,
                        )

    gsea_metadata["databases"] = sorted(list(gsea_metadata["databases"]))
    gsea_metadata["comparisons"] = {
        k: sorted(list(set(v))) for k, v in gsea_metadata["comparisons"].items()
    }

    with open(gsea_dir / "gsea_metadata.json", "w") as f:
        json.dump(gsea_metadata, f, indent=4)


def export_causal_for_web(mod8c_dir: Path, out_dir: Path) -> None:
    if not mod8c_dir or not mod8c_dir.exists():
        return
    target_dir = out_dir / "causal_ccc"
    target_dir.mkdir(parents=True, exist_ok=True)

    metadata_map = {}

    for net_path in mod8c_dir.rglob("causal_net_*.csv"):
        match = re.search(r"causal_net_(.*?)_(.*?)_to_(.*?)\.csv", net_path.name)
        if not match:
            continue

        comp_name, source_ct, target_ct = match.groups()
        causal_data = {
            "lr_interactions": [],
            "tf_activities": [],
            "network": {"nodes": [], "edges": []},
        }
        comp_folder = net_path.parent

        # Process LR
        lr_path = comp_folder / f"liana_lr_{comp_name}.csv"
        real_source, real_target = source_ct, target_ct

        if lr_path.exists():
            df_lr = pd.read_csv(lr_path)
            df_lr["safe_source"] = df_lr["source"].apply(
                lambda n: re.sub(r"[^\w\s-]", "", str(n)).replace(" ", "_")
            )
            df_lr["safe_target"] = df_lr["target"].apply(
                lambda n: re.sub(r"[^\w\s-]", "", str(n)).replace(" ", "_")
            )

            pair_df = df_lr[
                (df_lr["safe_source"] == source_ct)
                & (df_lr["safe_target"] == target_ct)
            ]
            if not pair_df.empty:
                real_source, real_target = (
                    pair_df.iloc[0]["source"],
                    pair_df.iloc[0]["target"],
                )
                for _, row in (
                    pair_df.sort_values("interaction_stat", ascending=False, key=abs)
                    .head(50)
                    .iterrows()
                ):
                    causal_data["lr_interactions"].append(
                        {
                            "source": str(row["source"]),
                            "target": str(row["target"]),
                            "ligand": str(row["ligand"]),
                            "receptor": str(row["receptor"]),
                            "stat": float(row.get("interaction_stat", 0)),
                            "pval": float(row.get("padj", row.get("pvalue", 1))),
                        }
                    )

        # Process TF
        tf_path = comp_folder / f"tf_estimates_{comp_name}.csv"
        if tf_path.exists():
            df_tf = pd.read_csv(tf_path, index_col=0)
            if real_target in df_tf.index:
                row = df_tf.loc[real_target]
                for tf, stat in (
                    row[row.abs() > 0]
                    .sort_values(ascending=False, key=abs)
                    .head(30)
                    .items()
                ):
                    causal_data["tf_activities"].append(
                        {
                            "cell_type": str(real_target),
                            "tf": str(tf),
                            "stat": float(stat),
                        }
                    )

        # Process Net
        df_net = pd.read_csv(net_path)
        node_dict = {}

        for _, row in df_net.iterrows():
            if pd.isna(row["source"]) or pd.isna(row["target"]):
                continue
            src, tgt = str(row["source"]), str(row["target"])

            src_type = (
                "Receptor" if row.get("source_type") == "input" else "Kinase/Protein"
            )
            tgt_type = "TF" if row.get("target_type") == "output" else "Kinase/Protein"

            if src not in node_dict or src_type != "Kinase/Protein":
                node_dict[src] = src_type
            if tgt not in node_dict or tgt_type != "Kinase/Protein":
                node_dict[tgt] = tgt_type

            causal_data["network"]["edges"].append(
                {
                    "source": src,
                    "target": tgt,
                    "weight": float(row.get("target_weight", row.get("weight", 1.0))),
                    "sign": int(float(row.get("edge_type", 1))),
                }
            )

        for n, t in node_dict.items():
            causal_data["network"]["nodes"].append({"id": n, "type": t})

        out_json_name = f"causal_data_{comp_name}_{source_ct}_to_{target_ct}.json"
        with open(target_dir / out_json_name, "w") as f:
            json.dump(causal_data, f, separators=(",", ":"))

        display_comp = comp_name.replace("_vs_", " vs ")
        metadata_map.setdefault(display_comp, []).append(
            {
                "source": str(real_source),
                "target": str(real_target),
                "file": out_json_name,
            }
        )

    with open(target_dir / "causal_metadata.json", "w") as f:
        json.dump(metadata_map, f, indent=4)


def _min_max_scale(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    v_min, v_max = arr.min(), arr.max()
    return (arr - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(arr)


def prepare_zarr_for_fastapi(
    input_h5ad: Path, output_zarr: Path, spatial_key: str, mod8b_dir: Path
) -> ad.AnnData:
    logger.info(f"Loading {input_h5ad.name}...")
    adata = sc.read_h5ad(input_h5ad)

    if "slide_ID" in adata.obs.columns:
        adata.obs = adata.obs.drop(columns=["slide_ID"])

    if mod8b_dir and mod8b_dir.exists():
        logger.info("Injecting LIANA+ Single-Cell CCC scores into Zarr...")
        lrdata_path = mod8b_dir / "lrdata.h5ad"
        nmf_path = mod8b_dir / "nmf_adata.h5ad"

        if lrdata_path.exists():
            try:
                lrdata = sc.read_h5ad(lrdata_path)
                top_lrs = (
                    lrdata.var.sort_values("morans", ascending=False)
                    .head(50)
                    .index.tolist()
                )
                for pair in top_lrs:
                    val_arr = (
                        lrdata[:, pair].X.toarray().flatten()
                        if sp.issparse(lrdata.X)
                        else lrdata[:, pair].X.flatten()
                    )
                    adata.obs[f"LR_{pair}"] = (
                        pd.Series(_min_max_scale(val_arr), index=lrdata.obs_names)
                        .fillna(0.0)
                        .astype(float)
                    )
            except Exception as e:
                logger.warning(f"Failed to inject LR data: {e}")

        if nmf_path.exists():
            try:
                nmf_adata = sc.read_h5ad(nmf_path)
                for factor in nmf_adata.var_names:
                    val_arr = (
                        nmf_adata[:, factor].X.toarray().flatten()
                        if sp.issparse(nmf_adata.X)
                        else nmf_adata[:, factor].X.flatten()
                    )
                    adata.obs[f"CCC_{factor}"] = (
                        pd.Series(_min_max_scale(val_arr), index=nmf_adata.obs_names)
                        .fillna(0.0)
                        .astype(float)
                    )
            except Exception as e:
                logger.warning(f"Failed to inject NMF data: {e}")

    # Retain invariant logic as requested by user
    if adata.raw is not None:
        adata.X = adata.raw.X.copy()
        del adata.raw
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsc()

    if hasattr(adata, "obsp"):
        del adata.obsp
    if hasattr(adata, "varp"):
        del adata.varp
    if hasattr(adata, "varm"):
        del adata.varm

    keys_to_keep = [spatial_key] + [
        k
        for k in adata.obsm.keys()
        if k.startswith(f"{spatial_key}_")
        or k.startswith("X_umap")
        or k.startswith("spatial_microenv_")
    ]
    for k in list(adata.obsm.keys()):
        if k not in keys_to_keep:
            del adata.obsm[k]

    logger.info(f"Saving optimized Zarr to {output_zarr.name}...")
    adata.write_zarr(output_zarr)
    gc.collect()
    return adata


def run_web_backend_prep(
    module_dir: str | Path,
    input_adata_path: str | Path,
    batch_key: str,
    sample_key: str,
    celltype_key: str,
    microenv_key: str,
    module_1_dir: Path,
    module_3_dir: Path,
    module_5_dir: Path,
    module_6_dir: Path,
    module_7_dir: Path,
    module_8_dir: Path,
    module_8b_dir: Path,
    module_8c_dir: Path,
    module_9_dir: Path,
    analysis_name: str,
    spatial_key: str,
    data_type: str,
    settings: dict,
    DEAnalysis: bool,
    anno_keywords: list,
    module_9b_dir: Path = None,
) -> None:
    module_dir = Path(module_dir)
    module_dir.mkdir(parents=True, exist_ok=True)
    aux_dir = module_dir / "aux_data"
    aux_dir.mkdir(exist_ok=True)

    zarr_filename = f"adata_{analysis_name}_web.zarr"
    zarr_path = module_dir / zarr_filename

    # 1. Zarr Creation
    prepare_zarr_for_fastapi(
        Path(input_adata_path), zarr_path, spatial_key, module_8b_dir
    )

    tf_input_adata_path = module_7_dir / "tf_activity_scores.h5ad"
    if tf_input_adata_path.exists():
        prepare_zarr_for_fastapi(
            tf_input_adata_path,
            module_dir / f"adata_{analysis_name}_tf_web.zarr",
            spatial_key,
            None,
        )

    # 2. Collect Auxiliary Data
    logger.info("\n--- Collecting Pre-Computed Analytics for Backend ---")
    if (module_8_dir / "cpdb_out").exists():
        export_cpdb_for_web_vis(
            module_8_dir / "cpdb_out",
            module_8_dir / "adata.h5ad",
            celltype_key,
            microenv_key,
            aux_dir,
        )

    export_spatial_stats_for_web(module_5_dir, module_6_dir, aux_dir)
    export_qc_for_web(module_1_dir, aux_dir)
    export_segmentations_for_web(Path(input_adata_path), data_type, settings, aux_dir)
    export_de_analysis_for_web(module_3_dir, aux_dir)

    tf_heatmap_src = module_7_dir / "tf_heatmap_data.json"
    if tf_heatmap_src.exists():
        shutil.copy(tf_heatmap_src, aux_dir / "tf_heatmap_data.json")

    if DEAnalysis:
        treatment_col = settings["modules"]["DEAnalysis"].get("treatment_col")
        export_conditions_de_for_web(module_9_dir, aux_dir, celltype_key, treatment_col)
        export_causal_for_web(module_8c_dir, aux_dir)
        export_gsea_for_web(module_9_dir, module_9b_dir, aux_dir)

    # 3. Write Backend Configuration
    logger.info("\n--- Writing Backend Configuration ---")
    temp_adata = sc.read_h5ad(input_adata_path, backed="r")
    backend_config = {
        "zarr_filename": zarr_filename,
        "spatial_key": spatial_key,
        "slide_col": batch_key if batch_key in temp_adata.obs.columns else None,
        "sample_col": sample_key if sample_key in temp_adata.obs.columns else None,
    }
    with open(module_dir / "dataset_config.json", "w") as f:
        json.dump(backend_config, f, indent=4)

    # 4. Archive
    logger.info(f"\n--- Archiving {module_dir} to {module_dir}.tar ---")
    abs_module_dir = module_dir.resolve()
    with tarfile.open(f"{abs_module_dir}.tar", "w") as tar:
        tar.add(abs_module_dir, arcname=abs_module_dir.name)
