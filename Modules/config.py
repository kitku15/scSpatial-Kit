import os
import sys
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import hashlib
import json


def generate_run_id(param_dict: dict) -> str:
    """Generates a consistent 6-character hash from a dictionary of parameters."""
    if not param_dict:
        return "run_default"
    # Convert dict to a sorted JSON string so order doesn't change the hash
    param_str = json.dumps(param_dict, sort_keys=True, default=str)
    hash_str = hashlib.md5(param_str.encode("utf-8")).hexdigest()[:6]
    return f"run_{hash_str}"


# --- Compatibility Layer ---
class DictCompatibleModel(BaseModel):
    """Allows Pydantic models to be accessed like dictionaries to prevent breaking __main__.py."""

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, item, default=None):
        return getattr(self, item, default)


# --- Pydantic Schemas ---
class ProjectConfig(DictCompatibleModel):
    analysis_name: str
    data_type: str = Field(pattern="^(CosMx|Xenium)$")
    slide_name: str | None = None
    batch_key: str = "slide_id"
    sample_key: str = "sample_id"


class IOConfig(DictCompatibleModel):
    base_raw_dir: str | None = None
    base_zarr_dir: str = "data_zarrs"
    dataset_dir: str | None = None
    zarr_dir: str | None = None
    dataset_id: str | None = None
    raw_data: dict[str, dict] = Field(default_factory=dict)
    entry_point: str = Field(
        default="0", description="Module number to start from (e.g., '3' or '8b')"
    )
    custom_input_adata: str | None = Field(
        default=None, description="Path to preprocessed .h5ad"
    )


class PipelineSettings(DictCompatibleModel):
    modules: list[str]


class AppConfig(DictCompatibleModel):
    log_level: str = "INFO"
    seed: int = 42
    project: ProjectConfig
    io: IOConfig
    pipeline: PipelineSettings
    # We leave modules as a raw dictionary so downstream code isn't broken!
    modules: dict[str, dict] = Field(default_factory=dict)


# These exist for Just-in-Time validation inside __main__.py
class QCConfig(BaseModel):
    min_counts: int | None = Field(default=None, ge=0)
    min_cells: int | None = Field(default=None, ge=0)
    min_genes: int | None = Field(default=None, ge=0)
    min_area: float | None = None
    max_area: float | None = None
    min_dapi: float | None = None
    fov_metadata_path: str | None = None
    metadata_join_col: str | None = None
    proseg_zarr_path: str | None = None
    proseg_cell_id_col: str = "original_cell_id"
    original_cell_id_col: str = "index"


class MergeConfig(BaseModel):
    input_files: list[str] = Field(default_factory=list)
    slide_names: list[str] = Field(default_factory=list)


class SpatialStatConfig(BaseModel):
    chosen_cluster: str
    condition_key: Optional[str] = None
    reference_condition: Optional[str] = None
    subsample_fraction: float = Field(
        1.0,
        ge=0.01,
        le=1.0,
        description="Fraction of cells to use for computationally heavy spatial metrics (co-occurence, morans). 1.0 = no subsampling.",
    )
    n_perms_moran: int = Field(100, ge=10)
    n_jobs: int = Field(-1, description="-1 uses all available cores.")


class MuSpanConfig(BaseModel):
    selection_names: List[str]
    chosen_cluster: str
    cell_types: List[str]
    transcripts: List[str]
    selected_celltypes: Optional[List[str]] = None
    selected_fovs: dict = Field(default_factory=dict)
    min_edge_distance: float = 0.0
    max_edge_distance: float = 50.0
    distance_list: List[float] = [20.0, 50.0]
    min_edge_distance_shape: float = 0.0
    max_edge_distance_shape: float = 10.0
    k_list: List[int] = [5, 10, 20]
    network_celltypes: List[str]
    condition_key: Optional[str] = None
    reference_condition: Optional[str] = None


class DecouplerConfig(BaseModel):
    organism: str = Field(pattern="^(human|mouse|rat)$")
    grn: str = Field(pattern="^(collectri|dorothea)$")
    dorothea_levels: list[str] = ["A", "B", "C"]
    celltype_key: str
    active_tfs_file_name: str = "active_tf.txt"


class CellphoneDBConfig(BaseModel):
    chosen_cluster: str
    cpdb_version: str = "v5.0.0"
    human: bool = True
    celltypes: Optional[list[str]] = None
    target_genes: Optional[list[str]] = None
    gene_family: Optional[str] = None
    microenv_resolution: float = 0.5
    target_microenv: Optional[list[str]] = None
    target_sample: Optional[list[str]] = None
    use_cellsign: bool = False


class LIANAConfig(BaseModel):
    bandwidth: int = 400
    cutoff: float = 0.1
    nz_prop: float = 0.05
    n_nmf_components: int = 5
    resource_name: str = "consensus"


class DEAnalysisConfig(BaseModel):
    celltype_col: str
    treatment_col: str
    comparisons: list[list[str]]  # e.g., [["Stim", "Ctrl"]]
    pca_n_top_genes: int = Field(
        default=2000,
        ge=10,
        description="Number of highly variable genes to compute for pseudobulk PCA prior to DE.",
    )
    pca_n_comps: int = Field(
        default=50,
        ge=2,
        description="Number of principal components to calculate for pseudobulk PCA prior to DE.",
    )
    exclude_genes_file: Optional[str] = Field(
        default=None,
        description="Path to a text file containing genes to exclude before PCA (one gene per line).",
    )


class GSEAConfig(BaseModel):
    databases: list[str] = ["MSigDB_Hallmark_2020", "KEGG_2021_Human", "Reactome_2022"]
    gene_col: str = "names"
    score_col: str = "logfoldchanges"  # Matches PyDESeq2 output
    nes_threshold: float = Field(
        default=2.0, description="Minimum absolute NES to consider significant."
    )
    padj_threshold: float = Field(
        default=0.05, description="Maximum Adjusted P-value to consider significant."
    )
    min_size: int = 5
    max_size: int = 1000
    threads: int = 4


class WebVisPrepConfig(BaseModel):
    primary_annotation: str = "Broad_Celltype"
    microenv_col: str = "spatial_microenvironment"
    annotation_columns: list[str] = ["leiden", "CellTypist", "sctype", "cluster"]


# --- Loader ---
def load_config() -> AppConfig:
    config_path_str = os.getenv("scSpatial-Kit", "config_CosMx.toml")
    config_path = Path(config_path_str)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "rb") as f:
        raw_data = tomllib.load(f)

    try:
        return AppConfig(**raw_data)
    except ValidationError as e:
        print(f"❌ CONFIGURATION ERROR in {config_path.name}:\n{e}")
        sys.exit(1)


settings = load_config()

analysis_name = settings.project.analysis_name
analysis_dir = Path(f"analysis_{analysis_name}")

MODULES = {
    name: {"name": name, "dir": analysis_dir / name}
    for name in settings.pipeline.modules
}


def get_module(index):
    key = next((k for k in MODULES if k.startswith(f"{index}_")), None)
    if key:
        return MODULES[key]["name"], MODULES[key]["dir"]
    raise ValueError(f"Module starting with {index}_ not found.")
