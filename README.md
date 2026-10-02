# scSpatial-Kit

<img src="docs/assets/logo.png" alt="Logo" width="500">

An end-to-end Python pipeline for the processing, analysis, and visualization of Spatial Transcriptomics data. Built specifically to handle **NanoString CosMx** and **10x Genomics Xenium** datasets, this pipeline uses modern spatial data frameworks (spatialdata, scanpy, squidpy), advanced spatial statistics (muspan), downstream TF/CCC Analysis, and Causal Network Inference. 

- [Read the documentation on the pipeline (**scSpatial-Kit**)](https://kitku15.github.io/scSpatial-Kit/)

- [Read the documentation on the visualization tool (**Spatial-VisKit**)](https://kitku15.github.io/scSpatial-Kit/svk/home/)

## 🔑 Key Features

- Non linear Dimension reduction with [scVI](https://docs.scvi-tools.org/en/1.3.3/user_guide/models/scvi.html#), [scANVI](https://docs.scvi-tools.org/en/1.3.3/user_guide/models/scanvi.html), and [scVIVA](https://docs.scvi-tools.org/en/1.3.3/user_guide/models/scviva.html)
- Supports both Machine Learning-based  ([CellTypist](https://www.celltypist.org/)) and Marker-based ([ScType](https://github.com/kris-nader/sc-type-py)) cell type annotation.
- Generates Delaunay, KNN, and Proximity graphs, along with cross-Pair Correlation Functions (PCF), and cell morphological metrics via the [SquidPy](https://squidpy.readthedocs.io/en/stable) and [MuSpAn](https://www.muspan.co.uk/) library.
- Transcription factor activity inference via [DecoupleR](https://decoupler.readthedocs.io/en/latest/index.html) and targeted pseudobulk Differential Expression using PyDESeq2.
- Cell-Cell Communication (CCC) & Causal Networks - Broad microenvironment CCC via [CellphoneDB](https://cellphonedb.readthedocs.io/en/latest), continuous spatial CCC via LIANA+, and downstream causal signaling cascade inference via Corneto.
- Fast Gene Set Enrichment Analysis with [PyFgsea](https://github.com/shayuanxukuang/pyfgsea)
- Start from raw machine outputs, or use your own pre-processed .h5ad file.

## ⭐ Pipeline Structure

```mermaid
graph TD
    %% Styling
    classDef array fill:#f9d0c4,stroke:#333,stroke-width:2px;
    classDef linear fill:#d4e157,stroke:#333,stroke-width:2px;
    classDef parallel fill:#81d4fa,stroke:#333,stroke-width:2px;
    classDef depend fill:#ce93d8,stroke:#333,stroke-width:2px;
    classDef sink fill:#ffcc80,stroke:#333,stroke-width:2px;

    subgraph "Phase 1: PBS Job Array (01_run_qc.sh)"
        M0[0_FormatData<br/>Raw -> Zarr]:::array
        M1[1_QualityControl<br/>Slide Level QC]:::array
        M0 --> M1
    end

    subgraph "Phase 2: Snakemake Orchestration (02_run_downstream.sh)"
        M1b[1b_MergeData<br/>Combine Slides]:::linear
        M2[2_DimensionReduction<br/>UMAP/scVIVA]:::linear
        M3[3_Annotate<br/>Cell Typing]:::linear
        
        M1 -.-> M1b
        M1b --> M2
        M2 --> M3
        
        %% The Parallel Fan-out
        M4[4_ViewImages]:::parallel
        M5[5_SpatialStat]:::parallel
        M6[6_MuSpan]:::parallel
        M7[7_Decoupler<br/>TF & Pseudobulk]:::parallel
        
        M3 --> M4
        M3 --> M5
        M3 --> M6
        M3 --> M7
        
        %% Downstream Blockers
        M8[8_Cellphonedb]:::depend
        M8b[8b_LIANA]:::depend
        M8c[8c_LIANA_Causal]:::depend
        M9[9_DEAnalysis]:::depend
        M9b[9b_GSEA]:::depend
        
        M3 --> M8
        M3 --> M8b
        M3 --> M8c
        
        M7 --> M8
        M7 --> M8b
        M7 --> M8c
        M7 --> M9
        M7 --> M9b
        
        %% Terminal Sink
        M10[10_WebVisPrep<br/>Pack Zarr & JSONs]:::sink
        
        M4 --> M10
        M5 --> M10
        M6 --> M10
        M8 --> M10
        M8b --> M10
        M8c --> M10
        M9 --> M10
        M9b --> M10
    end
```

## ⭐ About
This project was completed as part of the MRes in Bioinformatics and Theoretical Systems Biology at Imperial College London. It builds upon the [The ReCoDe-spatial-transcriptomics repository](https://github.com/ImperialCollegeLondon/ReCoDe-spatial-transcriptomics). The work was supervised by Tamas Korcsmaros and Balazs Bohar.

- 😸 Author: Bunga Tiasyaira Hutasuhut (Syaii)
- 📮 Email: akbarah97@gmail.com
