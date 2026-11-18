"""
Central configuration for the cross-condition transcriptomic integration pipeline.

All file paths, thresholds, and model hyperparameters used across the analysis
scripts are defined here so that the pipeline can be reproduced or adapted to
new input data by editing a single file.

Directory layout expected by default
-------------------------------------
    data/
        dataset_a/
            expression_matrix.csv      # samples as columns, probes/genes as rows
            sample_metadata.csv        # sample annotation (one row per sample)
        dataset_b/
            case_expression.csv        # e.g. tumor / disease-tissue RNA-seq quantification
            control_expression.csv     # e.g. normal / reference-tissue RNA-seq quantification
        gene_panel.txt                 # one gene identifier per line
        probe_gene_mapping.csv         # two columns: probe_id, gene_symbol (dataset_a only)
        deg_results/
            dataset_a_deg.csv          # differential expression table (gene, logFC, P.Value, ...)
            dataset_b_deg.csv
        enrichment_results/
            go_bp_enrichment.csv
            kegg_enrichment.csv
            reactome_enrichment.csv
    results/                            # all outputs are written here (created automatically)

All paths below are relative to the repository root so the project runs
unmodified on any machine after the expected input files are placed under
``data/``.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

DATASET_A_DIR: Path = DATA_DIR / "dataset_a"
DATASET_B_DIR: Path = DATA_DIR / "dataset_b"
DEG_RESULTS_DIR: Path = DATA_DIR / "deg_results"
ENRICHMENT_RESULTS_DIR: Path = DATA_DIR / "enrichment_results"

GENE_PANEL_PATH: Path = DATA_DIR / "gene_panel.txt"
PROBE_GENE_MAPPING_PATH: Path = DATA_DIR / "probe_gene_mapping.csv"

# Input files -----------------------------------------------------------------
DATASET_A_EXPRESSION_PATH: Path = DATASET_A_DIR / "expression_matrix.csv"
DATASET_A_METADATA_PATH: Path = DATASET_A_DIR / "sample_metadata.csv"

DATASET_B_CASE_EXPRESSION_PATH: Path = DATASET_B_DIR / "case_expression.csv"
DATASET_B_CONTROL_EXPRESSION_PATH: Path = DATASET_B_DIR / "control_expression.csv"

# Gene-level DE result tables (one row per gene symbol), used for the
# cross-dataset overlap analysis.
DATASET_A_DEG_PATH: Path = DEG_RESULTS_DIR / "dataset_a_deg.csv"
DATASET_B_DEG_PATH: Path = DEG_RESULTS_DIR / "dataset_b_deg.csv"

# Volcano-plot background tables. Dataset B's DE table is already gene-level
# and is reused directly. Dataset A's volcano background is at probe level
# (one row per probe, as produced by the microarray DE workflow) and is
# resolved to genes via PROBE_GENE_MAPPING_PATH.
DATASET_A_VOLCANO_PATH: Path = DEG_RESULTS_DIR / "dataset_a_deg_by_probe.csv"
DATASET_B_VOLCANO_PATH: Path = DATASET_B_DEG_PATH

ENRICHMENT_FILES: dict[str, Path] = {
    "GO_BP": ENRICHMENT_RESULTS_DIR / "go_bp_enrichment.csv",
    "KEGG": ENRICHMENT_RESULTS_DIR / "kegg_enrichment.csv",
    "Reactome": ENRICHMENT_RESULTS_DIR / "reactome_enrichment.csv",
}

# ---------------------------------------------------------------------------
# Sample metadata column conventions (dataset A, GEO-style annotation)
# ---------------------------------------------------------------------------
SAMPLE_ID_COLUMN: str = "sample_id"
GROUP_ANNOTATION_COLUMN: str = "annotation"
GROUP_ANNOTATION_PATTERN: str = r"subject group:\s*([^;]+)"
CONTROL_GROUP_LABEL: str = "CNTL"

# ---------------------------------------------------------------------------
# Dataset B (RNA-seq, two-cohort) column conventions
# ---------------------------------------------------------------------------
CONTROL_GENE_ID_COLUMN: str = "Description"
CONTROL_SAMPLE_LABEL_COLUMN: str = "Name"
CASE_GENE_ID_COLUMN: str = "gene_id"
CASE_GENE_NAME_COLUMN: str = "gene_name"
CASE_SAMPLE_ID_COLUMN: str = "case_id"
CASE_EXPRESSION_VALUE_COLUMN: str = "fpkm_unstranded"

# ---------------------------------------------------------------------------
# Differential expression thresholds
# ---------------------------------------------------------------------------
# Nominal-significance, modest effect-size threshold appropriate for a
# case/control comparison with subtle, coordinated effect sizes (dataset A).
DATASET_A_PVALUE_CUTOFF: float = 0.05
DATASET_A_LOGFC_CUTOFF: float = 0.3

# Stricter effect-size threshold appropriate for a tissue-level case/control
# comparison with larger expected fold changes (dataset B).
DATASET_B_PVALUE_CUTOFF: float = 0.05
DATASET_B_LOGFC_CUTOFF: float = 1.0

# ---------------------------------------------------------------------------
# Low-expression filtering of the candidate gene panel
# ---------------------------------------------------------------------------
# A gene is dropped only if it falls below EXPRESSION_THRESHOLD in more than
# PREVALENCE_THRESHOLD of samples in *both* groups being compared.
EXPRESSION_THRESHOLD: float = 1.0
PREVALENCE_THRESHOLD: float = 0.8

# ---------------------------------------------------------------------------
# Gene-signature classification models
# ---------------------------------------------------------------------------
DATASET_A_CLASSIFIER_PARAMS: dict = {
    "penalty": "l1",
    "solver": "liblinear",
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
}

DATASET_B_CLASSIFIER_PARAMS: dict = {
    "penalty": "elasticnet",
    "solver": "saga",
    "C": 0.1,
    "l1_ratio": 0.5,
    "class_weight": "balanced",
    "max_iter": 4000,
    "random_state": RANDOM_SEED,
}

CLASSIFIER_CV_SPLITS: int = 5
CLASSIFIER_PERMUTATION_TEST_ITERATIONS: int = 200

# ---------------------------------------------------------------------------
# Joint autoencoder architecture and training
# ---------------------------------------------------------------------------
AUTOENCODER_LATENT_DIM: int = 8
AUTOENCODER_ENCODER_UNITS: tuple[int, int] = (64, 32)
AUTOENCODER_DROPOUT_RATE: float = 0.15
AUTOENCODER_L2_LAMBDA: float = 1e-4
AUTOENCODER_LEARNING_RATE: float = 1e-3
AUTOENCODER_BATCH_SIZE: int = 16

AUTOENCODER_CV_SPLITS: int = 5
AUTOENCODER_CV_MAX_EPOCHS: int = 300
AUTOENCODER_CV_EARLY_STOPPING_PATIENCE: int = 30

AUTOENCODER_FINAL_MAX_EPOCHS: int = 320
AUTOENCODER_FINAL_EARLY_STOPPING_PATIENCE: int = 35
AUTOENCODER_FINAL_VALIDATION_SPLIT: float = 0.20

# Low-expression filtering thresholds used specifically when assembling
# dataset B's contribution to the joint expression matrix.
JOINT_EXPRESSION_THRESHOLD: float = EXPRESSION_THRESHOLD
JOINT_PREVALENCE_THRESHOLD: float = PREVALENCE_THRESHOLD

# ---------------------------------------------------------------------------
# Latent space projection and clustering
# ---------------------------------------------------------------------------
UMAP_N_NEIGHBORS: int = 40
UMAP_MIN_DIST: float = 0.5
UMAP_METRIC: str = "euclidean"

KMEANS_K_CANDIDATES: tuple[int, ...] = (2, 3, 4, 5, 6)
KMEANS_N_INIT: int = 30

# ---------------------------------------------------------------------------
# Plot styling (cosmetic only; does not affect any computed result)
# ---------------------------------------------------------------------------
PLOT_STYLE: dict = {
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}


def ensure_output_dirs() -> None:
    """Create the results directory tree if it does not already exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
