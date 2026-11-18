"""
Joint autoencoder latent-space integration of two transcriptomic datasets.

This is the core analysis of the pipeline. It projects samples from two
independently generated datasets (e.g. a microarray-based case/control
cohort and an RNA-seq-based case/control cohort) into a single shared
low-dimensional latent space, using an autoencoder trained on their
combined expression profiles restricted to a common candidate gene panel.

Pipeline steps
--------------
1.  Load and align both datasets to the shared gene panel.
2.  Combine them into a single expression matrix, impute missing values
    gene-wise, log-transform, and standardize each dataset separately
    (dataset-stratified z-scoring) to reduce platform/tissue-driven scale
    differences while preserving within-dataset biological variation.
3.  Train a symmetric dense autoencoder (64 -> 32 -> latent -> 32 -> 64)
    and evaluate it with stratified k-fold cross-validation, using a
    logistic regression probe on the latent embeddings to test whether
    case/control status remains recoverable from the compressed
    representation.
4.  Train a final autoencoder on the full dataset and extract the latent
    embedding.
5.  Project the latent space to two dimensions (UMAP, with a t-SNE
    fallback) and cluster it (k-means, with k selected by silhouette
    score).
6.  Estimate each gene's overall contribution to the latent space from the
    trained encoder's weights, and separately rank genes by a
    non-parametric case-vs-control signal (Mann-Whitney U).
7.  Save all intermediate matrices, metrics, and summary figures.

Usage
-----
    python joint_autoencoder_integration.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import config
from utils import (
    align_feature_order,
    configure_plot_style,
    load_gene_panel,
    load_probe_gene_mapping,
    print_header,
    save_dataframe,
    set_global_seed,
)

warnings.filterwarnings("ignore")

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import tensorflow as tf
    from tensorflow.keras import Model, layers
    from tensorflow.keras.callbacks import EarlyStopping

    HAS_TF = True
except ImportError:
    HAS_TF = False

if not HAS_TF:
    raise ImportError(
        "TensorFlow is required for this script. Install it with: pip install tensorflow"
    )

OUTPUT_DIR = config.RESULTS_DIR / "joint_autoencoder"

DATASET_A_LABEL = "dataset_a"
DATASET_B_LABEL = "dataset_b"


# =============================================================================
# 1. DATA LOADING
# =============================================================================
def load_dataset_a(
    expression_path: Path, metadata_path: Path, gene_to_probe: dict[str, list[str]], genes: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the microarray-based case/control dataset, averaging multi-probe genes."""
    print_header(f"LOADING {DATASET_A_LABEL.upper()}")

    expr_df = pd.read_csv(expression_path, index_col=0)
    metadata = pd.read_csv(metadata_path)

    metadata["group"] = metadata[config.GROUP_ANNOTATION_COLUMN].astype(str).str.extract(
        config.GROUP_ANNOTATION_PATTERN
    )
    metadata = metadata[metadata[config.SAMPLE_ID_COLUMN].isin(expr_df.columns)].copy()

    used_probes: set[str] = set()
    gene_expr_dict, missing_genes = {}, []

    for gene in genes:
        probes = [p for p in gene_to_probe.get(gene, []) if p in expr_df.index]
        if not probes:
            missing_genes.append(gene)
            gene_expr_dict[gene] = pd.Series(np.nan, index=expr_df.columns)
            continue
        available = [p for p in probes if p not in used_probes] or probes[:1]
        used_probes.update(available)
        gene_expr_dict[gene] = expr_df.loc[available].mean(axis=0)

    expr_gene = pd.DataFrame(gene_expr_dict)
    expr_gene = align_feature_order(expr_gene, genes)

    sample_metadata = metadata[[config.SAMPLE_ID_COLUMN, "group"]].copy()
    sample_metadata["dataset"] = DATASET_A_LABEL
    sample_metadata["status"] = np.where(
        sample_metadata["group"] == config.CONTROL_GROUP_LABEL, "Control", "Case"
    )
    sample_metadata = sample_metadata.rename(columns={config.SAMPLE_ID_COLUMN: "sample_id"})
    sample_metadata = sample_metadata[sample_metadata["sample_id"].isin(expr_gene.index)]

    expr_gene = expr_gene.loc[sample_metadata["sample_id"]]

    print(f"Samples kept: {expr_gene.shape[0]}")
    print(f"Matrix shape: {expr_gene.shape}")
    if missing_genes:
        print(f"Genes missing from probe mapping: {missing_genes}")

    return expr_gene, sample_metadata[["sample_id", "dataset", "status"]]


def load_dataset_b(
    control_path: Path,
    case_path: Path,
    genes: list[str],
    expression_threshold: float = config.JOINT_EXPRESSION_THRESHOLD,
    prevalence_threshold: float = config.JOINT_PREVALENCE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the RNA-seq-based case/control dataset and filter low-expression genes."""
    print_header(f"LOADING {DATASET_B_LABEL.upper()}")

    control_df = pd.read_csv(control_path)
    control_df = control_df[control_df[config.CONTROL_GENE_ID_COLUMN].isin(genes)].copy()
    control_df = control_df.set_index(config.CONTROL_GENE_ID_COLUMN)
    expr_control = control_df.drop(columns=[config.CONTROL_SAMPLE_LABEL_COLUMN], errors="ignore").T
    expr_control = expr_control.apply(pd.to_numeric, errors="coerce")

    case_df = pd.read_csv(case_path)
    case_df[config.CASE_GENE_ID_COLUMN] = (
        case_df[config.CASE_GENE_ID_COLUMN].astype(str).str.replace(r"\..*", "", regex=True)
    )
    case_df = case_df[case_df[config.CASE_GENE_NAME_COLUMN].isin(genes)].copy()
    expr_case = case_df.pivot_table(
        index=config.CASE_SAMPLE_ID_COLUMN,
        columns=config.CASE_GENE_NAME_COLUMN,
        values=config.CASE_EXPRESSION_VALUE_COLUMN,
        aggfunc="mean",
    )

    all_genes = sorted(set(expr_control.columns) | set(expr_case.columns))
    expr_control = expr_control.reindex(columns=all_genes)
    expr_case = expr_case.reindex(columns=all_genes)

    frac_low_control = (expr_control < expression_threshold).sum(axis=0) / expr_control.shape[0]
    frac_low_case = (expr_case < expression_threshold).sum(axis=0) / expr_case.shape[0]
    both_low = (frac_low_control >= prevalence_threshold) & (frac_low_case >= prevalence_threshold)

    keep_genes = both_low[~both_low].index.tolist()
    expr_control = expr_control[keep_genes]
    expr_case = expr_case[keep_genes]

    expr_control = align_feature_order(expr_control, genes)
    expr_case = align_feature_order(expr_case, genes)

    control_metadata = pd.DataFrame(
        {"sample_id": expr_control.index, "dataset": DATASET_B_LABEL, "status": "Control"}
    )
    case_metadata = pd.DataFrame(
        {"sample_id": expr_case.index, "dataset": DATASET_B_LABEL, "status": "Case"}
    )

    expr_combined = pd.concat([expr_control, expr_case], axis=0)
    metadata_combined = pd.concat([control_metadata, case_metadata], axis=0, ignore_index=True)
    expr_combined = expr_combined.loc[metadata_combined["sample_id"]]

    print(f"Control samples: {expr_control.shape[0]}")
    print(f"Case samples: {expr_case.shape[0]}")
    print(f"Matrix shape: {expr_combined.shape}")
    removed = both_low[both_low].index.tolist()
    print(f"Genes removed as low-expressed in both groups: {removed if removed else 'None'}")

    return expr_combined, metadata_combined


# =============================================================================
# 2. JOINT DATASET ASSEMBLY
# =============================================================================
def prepare_joint_dataset(
    dataset_a_expr: pd.DataFrame,
    dataset_a_meta: pd.DataFrame,
    dataset_b_expr: pd.DataFrame,
    dataset_b_meta: pd.DataFrame,
    genes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Combine both datasets, impute, log-transform, and scale dataset-wise."""
    print_header("PREPARING JOINT DATASET")

    X = pd.concat([dataset_a_expr, dataset_b_expr], axis=0)
    meta = pd.concat([dataset_a_meta, dataset_b_meta], axis=0, ignore_index=True)

    X = X.loc[meta["sample_id"]]
    X = align_feature_order(X, genes)
    X = X.apply(pd.to_numeric, errors="coerce")

    gene_medians = X.median(axis=0, skipna=True)
    X = X.fillna(gene_medians)

    # log1p is safe here because expression values are non-negative.
    X = np.log1p(X.clip(lower=0))

    # Standardize within each dataset separately, to preserve within-dataset
    # biological variation while reducing platform/tissue-driven scale
    # differences before joint latent modeling.
    X_scaled = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for dataset_label in meta["dataset"].unique():
        sample_ids = meta.loc[meta["dataset"] == dataset_label, "sample_id"]
        X_scaled.loc[sample_ids, :] = StandardScaler().fit_transform(X.loc[sample_ids, :])

    y_dataset = meta["dataset"].map({DATASET_A_LABEL: 0, DATASET_B_LABEL: 1}).values
    y_status = meta["status"].map({"Control": 0, "Case": 1}).values

    print(f"Joint matrix shape: {X_scaled.shape}")
    print(meta["dataset"].value_counts())
    print(meta.groupby(["dataset", "status"]).size())

    return X, X_scaled, meta, y_dataset, y_status


# =============================================================================
# 3. AUTOENCODER MODEL
# =============================================================================
def build_autoencoder(input_dim: int, latent_dim: int = config.AUTOENCODER_LATENT_DIM):
    """Build a symmetric dense autoencoder and return (autoencoder, encoder)."""
    reg = tf.keras.regularizers.l2(config.AUTOENCODER_L2_LAMBDA)
    units_1, units_2 = config.AUTOENCODER_ENCODER_UNITS

    inputs = layers.Input(shape=(input_dim,), name="input")
    x = layers.Dense(units_1, activation="relu", kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.AUTOENCODER_DROPOUT_RATE)(x)
    x = layers.Dense(units_2, activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.AUTOENCODER_DROPOUT_RATE)(x)
    latent = layers.Dense(latent_dim, activation="linear", name="latent")(x)

    x = layers.Dense(units_2, activation="relu", kernel_regularizer=reg)(latent)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units_1, activation="relu", kernel_regularizer=reg)(x)
    outputs = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    autoencoder = Model(inputs, outputs, name="joint_autoencoder")
    encoder = Model(inputs, latent, name="encoder")

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.AUTOENCODER_LEARNING_RATE),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return autoencoder, encoder


def validate_autoencoder_and_latent_classifier(
    X_scaled: pd.DataFrame, y_label: np.ndarray, latent_dim: int = config.AUTOENCODER_LATENT_DIM
) -> pd.DataFrame:
    """Stratified k-fold cross-validation of reconstruction quality and latent-space classification."""
    print_header("CROSS-VALIDATED VALIDATION")

    X_np = X_scaled.values.astype(np.float32)
    skf = StratifiedKFold(
        n_splits=config.AUTOENCODER_CV_SPLITS, shuffle=True, random_state=config.RANDOM_SEED
    )

    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_np, y_label), start=1):
        X_train, X_test = X_np[train_idx], X_np[test_idx]
        y_train, y_test = y_label[train_idx], y_label[test_idx]

        autoencoder, encoder = build_autoencoder(input_dim=X_np.shape[1], latent_dim=latent_dim)
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=config.AUTOENCODER_CV_EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )
        history = autoencoder.fit(
            X_train, X_train,
            validation_data=(X_test, X_test),
            epochs=config.AUTOENCODER_CV_MAX_EPOCHS,
            batch_size=config.AUTOENCODER_BATCH_SIZE,
            verbose=0,
            callbacks=[early_stopping],
        )

        train_mse = np.mean((X_train - autoencoder.predict(X_train, verbose=0)) ** 2)
        test_mse = np.mean((X_test - autoencoder.predict(X_test, verbose=0)) ** 2)

        Z_train = encoder.predict(X_train, verbose=0)
        Z_test = encoder.predict(X_test, verbose=0)

        classifier = LogisticRegression(max_iter=5000, random_state=config.RANDOM_SEED)
        classifier.fit(Z_train, y_train)
        test_prob = classifier.predict_proba(Z_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)

        fold_rows.append({
            "fold": fold,
            "best_epoch": len(history.history["loss"]),
            "train_reconstruction_mse": train_mse,
            "test_reconstruction_mse": test_mse,
            "latent_auc_status": roc_auc_score(y_test, test_prob),
            "latent_accuracy_status": accuracy_score(y_test, test_pred),
            "latent_f1_status": f1_score(y_test, test_pred),
        })
        print(f"Fold {fold}: train MSE={train_mse:.4f}, test MSE={test_mse:.4f}, "
              f"AUC={fold_rows[-1]['latent_auc_status']:.4f}")

    results = pd.DataFrame(fold_rows)
    print("\nCV summary (mean across folds):")
    print(results.mean(numeric_only=True).round(4))
    return results


def train_final_autoencoder(X_scaled: pd.DataFrame, latent_dim: int = config.AUTOENCODER_LATENT_DIM):
    """Train the autoencoder on the full dataset and return the learned latent embedding."""
    print_header("TRAINING FINAL AUTOENCODER ON FULL DATA")

    X_np = X_scaled.values.astype(np.float32)
    autoencoder, encoder = build_autoencoder(input_dim=X_np.shape[1], latent_dim=latent_dim)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config.AUTOENCODER_FINAL_EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    history = autoencoder.fit(
        X_np, X_np,
        validation_split=config.AUTOENCODER_FINAL_VALIDATION_SPLIT,
        epochs=config.AUTOENCODER_FINAL_MAX_EPOCHS,
        batch_size=config.AUTOENCODER_BATCH_SIZE,
        verbose=1,
        callbacks=[early_stopping],
    )

    Z = encoder.predict(X_np, verbose=0)
    Z = pd.DataFrame(Z, index=X_scaled.index, columns=[f"Latent_{i + 1}" for i in range(Z.shape[1])])

    recon = autoencoder.predict(X_np, verbose=0)
    recon_mse = np.mean((X_np - recon) ** 2, axis=1)
    print(f"Final mean reconstruction MSE: {recon_mse.mean():.4f}")

    return autoencoder, encoder, history, Z, recon_mse


# =============================================================================
# 4. LATENT SPACE PROJECTION AND CLUSTERING
# =============================================================================
def project_latent_space(Z: pd.DataFrame, method: str = "umap") -> tuple[pd.DataFrame, str]:
    """Project the latent embedding to 2D with UMAP (t-SNE fallback)."""
    Z_np = Z.values

    if method.lower() == "umap" and HAS_UMAP:
        reducer = umap.UMAP(
            n_neighbors=config.UMAP_N_NEIGHBORS,
            min_dist=config.UMAP_MIN_DIST,
            metric=config.UMAP_METRIC,
            random_state=config.RANDOM_SEED,
        )
        embedding = reducer.fit_transform(Z_np)
        return pd.DataFrame(embedding, index=Z.index, columns=["Dim1", "Dim2"]), "UMAP"

    reducer = TSNE(
        n_components=2,
        perplexity=min(30, max(5, Z.shape[0] // 10)),
        random_state=config.RANDOM_SEED,
        init="pca",
        learning_rate="auto",
    )
    embedding = reducer.fit_transform(Z_np)
    return pd.DataFrame(embedding, index=Z.index, columns=["Dim1", "Dim2"]), "t-SNE"


def choose_best_kmeans_k(Z: pd.DataFrame, k_values=config.KMEANS_K_CANDIDATES) -> tuple[int, pd.DataFrame]:
    """Select the number of k-means clusters that maximizes the silhouette score."""
    print_header("LATENT SPACE CLUSTERING")

    best_k, best_score, score_rows = None, -np.inf, []
    for k in k_values:
        labels = KMeans(n_clusters=k, n_init=config.KMEANS_N_INIT, random_state=config.RANDOM_SEED).fit_predict(Z.values)
        score = silhouette_score(Z.values, labels)
        score_rows.append({"k": k, "silhouette": score})
        print(f"k={k}, silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score

    print(f"Best k selected: {best_k}")
    return best_k, pd.DataFrame(score_rows)


# =============================================================================
# 5. GENE CONTRIBUTION AND VALIDATION
# =============================================================================
def compute_gene_contributions(encoder, gene_names: list[str]) -> pd.DataFrame:
    """Approximate each gene's overall contribution to the latent space from encoder weights."""
    print_header("GENE CONTRIBUTION ANALYSIS")

    dense_layers = [layer for layer in encoder.layers if isinstance(layer, tf.keras.layers.Dense)]
    if len(dense_layers) < 3:
        raise RuntimeError("Unexpected encoder architecture; could not extract weights safely.")

    w1 = dense_layers[0].get_weights()[0]  # input -> hidden 1
    w2 = dense_layers[1].get_weights()[0]  # hidden 1 -> hidden 2
    w3 = dense_layers[2].get_weights()[0]  # hidden 2 -> latent

    # Approximate total contribution as the absolute path strength through the encoder.
    contribution_matrix = np.abs(w1) @ np.abs(w2) @ np.abs(w3)
    contribution_df = pd.DataFrame(
        contribution_matrix, index=gene_names, columns=[f"Latent_{i + 1}" for i in range(w3.shape[1])]
    )
    contribution_df["overall_contribution"] = contribution_df.sum(axis=1)
    contribution_df = contribution_df.sort_values("overall_contribution", ascending=False)

    print("Top contributing genes:")
    print(contribution_df["overall_contribution"].head(15).round(4))
    return contribution_df


def rank_genes_by_case_control_signal(X_log: pd.DataFrame, meta: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Rank genes by a non-parametric case-vs-control effect (Mann-Whitney U)."""
    print_header("GENE-LEVEL CASE VS CONTROL SIGNAL")

    # Align sample-level status labels to X_log's sample_id index explicitly,
    # rather than relying on positional alignment with meta's row index.
    status_by_sample = meta.set_index("sample_id")["status"].reindex(X_log.index)

    rows = []
    for gene in X_log.columns:
        case_vals = X_log.loc[status_by_sample == "Case", gene]
        control_vals = X_log.loc[status_by_sample == "Control", gene]
        _, p_value = mannwhitneyu(case_vals, control_vals, alternative="two-sided")
        rows.append({
            "gene": gene,
            "mannwhitney_p": p_value,
            "median_case_minus_control": case_vals.median() - control_vals.median(),
        })

    result = pd.DataFrame(rows)
    result["abs_effect"] = result["median_case_minus_control"].abs()
    result = result.sort_values(["mannwhitney_p", "abs_effect"], ascending=[True, False])
    print(result.head(top_n).round(4))
    return result


# =============================================================================
# 6. PLOTTING
# =============================================================================
def make_four_panel_figure(
    emb_df: pd.DataFrame, meta: pd.DataFrame, history, contribution_df: pd.DataFrame,
    cluster_labels: np.ndarray, proj_name: str, output_path: Path,
) -> None:
    """Panel A: dataset-colored projection. B: cluster structure. C: real training loss. D: top gene contributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    dataset_labels = sorted(meta["dataset"].unique())
    dataset_palette = dict(zip(dataset_labels, ["#9fd3e6", "#b4aedf"]))
    for dataset_label in dataset_labels:
        idx = meta.loc[meta["dataset"] == dataset_label, "sample_id"]
        temp = emb_df.loc[idx]
        ax1.scatter(temp["Dim1"], temp["Dim2"], s=45, alpha=0.8, label=dataset_label,
                    color=dataset_palette[dataset_label], edgecolor="none")
    ax1.set_title(f"{proj_name} Projection")
    ax1.set_xlabel(f"{proj_name} 1")
    ax1.set_ylabel(f"{proj_name} 2")
    ax1.legend(frameon=False)
    ax1.text(-0.13, 1.06, "A", transform=ax1.transAxes, fontsize=20, fontweight="bold")

    cluster_palette = sns.color_palette("Set2", n_colors=len(np.unique(cluster_labels)))
    marker_map = dict(zip(dataset_labels, ["o", "s"]))
    for cluster_id in sorted(np.unique(cluster_labels)):
        for dataset_label in dataset_labels:
            sub_idx = meta.index[(meta["cluster"] == cluster_id) & (meta["dataset"] == dataset_label)]
            if len(sub_idx) == 0:
                continue
            temp = emb_df.loc[meta.loc[sub_idx, "sample_id"]]
            ax2.scatter(temp["Dim1"], temp["Dim2"], s=45, alpha=0.85,
                        color=cluster_palette[cluster_id], marker=marker_map[dataset_label], edgecolor="none")
    ax2.set_title("Latent Space Clustering")
    ax2.set_xlabel(f"{proj_name} 1")
    ax2.set_ylabel(f"{proj_name} 2")
    ax2.text(-0.13, 1.06, "B", transform=ax2.transAxes, fontsize=20, fontweight="bold")
    cluster_handles = [
        Line2D([0], [0], marker="o", color="w", label=f"Cluster {c + 1}",
               markerfacecolor=cluster_palette[c], markersize=10)
        for c in sorted(np.unique(cluster_labels))
    ]
    dataset_handles = [
        Line2D([0], [0], marker=marker_map[d], color="black", linestyle="None", label=d, markersize=8)
        for d in dataset_labels
    ]
    ax2.legend(handles=cluster_handles + dataset_handles, frameon=True, fontsize=9)

    ax3.plot(history.history["loss"], label="Training loss", linewidth=2)
    if "val_loss" in history.history:
        ax3.plot(history.history["val_loss"], label="Validation loss", linewidth=2)
    ax3.set_title("Autoencoder Loss Curve")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Loss")
    ax3.legend(frameon=False)
    ax3.text(-0.13, 1.06, "C", transform=ax3.transAxes, fontsize=20, fontweight="bold")

    top_genes = contribution_df.head(10).index.tolist()
    heatmap_df = contribution_df.loc[top_genes].drop(columns=["overall_contribution"])
    sns.heatmap(heatmap_df, cmap="coolwarm", ax=ax4, cbar=True, linewidths=0.3, linecolor="gray")
    ax4.set_title("Heatmap of Feature Contributions")
    ax4.set_xlabel("Latent Dimensions")
    ax4.text(-0.18, 1.06, "D", transform=ax4.transAxes, fontsize=20, fontweight="bold")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def make_contribution_barplot(contribution_df: pd.DataFrame, output_path: Path, top_n: int = 15) -> None:
    top_df = contribution_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top_df.index, top_df["overall_contribution"])
    plt.xlabel("Overall contribution to latent space")
    plt.title("Top Gene Contributions")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def make_latent_heatmap(Z: pd.DataFrame, meta: pd.DataFrame, output_path: Path) -> None:
    meta_indexed = meta.set_index("sample_id")
    Z_plot = Z.copy()
    Z_plot["dataset"] = meta_indexed.loc[Z.index, "dataset"]
    Z_plot["status"] = meta_indexed.loc[Z.index, "status"]
    Z_plot["cluster"] = meta_indexed.loc[Z.index, "cluster"]

    dataset_labels = sorted(meta["dataset"].unique())
    order = np.lexsort((
        Z_plot["status"].map({"Control": 0, "Case": 1}).values,
        Z_plot["dataset"].map({d: i for i, d in enumerate(dataset_labels)}).values,
        Z_plot["cluster"].values,
    ))
    Z_only = Z_plot.drop(columns=["dataset", "status", "cluster"]).iloc[order]

    plt.figure(figsize=(10, 8))
    sns.heatmap(Z_only, cmap="vlag", center=0)
    plt.title("Latent Feature Heatmap")
    plt.xlabel("Latent Dimensions")
    plt.ylabel("Samples")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    set_global_seed()
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    genes = load_gene_panel()
    gene_to_probe = load_probe_gene_mapping()

    dataset_a_expr, dataset_a_meta = load_dataset_a(
        config.DATASET_A_EXPRESSION_PATH, config.DATASET_A_METADATA_PATH, gene_to_probe, genes
    )
    dataset_b_expr, dataset_b_meta = load_dataset_b(
        config.DATASET_B_CONTROL_EXPRESSION_PATH, config.DATASET_B_CASE_EXPRESSION_PATH, genes
    )

    X_log, X_scaled, meta, y_dataset, y_status = prepare_joint_dataset(
        dataset_a_expr, dataset_a_meta, dataset_b_expr, dataset_b_meta, genes
    )

    save_dataframe(X_log, OUTPUT_DIR / "joint_expression_log1p.csv")
    save_dataframe(X_scaled, OUTPUT_DIR / "joint_expression_scaled.csv")
    save_dataframe(meta.set_index("sample_id"), OUTPUT_DIR / "joint_metadata.csv")

    cv_results = validate_autoencoder_and_latent_classifier(X_scaled, y_status)
    save_dataframe(cv_results.set_index("fold"), OUTPUT_DIR / "cv_validation_results.csv")

    autoencoder, encoder, history, Z, recon_mse = train_final_autoencoder(X_scaled)
    save_dataframe(Z, OUTPUT_DIR / "latent_features.csv")
    save_dataframe(
        pd.DataFrame({"sample_id": X_scaled.index, "reconstruction_mse": recon_mse}).set_index("sample_id"),
        OUTPUT_DIR / "reconstruction_error_per_sample.csv",
    )

    emb_df, proj_name = project_latent_space(Z, method="umap")
    emb_df.index.name = "sample_id"
    save_dataframe(emb_df, OUTPUT_DIR / f"{proj_name.lower().replace('-', '')}_projection.csv")

    best_k, k_scores = choose_best_kmeans_k(Z)
    save_dataframe(k_scores.set_index("k"), OUTPUT_DIR / "kmeans_silhouette_scores.csv")

    meta = meta.copy()
    meta["cluster"] = KMeans(n_clusters=best_k, n_init=config.KMEANS_N_INIT, random_state=config.RANDOM_SEED).fit_predict(Z.values)

    # Re-save the metadata now that cluster assignments are known, so that
    # downstream scripts (e.g. summarize_latent_results.py) can reuse the
    # actual silhouette-selected clustering instead of recomputing it.
    save_dataframe(meta.set_index("sample_id"), OUTPUT_DIR / "joint_metadata.csv")

    cluster_crosstab = pd.crosstab(meta["cluster"], meta["dataset"])
    save_dataframe(cluster_crosstab, OUTPUT_DIR / "cluster_vs_dataset_table.csv")

    dataset_numeric = meta["dataset"].map({DATASET_A_LABEL: 0, DATASET_B_LABEL: 1})
    ari = adjusted_rand_score(dataset_numeric, meta["cluster"])
    nmi = normalized_mutual_info_score(dataset_numeric, meta["cluster"])
    sil = silhouette_score(Z.values, meta["cluster"])

    print_header("CLUSTER VALIDATION")
    print(f"Adjusted Rand Index (dataset vs cluster): {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    print(f"Silhouette score: {sil:.4f}")
    print(cluster_crosstab)

    contribution_df = compute_gene_contributions(encoder, X_scaled.columns.tolist())
    save_dataframe(contribution_df, OUTPUT_DIR / "gene_contributions_to_latent_space.csv")

    gene_signal_df = rank_genes_by_case_control_signal(X_log, meta, top_n=20)
    save_dataframe(gene_signal_df.set_index("gene"), OUTPUT_DIR / "gene_case_control_signal.csv")

    make_four_panel_figure(
        emb_df=emb_df, meta=meta, history=history, contribution_df=contribution_df,
        cluster_labels=meta["cluster"].values, proj_name=proj_name,
        output_path=OUTPUT_DIR / "figure_joint_autoencoder_4panel.png",
    )
    make_contribution_barplot(contribution_df, OUTPUT_DIR / "figure_top_gene_contributions.png")
    make_latent_heatmap(Z, meta, OUTPUT_DIR / "figure_latent_feature_heatmap.png")

    with open(OUTPUT_DIR / "summary_report.txt", "w", encoding="utf-8") as f:
        f.write("JOINT AUTOENCODER SUMMARY\n" + "=" * 50 + "\n\n")
        f.write(f"Samples: {X_scaled.shape[0]}\nGenes: {X_scaled.shape[1]}\nLatent dimensions: {Z.shape[1]}\n\n")
        f.write("Cross-validated results:\n")
        f.write(str(cv_results.mean(numeric_only=True).round(4)) + "\n\n")
        f.write(f"Best k for KMeans: {best_k}\nARI: {ari:.4f}\nNMI: {nmi:.4f}\nSilhouette: {sil:.4f}\n\n")
        f.write("Cluster vs dataset table:\n" + cluster_crosstab.to_string())
        f.write("\n\nTop contributing genes:\n")
        f.write(contribution_df["overall_contribution"].head(15).round(4).to_string())

    print_header("DONE")
    print(f"All outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
