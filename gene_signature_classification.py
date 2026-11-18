"""
Evaluate the discriminatory power of the candidate gene panel as a multi-gene
classifier, separately for each dataset, and rank genes by their
contribution to the classifier.

Dataset A (microarray, single processed cohort): the expression matrix is
already normalized, so a single L1-regularized logistic regression model is
fit on the full sample set and evaluated by ROC AUC with a Mann-Whitney U
test on the predicted probabilities.

Dataset B (RNA-seq, two independently sourced cohorts): the case and control
cohorts require batch-effect correction before combination (ComBat, with a
per-batch quantile-normalization fallback if the ``combat`` package is not
installed), followed by 5-fold stratified cross-validation of an
elastic-net logistic regression model.

Usage
-----
    python gene_signature_classification.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.preprocessing import StandardScaler

import config
from utils import (
    configure_plot_style,
    load_gene_panel,
    load_probe_gene_mapping,
    print_header,
    save_dataframe,
    set_global_seed,
)


def build_dataset_a_matrix(gene_panel: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Assemble the feature matrix (samples x genes) for the microarray dataset."""
    expr_df = pd.read_csv(config.DATASET_A_EXPRESSION_PATH, index_col=0)
    metadata = pd.read_csv(config.DATASET_A_METADATA_PATH)

    metadata["group"] = metadata[config.GROUP_ANNOTATION_COLUMN].str.extract(
        config.GROUP_ANNOTATION_PATTERN
    )
    metadata["label"] = (metadata["group"] != config.CONTROL_GROUP_LABEL).astype(int)
    labels = metadata.set_index(config.SAMPLE_ID_COLUMN)["label"]

    common_samples = expr_df.columns.intersection(labels.index)
    expr_df = expr_df[common_samples]
    labels = labels[common_samples]

    gene_to_probes = load_probe_gene_mapping()
    feature_columns, valid_genes = [], []
    for gene in gene_panel:
        probes = [p for p in gene_to_probes.get(gene, []) if p in expr_df.index]
        if not probes:
            continue
        feature_columns.append(expr_df.loc[probes].mean(axis=0).values)
        valid_genes.append(gene)

    X = np.column_stack(feature_columns)
    y = labels.values
    return X, y, valid_genes


def apply_batch_correction(X_log: np.ndarray, batch_labels: np.ndarray, gene_names: list[str]) -> np.ndarray:
    """
    Correct for batch effects between two sample cohorts (e.g. two data
    sources contributing the case and control groups).

    Uses ComBat when the optional ``combat`` package is available; otherwise
    falls back to a per-batch quantile normalization.
    """
    try:
        from combat.pycombat import pycombat

        df_for_combat = pd.DataFrame(
            X_log.T, index=gene_names, columns=[f"s{i}" for i in range(X_log.shape[0])]
        )
        corrected = pycombat(df_for_combat, list(batch_labels))
        print("Applied ComBat batch correction.")
        return corrected.values.T
    except ImportError:
        from sklearn.preprocessing import QuantileTransformer

        print("Package 'combat' not installed; falling back to per-batch quantile normalization.")
        batches = np.unique(batch_labels)
        if len(batches) != 2:
            raise ValueError("Quantile-normalization fallback expects exactly two batches.")

        qt = QuantileTransformer(output_distribution="normal", random_state=config.RANDOM_SEED)
        reference_mask = batch_labels == batches[0]
        other_mask = batch_labels == batches[1]

        corrected = np.empty_like(X_log)
        corrected[reference_mask] = qt.fit_transform(X_log[reference_mask])
        corrected[other_mask] = qt.transform(X_log[other_mask])
        return corrected


def build_dataset_b_matrix(gene_panel: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Assemble the batch-corrected feature matrix (samples x genes) for the RNA-seq dataset."""
    control_df = pd.read_csv(config.DATASET_B_CONTROL_EXPRESSION_PATH)
    control_df = control_df.set_index(config.CONTROL_GENE_ID_COLUMN)

    case_df = pd.read_csv(config.DATASET_B_CASE_EXPRESSION_PATH)
    case_df[config.CASE_GENE_ID_COLUMN] = (
        case_df[config.CASE_GENE_ID_COLUMN].astype(str).str.replace(r"\..*", "", regex=True)
    )

    records = []
    for gene in gene_panel:
        if gene not in control_df.index:
            print(f"[skip] {gene} not found in control expression data")
            continue
        control_values = control_df.loc[gene].drop(
            labels=[config.CONTROL_SAMPLE_LABEL_COLUMN], errors="ignore"
        ).astype(float).values
        case_values_df = case_df[case_df[config.CASE_GENE_NAME_COLUMN] == gene]
        if case_values_df.empty:
            print(f"[skip] {gene} not found in case expression data")
            continue
        case_values = case_values_df[config.CASE_EXPRESSION_VALUE_COLUMN].astype(float).values
        records.append({"gene": gene, "control": control_values, "case": case_values})

    n_control, n_case = len(records[0]["control"]), len(records[0]["case"])
    valid_genes = [r["gene"] for r in records]

    X_control = np.column_stack([r["control"] for r in records])
    X_case = np.column_stack([r["case"] for r in records])
    X_raw = np.vstack([X_control, X_case])
    y = np.concatenate([np.zeros(n_control), np.ones(n_case)])
    batch_labels = np.array([0] * n_control + [1] * n_case)

    X_log = np.log2(X_raw + 1)
    X_corrected = apply_batch_correction(X_log, batch_labels, valid_genes)

    rng = np.random.default_rng(config.RANDOM_SEED)
    shuffle_idx = rng.permutation(len(y))
    X_corrected, y = X_corrected[shuffle_idx], y[shuffle_idx]

    keep_mask = (X_corrected < config.EXPRESSION_THRESHOLD).mean(axis=0) <= config.PREVALENCE_THRESHOLD
    valid_genes = [g for g, keep in zip(valid_genes, keep_mask) if keep]
    return X_corrected[:, keep_mask], y, valid_genes


def plot_roc(
    fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, p_value: float, title: str, output_path: Path | str
) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, color="blue", lw=3, label=f"AUC = {roc_auc:.2f}\nP = {p_value:.2e}")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", lw=1.5)
    ax.set_xlim([-0.01, 1])
    ax.set_ylim([0, 1.01])
    ax.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=11, frameon=False)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def run_dataset_a_classification(gene_panel: list[str]) -> None:
    print_header("GENE SIGNATURE CLASSIFICATION: DATASET A")
    X, y, valid_genes = build_dataset_a_matrix(gene_panel)
    print(f"Valid genes: {len(valid_genes)}")

    X_scaled = StandardScaler().fit_transform(X)
    model = LogisticRegression(**config.DATASET_A_CLASSIFIER_PARAMS)
    model.fit(X_scaled, y)

    y_prob = model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    _, p_value = mannwhitneyu(y_prob[y == 0], y_prob[y == 1], alternative="two-sided")

    plot_roc(fpr, tpr, roc_auc, p_value, "Multi-Gene Model - Dataset A",
              config.RESULTS_DIR / "roc_dataset_a.png")

    importance_df = pd.DataFrame(
        {"gene": valid_genes, "importance": np.abs(model.coef_[0])}
    ).sort_values("importance", ascending=False)
    print("\nTop 10 genes:")
    print(importance_df.head(10).to_string(index=False))
    save_dataframe(importance_df, config.RESULTS_DIR / "gene_ranking_dataset_a.csv", index=False)


def run_dataset_b_classification(gene_panel: list[str]) -> None:
    print_header("GENE SIGNATURE CLASSIFICATION: DATASET B")
    X, y, valid_genes = build_dataset_b_matrix(gene_panel)
    print(f"Genes after low-expression filtering: {len(valid_genes)}")

    cv = StratifiedKFold(
        n_splits=config.CLASSIFIER_CV_SPLITS, shuffle=True, random_state=config.RANDOM_SEED
    )
    model = LogisticRegression(**config.DATASET_B_CLASSIFIER_PARAMS)

    mean_fpr = np.linspace(0, 1, 200)
    tprs, aucs, all_probs, all_true = [], [], [], []

    for train_idx, test_idx in cv.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        model.fit(X_train, y[train_idx])
        y_prob = model.predict_proba(X_test)[:, 1]

        all_probs.append(y_prob)
        all_true.append(y[test_idx])

        fpr, tpr, _ = roc_curve(y[test_idx], y_prob)
        aucs.append(auc(fpr, tpr))
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    all_probs, all_true = np.concatenate(all_probs), np.concatenate(all_true)
    _, p_value = mannwhitneyu(all_probs[all_true == 0], all_probs[all_true == 1], alternative="two-sided")

    # Additional cross-validated significance check: is the observed mean AUC
    # higher than expected under random label permutation?
    _, _, permutation_p_value = permutation_test_score(
        LogisticRegression(**config.DATASET_B_CLASSIFIER_PARAMS),
        X, y,
        scoring="roc_auc",
        cv=cv,
        n_permutations=config.CLASSIFIER_PERMUTATION_TEST_ITERATIONS,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )

    print(f"Mean CV AUC: {mean_auc:.3f} (+/- {np.std(aucs):.3f})")
    print(f"Permutation test p-value: {permutation_p_value:.4f}")
    plot_roc(mean_fpr, mean_tpr, mean_auc, p_value, "Multi-Gene Model - Dataset B",
              config.RESULTS_DIR / "roc_dataset_b.png")

    scaler_final = StandardScaler()
    X_scaled_final = scaler_final.fit_transform(X)
    model.fit(X_scaled_final, y)
    importance_df = pd.DataFrame(
        {"gene": valid_genes, "importance": np.abs(model.coef_[0])}
    ).sort_values("importance", ascending=False)
    print("\nTop 10 genes:")
    print(importance_df.head(10).to_string(index=False))
    save_dataframe(importance_df, config.RESULTS_DIR / "gene_ranking_dataset_b.csv", index=False)


def main() -> None:
    config.ensure_output_dirs()
    configure_plot_style()
    set_global_seed()

    gene_panel = load_gene_panel()
    run_dataset_a_classification(gene_panel)
    run_dataset_b_classification(gene_panel)


if __name__ == "__main__":
    main()
