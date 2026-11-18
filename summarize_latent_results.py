"""
Print a concise, reviewer-ready summary of the joint autoencoder outputs.

Loads the latent features, sample metadata, and gene contribution table
previously produced by ``joint_autoencoder_integration.py`` and reports
cluster composition, clustering quality, and the top contributing genes.

This script does not choose a number of clusters itself: the cluster
assignments it summarizes are the ones produced by
``joint_autoencoder_integration.py`` using silhouette-selected k (see
``choose_best_kmeans_k``), loaded here from the saved metadata.

Usage
-----
    python summarize_latent_results.py
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

import config
from utils import print_header, save_dataframe

OUTPUT_DIR = config.RESULTS_DIR / "joint_autoencoder"


def main() -> None:
    latent_features = pd.read_csv(OUTPUT_DIR / "latent_features.csv", index_col=0)
    metadata = pd.read_csv(OUTPUT_DIR / "joint_metadata.csv", index_col=0)
    contribution_df = pd.read_csv(OUTPUT_DIR / "gene_contributions_to_latent_space.csv", index_col=0)

    if "cluster" not in metadata.columns:
        raise ValueError(
            "No 'cluster' column found in joint_metadata.csv. Run "
            "joint_autoencoder_integration.py first; it selects the number "
            "of clusters by silhouette score and saves the resulting "
            "cluster assignments alongside the rest of the metadata."
        )

    print_header("LATENT SPACE SUMMARY")
    print(f"\nNumber of samples: {latent_features.shape[0]}")
    print(f"Latent dimensions: {latent_features.shape[1]}")

    print("\n[Cluster Counts]")
    cluster_counts = metadata["cluster"].value_counts().sort_index()
    print(cluster_counts)

    print("\n[Cluster Composition by Dataset]")
    cluster_by_dataset = pd.crosstab(metadata["cluster"], metadata["dataset"])
    print(cluster_by_dataset)

    silhouette = silhouette_score(latent_features.values, metadata["cluster"])
    print(f"\n[Clustering Quality]\nSilhouette Score: {silhouette:.4f}")

    dataset_labels = sorted(metadata["dataset"].unique())
    dataset_numeric = metadata["dataset"].map({label: i for i, label in enumerate(dataset_labels)})
    ari = adjusted_rand_score(dataset_numeric, metadata["cluster"])
    nmi = normalized_mutual_info_score(dataset_numeric, metadata["cluster"])
    print("\n[Dataset vs Cluster Alignment]")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

    print("\n[Feature Contributions]")
    top_genes = contribution_df.sort_values("overall_contribution", ascending=False).head(10)
    print(top_genes["overall_contribution"].round(4))
    print("\nContribution method: approximate absolute weight propagation across "
          "encoder layers (|W1| . |W2| . |W3|)")

    summary_df = cluster_by_dataset.copy()
    summary_df["Total"] = cluster_counts
    save_dataframe(summary_df, OUTPUT_DIR / "cluster_summary_full.csv")


if __name__ == "__main__":
    main()
