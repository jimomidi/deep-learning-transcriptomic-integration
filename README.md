# Deep Learning Transcriptomic Integration

A reproducible pipeline for identifying shared transcriptional signal between
two independently generated gene-expression datasets and projecting both
into a single shared latent space with a joint deep autoencoder.

The pipeline was developed for a case/control study design in which one
dataset is a microarray-based cohort (already normalized) and the second is
an RNA-seq-based cohort assembled from two separate sample sources (e.g. a
case cohort and a reference/control cohort profiled independently), but it
is written to generalize to any two comparably structured
expression datasets.

## What this repository contains

The analysis is organized as a sequence of standalone scripts, each
corresponding to one stage of the workflow. All of them read their
parameters from `config.py` and share the helper functions in `utils.py`.

| Script | Purpose |
|---|---|
| `differential_expression_overlap.py` | Intersects two differential-expression (DE) result tables and reports concordant/discordant shared gene sets. |
| `volcano_plot.py` | Draws a volcano plot per dataset with a candidate gene panel highlighted. |
| `differential_coexpression_heatmap.py` | Computes and plots the case-minus-control differential co-expression matrix for the candidate gene panel, per dataset. |
| `gene_signature_classification.py` | Fits a multi-gene logistic-regression classifier per dataset, reports ROC AUC and a Mann-Whitney significance test, and ranks genes by model coefficient. |
| `enrichment_term_deduplication.py` | Flags duplicated annotation terms across functional enrichment result exports (GO/KEGG/Reactome). |
| `joint_autoencoder_integration.py` | **Core analysis.** Trains a joint autoencoder on the combined, dataset-stratified-scaled expression matrix, cross-validates it, projects the learned latent space (UMAP/t-SNE), clusters it, and ranks genes by their contribution to the latent representation. |
| `summarize_latent_results.py` | Prints a compact summary of the joint autoencoder outputs (cluster composition, clustering quality, top contributing genes). |

Each script can be run independently, provided its expected input files (see
below) are present; a full run of the pipeline is simply each script run in
the order listed above.

## Workflow

1. **Differential expression overlap** — genes independently identified as
   differentially expressed in each dataset are intersected, both
   irrespective of direction and split into concordant/discordant
   direction-specific categories.
2. **Volcano plots** — the shared gene panel is visualized against each
   dataset's full differential-expression background.
3. **Differential co-expression** — pairwise correlation structure among the
   shared gene panel is compared between case and control samples, within
   each dataset.
4. **Gene-signature classification** — the discriminatory power of the
   shared gene panel is evaluated as a multi-gene classifier in each
   dataset independently (with RNA-seq batch correction where the dataset
   combines two sample sources).
5. **Enrichment term deduplication** — a lightweight quality-control pass
   over external functional enrichment exports.
6. **Joint autoencoder integration** — the two datasets are combined into a
   single matrix restricted to the shared gene panel, scaled
   dataset-wise, and passed through a symmetric dense autoencoder. The
   resulting latent space is validated by cross-validation, projected to
   2D, clustered, and interrogated for the genes driving its structure.
7. **Summary** — a plain-text and tabular summary of the joint latent space
   is produced for reporting.

## Requirements

- Python 3.10+
- See `requirements.txt` for package dependencies. Install with:

  ```bash
  pip install -r requirements.txt
  ```

`umap-learn` and `combat` are optional: the pipeline automatically falls
back to t-SNE (for latent-space projection) and per-batch quantile
normalization (for RNA-seq batch correction) respectively if they are not
installed, with a printed notice when the fallback is used.

## Expected input data layout

None of the input data, gene identifiers, or example files are included in
this repository. Place your own data under `data/` following this layout
(also documented at the top of `config.py`):

```
data/
    dataset_a/
        expression_matrix.csv      # probes/genes (rows) x samples (columns)
        sample_metadata.csv        # sample_id, annotation columns
    dataset_b/
        case_expression.csv        # RNA-seq quantification, case cohort
        control_expression.csv     # RNA-seq quantification, control cohort
    gene_panel.txt                 # one gene identifier per line
    probe_gene_mapping.csv         # columns: probe_id, gene_symbol
    deg_results/
        dataset_a_deg.csv          # gene-level DE table: gene, logFC, P.Value, ...
        dataset_a_deg_by_probe.csv # probe-level DE table for the volcano background: gene (probe id), logFC, P.Value
        dataset_b_deg.csv          # gene-level DE table, also used as the dataset B volcano background
    enrichment_results/
        go_bp_enrichment.csv       # column: Term (plus tool-specific columns)
        kegg_enrichment.csv
        reactome_enrichment.csv
```

Column-name conventions (e.g. `gene_id`, `gene_name`, `fpkm_unstranded`,
`case_id` for `dataset_b`, and the free-text `annotation` field parsed for
`dataset_a`) match commonly used RNA-seq quantification export formats and
sample-annotation metadata; adjust the column-name constants in `config.py`
if your own exports use different headers.

Every path and column-name convention referenced above is defined once in
`config.py` — update it there rather than editing the analysis scripts.

## Usage

Run each stage from the repository root:

```bash
python differential_expression_overlap.py
python volcano_plot.py
python differential_coexpression_heatmap.py
python gene_signature_classification.py
python enrichment_term_deduplication.py
python joint_autoencoder_integration.py
python summarize_latent_results.py
```

All outputs are written under `results/` (created automatically), with the
joint autoencoder outputs further grouped under `results/joint_autoencoder/`.

## Outputs

- Overlap tables, gene rankings, and cluster/validation tables as CSV files.
- Volcano, ROC, differential co-expression, and latent-space figures as PNG
  files.
- `results/joint_autoencoder/summary_report.txt` — a plain-text summary of
  the joint autoencoder run (sample/gene counts, cross-validation metrics,
  clustering quality, top contributing genes).

## Reproducibility

- A single random seed (`config.RANDOM_SEED`, default `42`) is applied to
  Python's `random`, NumPy, and TensorFlow at the start of every script via
  `utils.set_global_seed`.
- All model hyperparameters (autoencoder architecture, learning rate,
  dropout, regularization, cross-validation folds, UMAP parameters, k-means
  candidate range, classifier regularization) are centralized in
  `config.py` and were preserved unchanged from the original analysis.
- Expression preprocessing (log-transform, dataset-wise z-scoring,
  low-expression filtering thresholds) is likewise defined once in
  `config.py`.

## Customization

- To adapt the pipeline to a different pair of datasets, update the path
  and column-name constants in `config.py` and, if your case/control
  cohorts do not require batch correction (or use a different correction
  method), adjust `gene_signature_classification.apply_batch_correction`
  and `joint_autoencoder_integration.load_dataset_b` accordingly.
- To change the candidate gene panel, edit `data/gene_panel.txt` — no code
  changes are required.
- To change any threshold (differential-expression cutoffs, low-expression
  filtering, classifier regularization, autoencoder architecture, latent
  dimensionality, UMAP parameters, k-means candidate range), edit the
  corresponding constant in `config.py`.

## Known limitation

The candidate gene panel used for the classification and co-expression
analyses is supplied via probe-level identifiers when it must be resolved
to one of several candidate probes (microarray dataset). The general rule
implemented here — selecting the probe with the largest absolute log2 fold
change — reproduces the original selection rule for genes mapped to a
single ambiguous set of probes. It does not special-case genes whose probe
annotation is shared with a paralogous gene (an edge case affecting a
small number of gene pairs in the original analysis); such cases should be
resolved manually in `data/probe_gene_mapping.csv` if they arise in a new
dataset.

## Batch-effect correction: scope and caveat

Two different, deliberately different strategies are used for combining
data from separate sources, and neither should be swapped for the other
without re-checking the reasoning below:

- `gene_signature_classification.apply_batch_correction` applies ComBat
  (or a per-batch quantile-normalization fallback) to merge dataset B's
  case and control cohorts, which are RNA-seq quantifications assembled
  from two independently processed sources. This is standard practice for
  combining same-platform RNA-seq cohorts collected/processed separately.
- `joint_autoencoder_integration.prepare_joint_dataset` instead uses
  simple per-dataset z-score standardization (no ComBat) when merging the
  microarray and RNA-seq datasets into the joint autoencoder input, since
  ComBat-style correction is designed for same-platform data and is not
  an appropriate way to reconcile two different expression-profiling
  platforms.

**Caveat for `apply_batch_correction`:** in the case/control cohort layout
this pipeline was built around, the "batch" variable (data source) and the
biological variable of interest (case vs. control status) are completely
confounded — every control sample comes from one source and every case
sample from the other. Under that layout, ComBat (or any batch-correction
method) cannot fully distinguish genuine technical batch effects from
genuine case-vs-control biological differences, since it has no samples
where the two vary independently. This is a well-known limitation of
batch correction in this kind of two-cohort design, not a bug in the
implementation, and it is not unique to this pipeline (it applies equally
to the original analysis this code was derived from). It is surfaced here
so that anyone re-running or extending this pipeline is aware of it: at a
minimum, results should be interpreted alongside the uncorrected
comparison, and if your own data lets case and control samples come from
a mix of batches, that is the more statistically identifiable design and
is preferable going forward.
No change was made to whether or how batch correction is applied, since
removing it would leave the two cohorts on incompatible scales and adding
a different scheme is outside the scope of this cleanup — this section
documents the existing, carried-over design choice rather than altering it.
