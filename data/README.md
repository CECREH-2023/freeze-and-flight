# Data sources and availability

Nebraska deed records; county assessor/parcel records; inherited inundation and FEMA SFHA geometries; BLS CPI.

Only aggregate results and figures are released. Deeds, grantee names, parcel records, and restricted source-review records remain local.

## External inputs for the entry point

Paths are relative to the repository root. They describe files or directories to supply; they are not bundled download links. Upstream acquisition and optional analyses may require additional inputs.

| Path | Availability |
|---|---|
| `data_work/transactions_raw.parquet` | external; not bundled |
| `data_work/transactions_linked.parquet` | external; not bundled |
| `data_work/parcel_boundary_distances.parquet` | external; not bundled |
| `data_work/parcel_centroids.gpkg` | external; not bundled |
| `data_work/assessor_raw.parquet` | external; not bundled |
| `data_work/panel_parcel_month.parquet` | external; not bundled |
| `data_work/restricted/grantee_classification/grantee_model_labels.parquet` | external; not bundled |
| `data/external/ownership_model/data/processed/standardized_training_dataset_final.csv` | external; not bundled |
| `data/external/ownership_model/results/production_classification_results/owner_type_mapping_20250802_192450.csv` | external; not bundled |

## File definitions and provenance

- [Table inventory](TABLES.csv): distributed CSV columns and row counts.
- [Input inventory](INPUTS.json): paths checked by the input preflight.
- [Source fingerprints](../SOURCE_FILES.csv): hashes of retained source files and their public versions.
