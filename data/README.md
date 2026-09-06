# Data sources and availability

Nebraska deed records; county assessor/parcel records; inherited inundation and FEMA SFHA geometries; BLS CPI.

Only aggregate results and figures are released. Deeds, grantee names, parcel records, and restricted source-review records remain local.

All ten focused scientific tests pass. Full estimation needs the records listed in data/INPUTS.json, including the classifier reference corpus and stored production predictions. Serialized original BERT weights are not locally available; the code documents stored-prediction matching and the distilled classifier. No new classification or estimation was run in this package.

## File-level records

- [Required inputs](INPUTS.json) describes separately acquired files.
- [Released table inventory](TABLES.csv) lists distributed table columns and row counts.
- [Source fingerprints](../SOURCE_FILES.csv) links retained source content to its hashes without publishing private workspace paths.

Raw records, access credentials, personal notes, correspondence, and publisher full-text collections are not distributed.
