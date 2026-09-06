# Reproduction guide

Scientific unit checks verified; deed-level data and ownership-model inputs remain external.

## Verify the distribution

From the package root:

```bash
python scripts/check_package.py
python scripts/check_inputs.py
```

The first command verifies the shipped files and hashes. The second checks whether separately acquired inputs are present and exits with code 2 when they are missing. Neither command estimates a statistical model.

## Run the selected workflow

All ten focused scientific tests pass. Full estimation needs the records listed in data/INPUTS.json, including the classifier reference corpus and stored production predictions. Serialized original BERT weights are not locally available; the code documents stored-prediction matching and the distilled classifier. No new classification or estimation was run in this package.

Use a disposable working copy when running the original analysis: several original scripts overwrite their project-relative output locations. Keep the distributed reference snapshot for comparison.

Environment: `python -m pip install -r requirements.txt`

```bash
python src/07_estimation/flood_market_composition.py --boot 9999
```

## Input contract

Required paths are listed in [data/INPUTS.json](../data/INPUTS.json). Only aggregate results and figures are released. Deeds, grantee names, parcel records, and restricted source-review records remain local.

[Source guide](CODE_MAP.md) identifies additional acquisition, sensitivity, and rendering modules. Original modeling and uncertainty procedures are retained. Use the documented input definitions; undocumented data substitutions can change the analysis.

## Evidence

[VALIDATION.json](../VALIDATION.json) records the checks performed on this snapshot. A partial model run or a fictional demo is identified by its limited scope. Full reproduction is claimed only where that record explicitly supports it.
