# Reproduction guide

Scientific unit checks verified; deed-level data and ownership-model inputs remain external.

## Software

Run from the repository root in a separate Python environment. The documented installation profile is:

```bash
python -m pip install -r requirements.txt
```

See [software requirements](SOFTWARE.md) for optional stages and environment limits.

## Run the workflow

All ten focused scientific tests pass. Full estimation needs the records listed in data/INPUTS.json, including the classifier reference corpus and stored production predictions. Serialized original BERT weights are not locally available; the code documents stored-prediction matching and the distilled classifier. No new classification or estimation was run in this package.

```bash
python src/07_estimation/flood_market_composition.py --boot 9999
```

Research scripts may overwrite their project-relative output files. Run analyses in a working copy and retain the checked-in reference results for comparison.

The focused model tests do not require the full estimation inputs:

```bash
python -m unittest discover -s tests -p test_flood_market_composition.py
```

## Inputs

The exact external paths are listed in [data/README.md](../data/README.md) and [INPUTS.json](../data/INPUTS.json). `python scripts/check_inputs.py` checks their presence and exits with code 2 if any listed path is missing. It does not check the schema, verify access rights, or acquire upstream data.

## Verification scope

`python scripts/check_package.py` checks the distributed file hashes. It uses only the Python standard library and does not fit a model. Run it before generating outputs; new files outside the designated generated-results directory may be reported as extras.

[VALIDATION.json](../VALIDATION.json) records the checks performed for this version and their limits. Inclusion of an analysis module is not evidence that it has been executed. The [source guide](CODE_MAP.md) identifies the distributed modules.
