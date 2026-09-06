# Residential Transactions After the 2019 Nebraska Flood

How did recorded residential transaction composition and prices change within the inherited flood footprint?

Descriptive deed-record analysis for 2015–2022. Improvement-value categories are not independently verified building conditions; inherited inundation metadata and later parcel linkage limit interpretation.

## Results and interpretation

Inside the inherited 2019 flood footprint, the share of retained deeds with zero recorded improvement value rises from **30.5% before the flood to 56.4% afterward**. The strongest descriptive shift occurs after March 2020.

The adjusted relative change is **+8.7 percentage points (95% CI −6.6 to +23.9)**. Adjusted pooled residential prices change by **+6.9% (−15.8% to +35.7%)**, and prices among positive-improvement deeds by **+3.4% (−19.4% to +32.6%)**. These wide intervals and timing sensitivities do not establish an immediate flood-induced change. A recorded zero-improvement value is not independent evidence of demolition or vacancy. See the [primary results](manuscript_quarto/data/flood_market_main_table.csv).

![Residential Transactions After the 2019 Nebraska Flood: reference figure](manuscript_quarto/figures/fig_flood_market_annual.png)

## Explore this repository

- [Research figures](manuscript_quarto/figures/)
- [Methods](docs/METHODS.md)
- [Reproduction and dependencies](docs/REPRODUCING.md)
- [Analysis source guide](docs/CODE_MAP.md)
- [Data sources and availability](data/README.md)

## Reproduce the work

**Available reproduction:** Scientific unit checks verified; deed-level data and ownership-model inputs remain external.

Start with `python scripts/check_package.py` to check the file manifest, then follow the [reproduction guide](docs/REPRODUCING.md). A file-integrity check does not rerun the research analysis. Only aggregate results and figures are released. Deeds, grantee names, parcel records, and restricted source-review records remain local.

## Attribution and use

The associated study is “Residential Transaction Composition and Prices After the 2019 Nebraska Flood.” The selected analysis covers 2015–2022.

A research resource from [CECREH at Texas Tech University](https://www.depts.ttu.edu/cecreh/). Snapshot: September 6, 2026. For code citation, use the repository URL and the commit identifier for the version you used; see [citation guidance](CITATION.md).

No additional reuse license is granted by this snapshot. Contact the authors through CECREH about permissions; source-data terms apply separately.
