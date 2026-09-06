# Source guide

Paths below refer to the distributed source. Analysis modules may require external inputs; inclusion does not imply the entire source workflow has been executed.

| Module | Purpose |
|---|---|
| [src/02_labels/deed_grantee_classification.py](../src/02_labels/deed_grantee_classification.py) | Classify deed grantee legal form using the published ownership model. |
| [src/07_estimation/flood_market_composition.py](../src/07_estimation/flood_market_composition.py) | Estimate descriptive changes in deed composition and prices after the 2019 Nebraska flood. |
| [src/08_figures/flood_market_manuscript_inputs.py](../src/08_figures/flood_market_manuscript_inputs.py) | Export validated aggregate composition evidence and three main figures. |
| [src/transaction_mode.py](../src/transaction_mode.py) | Transaction-mode provenance and validation helpers. |
| [src/utils/__init__.py](../src/utils/__init__.py) | Utility modules for Freeze and Flight analysis. |
| [src/utils/figure_style.py](../src/utils/figure_style.py) | Figure styling configuration for Freeze and Flight manuscript. |
| [tests/test_flood_market_composition.py](../tests/test_flood_market_composition.py) | Scientific invariants for the expanded-footprint analysis. |
