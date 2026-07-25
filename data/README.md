# Data

Raw, interim, and processed datasets are not committed to GitHub. The Criteo Uplift Prediction Dataset is the working dataset for Phase 1.

Place the downloaded Criteo file at:

```text
data/raw/criteo/criteo-research-uplift-v2.1.csv.gz
```

The local path above is relative to the repository root. Source code and notebooks must not depend on absolute local paths or Kaggle-specific absolute paths.

For local development, use the loader's `nrows` argument to inspect a subset without loading the full dataset. The full dataset can be processed in Kaggle by using the same repository code and overriding the data path in the notebook when needed.

RetailHero raw files will later live under:

```text
data/raw/retailhero/
```

RetailHero should reuse the shared framework, not a model trained on Criteo.

Confirmed Criteo header from the local file:

```text
f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,treatment,conversion,visit,exposure
```

This matches the expected column set. The confirmed file order places `conversion` and `visit` before `exposure`; validation code checks by column name, not by column position.
