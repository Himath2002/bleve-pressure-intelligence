# Data boundary

The modelling notebook expects two local files in this directory:

- `train.csv` - scenario inputs plus `Target Pressure (bar)`.
- `test.csv` - scenario inputs without the target column.

The original benchmark data is not included because no redistribution terms
were supplied with it. The repository therefore publishes the exact schema,
data-quality assumptions, and a deterministic synthetic-data generator while
keeping the source dataset outside version control.

Generate a non-authoritative demo dataset from the repository root:

```bash
python scripts/generate_demo_data.py --output-dir data/demo
```

Then point the notebook at it before starting Jupyter:

```bash
export BLEVE_DATA_DIR=data/demo
jupyter lab notebooks/bleve-pressure-modeling.ipynb
```

Synthetic values are designed only to exercise the engineering workflow. They
do not reproduce the benchmark distribution and must not be treated as
physical measurements or safety evidence.

See [`schema.csv`](schema.csv) for field names, types, units, and modelling
roles.
