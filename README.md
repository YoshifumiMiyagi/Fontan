# Fontan 7-domain diffusion manifold

Reusable implementation of a 7-domain Fontan diffusion manifold with fixed-reference Nyström projection, multiple imputation, and Rubin-pooled outcome analysis.

## Structure

- `src/fontan_manifold.py`: reusable analysis classes
- `example_colab.py`: minimal usage example for Google Colab
- `requirements.txt`: Python dependencies
- `.gitignore`: excludes patient-level data and local artifacts

## Important

Do **not** commit PHN/raw patient-level data or derived files containing subject identifiers to this public repository.

## Minimal usage

```python
from src.fontan_manifold import FontanManifold, FontanOutcomeAnalyzer

df7 = FontanManifold.prepare_dataframe(raw)

model = FontanManifold(
    n_neighbors=50,
    n_dm=5,
    id_col="subj_id",
)

model.fit(df7)
print(model.self_validate(n_dm=3))

projection_long, diagnostics = model.project_multiple_imputation(
    df7,
    n_imputations=20,
    seeds=range(1001, 1021),
    n_dm=3,
)

summary = model.summarize_mi(projection_long, n_dm=3)
```
