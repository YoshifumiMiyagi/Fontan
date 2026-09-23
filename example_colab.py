from src.fontan_manifold import (
    FontanManifold,
    FontanOutcomeAnalyzer,
    default_sex_binary,
)

# raw = your original dataframe

df7 = FontanManifold.prepare_dataframe(raw)

model = FontanManifold(
    n_neighbors=50,
    n_dm=5,
    id_col="subj_id",
)

model.fit(df7)

print("Full cohort N =", len(df7))
print("Reference N =", len(model.reference_df_))
print(model.self_validate(n_dm=3))

projection_long, mi_diagnostics = model.project_multiple_imputation(
    df7,
    n_imputations=20,
    seeds=list(range(1001, 1021)),
    n_dm=3,
)

projection_summary = model.summarize_mi(
    projection_long,
    n_dm=3,
)

clinical = df7.copy()
clinical["sex_binary"] = default_sex_binary(clinical["sex"])

analyzer = FontanOutcomeAnalyzer(id_col="subj_id")

individual = analyzer.fit_univariable_dm_models(
    projection_long,
    clinical,
    dms=("DM1", "DM2", "DM3"),
    covariates=("fontan_year", "sex_binary"),
)

pooled = analyzer.pool(individual, fdr=True)

print(
    pooled[
        ["DM", "Outcome", "N", "Events", "OR_95CI", "P_value", "q_value", "FMI"]
    ]
)
