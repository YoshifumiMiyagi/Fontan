"""
Reusable 7-domain Fontan diffusion manifold.

Core workflow
-------------
1. Fit a complete-case reference manifold.
2. Freeze all reference transformations and graph parameters.
3. Project new / imputed subjects by Nyström extension without refitting.
4. Fit logistic outcome models within each imputation.
5. Pool coefficients with Rubin's rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, List, Tuple

import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.stats import norm, spearmanr, t as student_t

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm


DEFAULT_DOMAIN_DEFINITION = {
    "Exercise": {"type": "continuous", "vars": ["pp_peakvo2", "pp_vat"]},
    "Remodeling": {"type": "continuous", "vars": ["echoedv", "echomass"]},
    "Function": {"type": "continuous", "vars": ["echoef", "tei_index"]},
    "Diastolic": {"type": "continuous", "vars": ["e_a", "e_tde"]},
    "Valve": {"type": "ordinal", "vars": ["oavvregurg_ord"]},
    "Anatomy_Surgery": {
        "type": "categorical",
        "vars": [
            "Ventricular dominance",
            "preop_dxcat",
            "fontan_sergery_cat",
            "pulmonary_caval_anastomosis",
            "fenestration_performed",
        ],
    },
    "Biomarker": {"type": "continuous", "vars": ["log_BNP"]},
}

DEFAULT_CONTINUOUS_MI_VARS = [
    "pp_peakvo2",
    "pp_vat",
    "echoedv",
    "echomass",
    "echoef",
    "tei_index",
    "e_a",
    "e_tde",
    "log_BNP",
]

DEFAULT_OUTCOMES = [
    "AT_BIN",
    "VT_BIN",
    "PACEMAKER_BIN",
    "THROMBOS_BIN",
    "STROKE_BIN",
    "PLE_BIN",
]


@dataclass
class ProjectionResult:
    coordinates: pd.DataFrame
    sigma_new: np.ndarray


class FontanManifold:
    def __init__(
        self,
        domain_definition: Optional[Mapping] = None,
        n_neighbors: int = 50,
        n_dm: int = 5,
        id_col: str = "subj_id",
        eps: float = 1e-12,
    ):
        self.domain_definition = dict(
            DEFAULT_DOMAIN_DEFINITION if domain_definition is None else domain_definition
        )
        self.n_neighbors = int(n_neighbors)
        self.n_dm = int(n_dm)
        self.id_col = id_col
        self.eps = float(eps)
        self.is_fitted_ = False

    @staticmethod
    def prepare_dataframe(
        df: pd.DataFrame,
        bnp_col: str = "BNP",
        valve_col: str = "oavvregurg",
    ) -> pd.DataFrame:
        out = df.copy()

        if "log_BNP" not in out.columns:
            if bnp_col not in out.columns:
                raise KeyError(f"Neither 'log_BNP' nor '{bnp_col}' is present.")
            out["log_BNP"] = np.log1p(pd.to_numeric(out[bnp_col], errors="coerce"))

        if "oavvregurg_ord" not in out.columns:
            if valve_col not in out.columns:
                raise KeyError(
                    f"Neither 'oavvregurg_ord' nor '{valve_col}' is present."
                )
            out["oavvregurg_ord"] = out[valve_col].map(
                {
                    "0:NONE": 0,
                    "1:MILD": 1,
                    "2:MODERATE": 2,
                    "3:SEVERE": 3,
                }
            )

        return out

    @property
    def required_vars(self) -> List[str]:
        cols = []
        for spec in self.domain_definition.values():
            cols.extend(spec["vars"])
        return list(dict.fromkeys(cols))

    def make_complete_case_reference(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare_dataframe(df)
        ref = df.dropna(subset=self.required_vars).copy().reset_index(drop=True)
        if self.id_col not in ref.columns:
            raise KeyError(f"Missing ID column: {self.id_col}")
        if not ref[self.id_col].is_unique:
            raise ValueError(f"{self.id_col} must be unique in the reference cohort.")
        return ref

    def _normalize_distance_matrix(self, D: np.ndarray) -> Tuple[np.ndarray, float]:
        D = np.asarray(D, dtype=float)
        tri = D[np.triu_indices_from(D, k=1)]
        positive = tri[np.isfinite(tri) & (tri > 0)]
        if len(positive) == 0:
            raise ValueError("No positive pairwise distances found.")
        scale = float(np.median(positive))
        return D / scale, scale

    def _fit_domain(self, df: pd.DataFrame, spec: Mapping):
        vars_ = list(spec["vars"])
        dtype = spec["type"]

        if dtype == "continuous":
            scaler = StandardScaler()
            X_ref = scaler.fit_transform(df[vars_].astype(float))
            D_raw = pairwise_distances(X_ref, metric="euclidean") / np.sqrt(len(vars_))
            D, scale = self._normalize_distance_matrix(D_raw)
            params = {
                "type": dtype,
                "vars": vars_,
                "scaler": scaler,
                "X_ref": X_ref,
                "distance_scale": scale,
            }

        elif dtype == "ordinal":
            X_raw = df[vars_].astype(float).to_numpy()
            mins = np.nanmin(X_raw, axis=0)
            maxs = np.nanmax(X_raw, axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            X_ref = (X_raw - mins) / ranges
            D_raw = pairwise_distances(X_ref, metric="manhattan") / len(vars_)
            D, scale = self._normalize_distance_matrix(D_raw)
            params = {
                "type": dtype,
                "vars": vars_,
                "min": mins,
                "max": maxs,
                "range": ranges,
                "X_ref": X_ref,
                "distance_scale": scale,
            }

        elif dtype == "categorical":
            X_ref = df[vars_].astype(str).to_numpy()
            n = len(X_ref)
            D_raw = np.zeros((n, n), dtype=float)
            for j in range(len(vars_)):
                col = X_ref[:, j]
                D_raw += (col[:, None] != col[None, :]).astype(float)
            D_raw /= len(vars_)
            D, scale = self._normalize_distance_matrix(D_raw)
            params = {
                "type": dtype,
                "vars": vars_,
                "X_ref": X_ref,
                "distance_scale": scale,
            }
        else:
            raise ValueError(f"Unknown domain type: {dtype}")

        return D, params

    def _adaptive_affinity(self, D: np.ndarray):
        n = D.shape[0]
        k = min(self.n_neighbors, n - 1)
        order = np.argsort(D, axis=1)
        knn_idx = order[:, 1 : k + 1]

        sigma = np.array([np.median(D[i, knn_idx[i]]) for i in range(n)])
        positive = sigma[sigma > self.eps]
        if len(positive) == 0:
            raise ValueError("All local sigma values are zero.")
        sigma[sigma <= self.eps] = np.median(positive)

        W = np.zeros((n, n), dtype=float)
        for i in range(n):
            js = knn_idx[i]
            denom = sigma[i] * sigma[js]
            W[i, js] = np.exp(-(D[i, js] ** 2) / (denom + self.eps))

        W = np.maximum(W, W.T)
        np.fill_diagonal(W, 0.0)
        return csr_matrix(W), sigma

    def _diffusion_map(self, W: csr_matrix):
        degree = np.asarray(W.sum(axis=1)).ravel()
        degree = np.maximum(degree, self.eps)
        d_inv_sqrt = 1.0 / np.sqrt(degree)
        S = diags(d_inv_sqrt) @ W @ diags(d_inv_sqrt)

        n_components = min(self.n_dm + 1, W.shape[0] - 1)
        eigvals, eigvecs = eigsh(S, k=n_components, which="LM")
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        psi = d_inv_sqrt[:, None] * eigvecs
        return psi, eigvals, degree

    def fit(self, df_reference: pd.DataFrame, complete_case: bool = True):
        df = self.prepare_dataframe(df_reference)
        ref = self.make_complete_case_reference(df) if complete_case else df.copy()

        if ref[self.required_vars].isna().any().any():
            raise ValueError("Reference data contain missing domain variables.")

        self.reference_df_ = ref.reset_index(drop=True)
        self.domain_params_ = {}
        self.domain_distances_ = {}

        for name, spec in self.domain_definition.items():
            D, params = self._fit_domain(self.reference_df_, spec)
            self.domain_distances_[name] = D
            self.domain_params_[name] = params

        self.D_ref_ = np.mean(
            np.stack([self.domain_distances_[name] for name in self.domain_definition], axis=0),
            axis=0,
        )

        self.W_ref_, self.sigma_ref_ = self._adaptive_affinity(self.D_ref_)
        self.psi_ref_, self.eigvals_ref_, self.degree_ref_ = self._diffusion_map(self.W_ref_)
        self.n_dm_effective_ = min(self.n_dm, self.psi_ref_.shape[1] - 1)
        self.is_fitted_ = True
        return self

    def reference_coordinates(self, n_dm: Optional[int] = None) -> pd.DataFrame:
        self._check_fitted()
        n_dm = self.n_dm_effective_ if n_dm is None else min(n_dm, self.n_dm_effective_)
        out = pd.DataFrame({self.id_col: self.reference_df_[self.id_col].to_numpy()})
        for j in range(1, n_dm + 1):
            out[f"DM{j}"] = self.psi_ref_[:, j]
        return out

    def _cross_domain_distance(self, df_new: pd.DataFrame, domain_name: str):
        p = self.domain_params_[domain_name]
        vars_ = p["vars"]
        dtype = p["type"]

        if dtype == "continuous":
            X_new = p["scaler"].transform(df_new[vars_].astype(float))
            D = cdist(X_new, p["X_ref"], metric="euclidean") / np.sqrt(len(vars_))

        elif dtype == "ordinal":
            X_new = df_new[vars_].astype(float).to_numpy()
            X_new = (X_new - p["min"]) / p["range"]
            D = cdist(X_new, p["X_ref"], metric="cityblock") / len(vars_)

        elif dtype == "categorical":
            X_new = df_new[vars_].astype(str).to_numpy()
            X_ref = p["X_ref"]
            D = np.zeros((len(X_new), len(X_ref)), dtype=float)
            for i in range(len(X_new)):
                D[i, :] = (X_new[i, :] != X_ref).mean(axis=1)
        else:
            raise ValueError(dtype)

        return D / p["distance_scale"]

    def integrated_cross_distance(self, df_new: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if df_new[self.required_vars].isna().any().any():
            raise ValueError("Missing domain values remain in projection data.")

        distances = [
            self._cross_domain_distance(df_new, name)
            for name in self.domain_definition
        ]
        return np.mean(np.stack(distances, axis=0), axis=0)

    def _nystrom_project(self, D_cross: np.ndarray, n_dm: int):
        n_new = D_cross.shape[0]
        k = min(self.n_neighbors, D_cross.shape[1])

        sigma_new = np.zeros(n_new)
        for i in range(n_new):
            nearest = np.partition(D_cross[i], k - 1)[:k]
            sigma_new[i] = np.median(nearest)
        sigma_new[sigma_new < self.eps] = self.eps

        denominator = sigma_new[:, None] * self.sigma_ref_[None, :]
        W_cross = np.exp(-(D_cross ** 2) / (denominator + self.eps))

        degree_new = W_cross.sum(axis=1)
        degree_new[degree_new < self.eps] = self.eps

        S_cross = W_cross / np.sqrt(
            degree_new[:, None] * self.degree_ref_[None, :]
        )

        projected = np.zeros((n_new, n_dm), dtype=float)
        for dm in range(1, n_dm + 1):
            lam = self.eigvals_ref_[dm]
            projected[:, dm - 1] = (S_cross @ self.psi_ref_[:, dm]) / lam

        return projected, sigma_new

    def transform(self, df_new: pd.DataFrame, n_dm: int = 3) -> ProjectionResult:
        self._check_fitted()
        df_new = self.prepare_dataframe(df_new)
        n_dm = min(n_dm, self.n_dm_effective_)
        D_cross = self.integrated_cross_distance(df_new)
        projected, sigma_new = self._nystrom_project(D_cross, n_dm=n_dm)

        coords = pd.DataFrame({self.id_col: df_new[self.id_col].to_numpy()})
        for j in range(n_dm):
            coords[f"DM{j + 1}"] = projected[:, j]
        coords["sigma_new"] = sigma_new
        return ProjectionResult(coords, sigma_new)

    def self_validate(self, n_dm: int = 3) -> pd.DataFrame:
        projected = self.transform(self.reference_df_, n_dm=n_dm).coordinates
        rows = []
        for j in range(1, min(n_dm, self.n_dm_effective_) + 1):
            rho, p = spearmanr(self.psi_ref_[:, j], projected[f"DM{j}"])
            rows.append({"DM": f"DM{j}", "Spearman_rho": rho, "P_value": p})
        return pd.DataFrame(rows)

    @staticmethod
    def _impute_valve_stochastic(
        data: pd.DataFrame,
        rng: np.random.Generator,
        col: str = "oavvregurg_ord",
    ) -> pd.DataFrame:
        out = data.copy()
        observed = pd.to_numeric(out[col], errors="coerce").dropna()
        probs = observed.value_counts(normalize=True).sort_index()
        missing_idx = out.index[out[col].isna()]
        if len(missing_idx):
            draws = rng.choice(
                probs.index.to_numpy(),
                size=len(missing_idx),
                replace=True,
                p=probs.to_numpy(),
            )
            out.loc[missing_idx, col] = draws
        return out

    def project_multiple_imputation(
        self,
        df: pd.DataFrame,
        n_imputations: int = 20,
        seeds: Optional[Sequence[int]] = None,
        continuous_vars: Optional[Sequence[str]] = None,
        n_dm: int = 3,
        max_iter: int = 20,
    ):
        self._check_fitted()
        data = self.prepare_dataframe(df)
        continuous_vars = list(
            DEFAULT_CONTINUOUS_MI_VARS if continuous_vars is None else continuous_vars
        )
        if seeds is None:
            seeds = list(range(1001, 1001 + n_imputations))

        results, diagnostics = [], []

        for m, seed in enumerate(seeds, start=1):
            df_mi = data.copy()

            imputer = IterativeImputer(
                random_state=int(seed),
                sample_posterior=True,
                max_iter=max_iter,
                initial_strategy="median",
                skip_complete=True,
            )
            df_mi.loc[:, continuous_vars] = imputer.fit_transform(df_mi[continuous_vars])

            rng = np.random.default_rng(int(seed))
            df_mi = self._impute_valve_stochastic(df_mi, rng)

            projection = self.transform(df_mi, n_dm=n_dm).coordinates
            projection.insert(1, "MI", m)
            results.append(projection)

            row = {"MI": m, "seed": int(seed)}
            for j in range(1, n_dm + 1):
                row[f"DM{j}_mean"] = projection[f"DM{j}"].mean()
                row[f"DM{j}_sd"] = projection[f"DM{j}"].std(ddof=0)
            row["sigma_mean"] = projection["sigma_new"].mean()
            diagnostics.append(row)

        return pd.concat(results, ignore_index=True), pd.DataFrame(diagnostics)

    def summarize_mi(self, projection_long: pd.DataFrame, n_dm: int = 3):
        agg = {}
        for j in range(1, n_dm + 1):
            agg[f"DM{j}_mean"] = (f"DM{j}", "mean")
            agg[f"DM{j}_sd"] = (f"DM{j}", "std")
        if "sigma_new" in projection_long:
            agg["sigma_mean"] = ("sigma_new", "mean")
            agg["sigma_sd"] = ("sigma_new", "std")
        return projection_long.groupby(self.id_col).agg(**agg).reset_index()

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before projection.")


class FontanOutcomeAnalyzer:
    def __init__(self, id_col: str = "subj_id", outcomes: Optional[Sequence[str]] = None):
        self.id_col = id_col
        self.outcomes = list(DEFAULT_OUTCOMES if outcomes is None else outcomes)

    @staticmethod
    def rubin_pool(group: pd.DataFrame) -> pd.Series:
        g = group[np.isfinite(group["Beta"]) & np.isfinite(group["Variance"])].copy()
        m = len(g)
        if m < 2:
            return pd.Series({"M": m, "Beta": np.nan, "SE": np.nan, "OR": np.nan,
                              "CI_low": np.nan, "CI_high": np.nan, "P_value": np.nan,
                              "Within_var": np.nan, "Between_var": np.nan, "FMI": np.nan})

        Q = g["Beta"].to_numpy()
        U = g["Variance"].to_numpy()
        Q_bar = np.mean(Q)
        U_bar = np.mean(U)
        B = np.var(Q, ddof=1)
        T = U_bar + (1 + 1 / m) * B
        pooled_se = np.sqrt(T)

        if B <= 1e-15 or U_bar <= 0:
            df_rubin = np.inf
        else:
            r = ((1 + 1 / m) * B) / U_bar
            df_rubin = (m - 1) * (1 + 1 / r) ** 2

        stat = Q_bar / pooled_se
        if np.isfinite(df_rubin):
            p = 2 * student_t.sf(abs(stat), df=df_rubin)
            crit = student_t.ppf(0.975, df=df_rubin)
        else:
            p = 2 * norm.sf(abs(stat))
            crit = 1.96

        fmi = ((1 + 1 / m) * B) / T if T > 0 else np.nan

        return pd.Series({
            "M": m,
            "Beta": Q_bar,
            "SE": pooled_se,
            "OR": np.exp(Q_bar),
            "CI_low": np.exp(Q_bar - crit * pooled_se),
            "CI_high": np.exp(Q_bar + crit * pooled_se),
            "P_value": p,
            "Within_var": U_bar,
            "Between_var": B,
            "FMI": fmi,
        })

    def fit_univariable_dm_models(
        self,
        projection_long: pd.DataFrame,
        clinical_df: pd.DataFrame,
        dms: Sequence[str] = ("DM1", "DM2", "DM3"),
        covariates: Sequence[str] = ("fontan_year", "sex_binary"),
    ) -> pd.DataFrame:
        rows = []
        clinical_cols = list(dict.fromkeys([self.id_col, *self.outcomes, *covariates]))

        for mi in sorted(projection_long["MI"].unique()):
            proj = projection_long[projection_long["MI"] == mi].copy()
            for dm in dms:
                proj[f"{dm}_z"] = (proj[dm] - proj[dm].mean()) / proj[dm].std(ddof=1)

            dat = proj.merge(clinical_df[clinical_cols], on=self.id_col, how="left")

            for outcome in self.outcomes:
                for dm in dms:
                    cols = [outcome, f"{dm}_z", *covariates]
                    d = dat[cols].apply(pd.to_numeric, errors="coerce").dropna()
                    X = sm.add_constant(d[[f"{dm}_z", *covariates]], has_constant="add")
                    y = d[outcome]

                    try:
                        fit = sm.Logit(y, X).fit(disp=False, maxiter=200)
                        beta = fit.params[f"{dm}_z"]
                        se = fit.bse[f"{dm}_z"]
                        converged = bool(fit.mle_retvals.get("converged", True))
                    except Exception:
                        beta = se = np.nan
                        converged = False

                    rows.append({
                        "MI": mi,
                        "Outcome": outcome,
                        "DM": dm,
                        "N": len(d),
                        "Events": int(d[outcome].sum()) if len(d) else 0,
                        "Beta": beta,
                        "SE": se,
                        "Variance": se**2 if np.isfinite(se) else np.nan,
                        "Converged": converged,
                    })

        return pd.DataFrame(rows)

    def pool(self, mi_model_results: pd.DataFrame, fdr: bool = True) -> pd.DataFrame:
        pooled_rows = []
        for (outcome, dm), g in mi_model_results.groupby(["Outcome", "DM"]):
            pooled = self.rubin_pool(g).to_dict()
            pooled["Outcome"] = outcome
            pooled["DM"] = dm
            pooled["N"] = g["N"].median()
            pooled["Events"] = g["Events"].median()
            pooled_rows.append(pooled)

        out = pd.DataFrame(pooled_rows)

        if fdr and len(out):
            valid = out["P_value"].notna()
            out["q_value"] = np.nan
            if valid.any():
                out.loc[valid, "q_value"] = multipletests(
                    out.loc[valid, "P_value"],
                    method="fdr_bh",
                )[1]

        out["OR_95CI"] = out.apply(
            lambda r: (
                f"{r['OR']:.2f} ({r['CI_low']:.2f}–{r['CI_high']:.2f})"
                if np.isfinite(r["OR"]) else np.nan
            ),
            axis=1,
        )
        return out.sort_values(["DM", "P_value"], na_position="last").reset_index(drop=True)


def default_sex_binary(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .map({"1:MALE": 1, "2:FEMALE": 0})
    )
