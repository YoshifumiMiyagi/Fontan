from __future__ import annotations
from typing import Sequence, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_manifold_3d(df: pd.DataFrame, color: str = "pseudo_time",
                     dms: Sequence[str] = ("DM1","DM2","DM3"),
                     title: str = "Fontan diffusion manifold"):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(df[dms[0]], df[dms[1]], df[dms[2]],
                    c=df[color], s=18, alpha=.8)
    ax.set_xlabel(dms[0]); ax.set_ylabel(dms[1]); ax.set_zlabel(dms[2])
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, pad=.1, label=color)
    fig.tight_layout()
    return fig, ax


def plot_component_characterization(characterization: pd.DataFrame,
                                    top_n: int = 6):
    figs = {}
    for dm, g in characterization.groupby("DM"):
        x = g.nlargest(top_n, "abs_rho").sort_values("rho")
        fig, ax = plt.subplots(figsize=(7, max(3.5, .5*len(x)+1)))
        ax.barh(x["Variable"], x["rho"])
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Spearman rho")
        ax.set_title(f"{dm}: clinical characterization")
        for i, (_, r) in enumerate(x.iterrows()):
            q = r.get("q_value", np.nan)
            label = f"q={q:.3g}" if np.isfinite(q) else ""
            ax.text(r["rho"], i, "  "+label, va="center",
                    ha="left" if r["rho"] >= 0 else "right")
        fig.tight_layout()
        figs[dm] = fig
    return figs


def plot_pt_scatter(df: pd.DataFrame, y: str, x: str = "PT_mean",
                    ylabel: Optional[str] = None):
    d = df[[x,y]].apply(pd.to_numeric, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(d[x], d[y], s=20, alpha=.65)
    if len(d) >= 2:
        b1, b0 = np.polyfit(d[x], d[y], 1)
        xx = np.linspace(d[x].min(), d[x].max(), 100)
        ax.plot(xx, b1*xx+b0, linewidth=2)
    ax.set_xlabel("Pseudo-time")
    ax.set_ylabel(ylabel or y)
    ax.set_title(f"Pseudo-time vs {ylabel or y}")
    fig.tight_layout()
    return fig, ax


def plot_outcome_forest(results: pd.DataFrame, title: str = "Complication associations"):
    d = results.copy().reset_index(drop=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7, max(4, .65*len(d)+1.5)))
    err = np.vstack([d["OR"]-d["CI_low"], d["CI_high"]-d["OR"]])
    ax.errorbar(d["OR"], y, xerr=err, fmt="o", capsize=3)
    ax.axvline(1, linewidth=1)
    ax.set_yticks(y, d["Outcome"])
    ax.set_xlabel("Odds ratio (95% CI)")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig, ax
