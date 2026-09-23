from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


DEFAULT_LABELS = {
    "log_BNP": "log BNP",
    "echomass": "Ventricular mass",
    "echoedv": "EDV",
    "pp_peakvo2": "Peak VO2",
    "pp_vat": "VAT",
    "oavvregurg_ord": "Valve regurgitation",
    "echoef": "EF",
    "tei_index": "Tei index",
    "e_a": "E/A",
    "e_tde": "e'",
}

AXIS_TITLES = {
    "DM1": "Biomarker–remodeling axis",
    "DM2": "Valve–remodeling/function axis",
    "DM3": "Diastolic–functional axis",
}


def make_figure1(
    figure_df: pd.DataFrame,
    characterization: pd.DataFrame,
    pt_results: pd.DataFrame,
    pt_clinical: pd.DataFrame,
    pt_y: str = "pp_peakvo2",
    pt_y_label: str = "% predicted peak VO2",
    top_n: int = 4,
    figsize=(17, 10),
):
    """Create a manuscript-style multipanel summary figure.

    Panels:
      A seven-domain conceptual input
      B 3D diffusion manifold colored by pseudo-time
      C DM1–DM3 clinical characterization
      D pseudo-time vs a representative phenotype
      E forest plot for pseudo-time and complications
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1.0, 1.35, 1.1, 1.1, 1.2])

    # A
    axA = fig.add_subplot(gs[:, 0])
    axA.axis("off")
    axA.set_title("A  Seven-domain clinical input", loc="left", fontweight="bold")
    domains = [
        "Exercise", "Remodeling", "Function", "Diastolic",
        "Valve", "Anatomy / Surgery", "Biomarker"
    ]
    y = np.linspace(.85, .25, len(domains))
    for yy, name in zip(y, domains):
        axA.text(.5, yy, name, ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=.45", fill=False))
    axA.annotate("Diffusion map", xy=(.5,.08), xytext=(.5,.17),
                 ha="center", arrowprops=dict(arrowstyle="->"))
    axA.text(.5,.03,"Fixed reference + Nyström projection",
             ha="center", va="bottom", fontsize=9)

    # B
    axB = fig.add_subplot(gs[:, 1], projection="3d")
    sc = axB.scatter(figure_df["DM1"], figure_df["DM2"], figure_df["DM3"],
                     c=figure_df["pseudo_time"], s=16, alpha=.8)
    axB.set_xlabel("DM1"); axB.set_ylabel("DM2"); axB.set_zlabel("DM3")
    axB.set_title("B  Latent Fontan manifold", loc="left", fontweight="bold")
    cb = fig.colorbar(sc, ax=axB, shrink=.45, pad=.08)
    cb.set_label("Pseudo-time")

    # C: nested three axes
    sub = gs[0, 2:5].subgridspec(1, 3, wspace=.35)
    for j, dm in enumerate(["DM1","DM2","DM3"]):
        ax = fig.add_subplot(sub[0,j])
        g = characterization[characterization["DM"]==dm].nlargest(top_n,"abs_rho")
        g = g.sort_values("rho")
        labels = [DEFAULT_LABELS.get(v,v) for v in g["Variable"]]
        ax.barh(labels, g["rho"])
        ax.axvline(0, linewidth=1)
        ax.set_xlim(-1,1)
        ax.set_xlabel("Spearman rho")
        title = AXIS_TITLES.get(dm, dm)
        ax.set_title(f"{dm}\n{title}", fontsize=10, fontweight="bold")
        for i, (_,r) in enumerate(g.iterrows()):
            ax.text(r["rho"], i, f" {r['rho']:+.2f}",
                    va="center", ha="left" if r["rho"]>=0 else "right",
                    fontsize=8)
    fig.text(.50,.985,"C  Three latent clinical axes", fontweight="bold")

    # D
    axD = fig.add_subplot(gs[1,2])
    d = pt_clinical[["PT_mean",pt_y]].apply(pd.to_numeric,errors="coerce").dropna()
    axD.scatter(d["PT_mean"],d[pt_y],s=16,alpha=.6)
    if len(d)>2:
        b1,b0=np.polyfit(d["PT_mean"],d[pt_y],1)
        xx=np.linspace(d["PT_mean"].min(),d["PT_mean"].max(),100)
        axD.plot(xx,b1*xx+b0,linewidth=2)
        from scipy.stats import spearmanr
        rho,p=spearmanr(d["PT_mean"],d[pt_y])
        axD.text(.04,.96,f"rho = {rho:.2f}\nP = {p:.2g}",
                 transform=axD.transAxes,va="top")
    axD.set_xlabel("Pseudo-time")
    axD.set_ylabel(pt_y_label)
    axD.set_title("D  Clinical progression", loc="left", fontweight="bold")

    # E
    axE = fig.add_subplot(gs[1,3:5])
    d = pt_results.copy().sort_values("OR").reset_index(drop=True)
    names = {
        "AT_BIN":"Atrial tachyarrhythmia", "VT_BIN":"Ventricular tachyarrhythmia",
        "PACEMAKER_BIN":"Pacemaker", "THROMBOS_BIN":"Thrombosis",
        "STROKE_BIN":"Stroke", "PLE_BIN":"PLE",
    }
    d["label"] = d["Outcome"].map(names).fillna(d["Outcome"])
    yy=np.arange(len(d))
    err=np.vstack([d["OR"]-d["CI_low"],d["CI_high"]-d["OR"]])
    axE.errorbar(d["OR"],yy,xerr=err,fmt="o",capsize=3)
    axE.axvline(1,linewidth=1)
    axE.set_yticks(yy,d["label"])
    axE.set_xlabel("OR per 1-SD higher pseudo-time (95% CI)")
    axE.set_title("E  Complication associations", loc="left", fontweight="bold")
    for i,r in d.iterrows():
        q=r.get("q_value",np.nan)
        txt=f"{r['OR']:.2f} ({r['CI_low']:.2f}–{r['CI_high']:.2f})"
        if np.isfinite(q): txt += f", q={q:.3g}"
        axE.text(r["CI_high"]*1.03,i,txt,va="center",fontsize=8)
    axE.set_xscale("log")

    return fig
