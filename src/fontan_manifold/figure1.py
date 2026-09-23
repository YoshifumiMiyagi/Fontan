from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr

DEFAULT_LABELS = {
    "log_BNP": "log BNP", "echomass": "Ventricular mass", "echoedv": "EDV",
    "pp_peakvo2": "Peak VO2", "pp_vat": "VAT", "oavvregurg_ord": "Valve regurgitation",
    "echoef": "EF", "tei_index": "Tei index", "e_a": "E/A", "e_tde": "e'",
}
AXIS_TITLES = {
    "DM1": "Biomarker–remodeling",
    "DM2": "Valve–remodeling/function",
    "DM3": "Diastolic–functional",
}
OUTCOME_ORDER = ["AT_BIN","VT_BIN","PACEMAKER_BIN","THROMBOS_BIN","STROKE_BIN","PLE_BIN"]
OUTCOME_LABELS = {
    "AT_BIN":"Atrial tachyarrhythmia", "VT_BIN":"Ventricular tachyarrhythmia",
    "PACEMAKER_BIN":"Pacemaker", "THROMBOS_BIN":"Thrombosis",
    "STROKE_BIN":"Stroke", "PLE_BIN":"PLE",
}

def make_figure1(
    figure_df, characterization, pt_results, pt_clinical,
    pt_y="pp_peakvo2", pt_y_label="% predicted peak VO2",
    top_n=4, root_subject_id=None, id_col="subj_id",
    figsize=(16,11),
):
    """Publication-oriented integrated Fontan manifold Figure 1."""
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(3, 12, figure=fig, height_ratios=[0.75, 3.2, 2.4])

    # A: compact horizontal workflow
    axA=fig.add_subplot(gs[0,:]); axA.axis("off")
    axA.set_title("A  Seven-domain clinical framework", loc="left", fontweight="bold")
    domains=["Exercise","Remodeling","Function","Diastolic","Valve","Anatomy / Surgery","Biomarker"]
    xs=np.linspace(.04,.57,len(domains))
    for x,name in zip(xs,domains):
        axA.text(x,.55,name,ha="center",va="center",fontsize=9,
                 bbox=dict(boxstyle="round,pad=.35",fill=False))
    axA.annotate("Diffusion map",xy=(.72,.55),xytext=(.62,.55),ha="center",va="center",
                 arrowprops=dict(arrowstyle="->",lw=1.4))
    axA.annotate("DM1–3 + pseudo-time",xy=(.96,.55),xytext=(.78,.55),ha="center",va="center",
                 arrowprops=dict(arrowstyle="->",lw=1.4))
    axA.text(.50,.08,"Fixed-reference manifold  |  multiple imputation  |  Nyström projection",
             ha="center",fontsize=9)

    # B: larger manifold
    axB=fig.add_subplot(gs[1,0:6],projection="3d")
    sc=axB.scatter(figure_df.DM1,figure_df.DM2,figure_df.DM3,
                   c=figure_df.pseudo_time,s=22,alpha=.82)
    axB.set_xlabel("DM1"); axB.set_ylabel("DM2"); axB.set_zlabel("DM3")
    axB.set_title("B  Latent Fontan manifold",loc="left",fontweight="bold")
    if root_subject_id is not None and id_col in figure_df.columns:
        r=figure_df[figure_df[id_col]==root_subject_id]
        if len(r):
            axB.scatter(r.DM1,r.DM2,r.DM3,marker="*",s=220,
                        edgecolor="black",linewidth=1.2,label="Clinical low-burden root")
            axB.legend(loc="upper left",fontsize=8)
    cb=fig.colorbar(sc,ax=axB,shrink=.68,pad=.08)
    cb.set_label("Pseudo-time\n(distance from clinical low-burden root)")

    # C: compact characterization block
    sub=gs[1,6:12].subgridspec(1,3,wspace=.5)
    for j,dm in enumerate(["DM1","DM2","DM3"]):
        ax=fig.add_subplot(sub[0,j])
        g=characterization[characterization.DM==dm].nlargest(top_n,"abs_rho").sort_values("rho")
        labels=[DEFAULT_LABELS.get(v,v) for v in g.Variable]
        ax.barh(labels,g.rho)
        ax.axvline(0,lw=1); ax.set_xlim(-1,1)
        ax.set_xlabel("Spearman rho",fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_title(f"{dm}\n{AXIS_TITLES[dm]}",fontsize=9,fontweight="bold")
        for i,(_,r) in enumerate(g.iterrows()):
            ax.text(r.rho,i,f" {r.rho:+.2f}",va="center",
                    ha="left" if r.rho>=0 else "right",fontsize=7)
    fig.text(.515,.625,"C  Three latent clinical axes",fontweight="bold")

    # D: PT vs representative clinical phenotype
    axD=fig.add_subplot(gs[2,0:5])
    d=pt_clinical[["PT_mean",pt_y]].apply(pd.to_numeric,errors="coerce").dropna()
    axD.scatter(d.PT_mean,d[pt_y],s=18,alpha=.6)
    if len(d)>2:
        b1,b0=np.polyfit(d.PT_mean,d[pt_y],1)
        xx=np.linspace(d.PT_mean.min(),d.PT_mean.max(),100)
        axD.plot(xx,b1*xx+b0,lw=2)
        rho,p=spearmanr(d.PT_mean,d[pt_y])
        axD.text(.04,.96,f"rho = {rho:.2f}\nP = {p:.2g}",transform=axD.transAxes,va="top")
    axD.set_xlabel("Pseudo-time"); axD.set_ylabel(pt_y_label)
    axD.set_title("D  Functional phenotype along pseudo-time",loc="left",fontweight="bold")

    # E: clinically ordered forest plot
    axE=fig.add_subplot(gs[2,5:12])
    d=pt_results.copy()
    rank={x:i for i,x in enumerate(OUTCOME_ORDER)}
    d=d[d.Outcome.isin(OUTCOME_ORDER)].copy()
    d["_rank"]=d.Outcome.map(rank); d=d.sort_values("_rank").reset_index(drop=True)
    yy=np.arange(len(d))
    err=np.vstack([d.OR-d.CI_low,d.CI_high-d.OR])
    axE.errorbar(d.OR,yy,xerr=err,fmt="o",capsize=3)
    axE.axvline(1,lw=1)
    axE.set_yticks(yy,[OUTCOME_LABELS[x] for x in d.Outcome])
    axE.invert_yaxis(); axE.set_xscale("log")
    axE.set_xlabel("OR per 1-SD higher pseudo-time (95% CI)")
    axE.set_title("E  Association with clinical complications",loc="left",fontweight="bold")
    xmax=float(np.nanmax(d.CI_high))
    for i,r in d.iterrows():
        q=r.get("q_value",np.nan)
        txt=f"{r.OR:.2f} ({r.CI_low:.2f}–{r.CI_high:.2f})"
        if np.isfinite(q): txt+=f", q={q:.3g}"
        weight="bold" if np.isfinite(q) and q<.05 else "normal"
        axE.text(r.CI_high*1.035,i,txt,va="center",fontsize=8,fontweight=weight)
    axE.set_xlim(min(.5,float(np.nanmin(d.CI_low))*.9),xmax*1.9)

    return fig
