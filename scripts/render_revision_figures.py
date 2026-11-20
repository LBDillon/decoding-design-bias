"""Draw Figure 4 and Supplementary Figures S9-S11 from the inputs, Source Data and render assets.

The plotted tables are exported under OUTPUT_DIR/source_data; the Source Data
directory is only read.
"""
from pathlib import Path
import json
import shutil
import sys

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-data", type=Path, required=True)
parser.add_argument("--render-assets", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
ROOT = args.output_dir.resolve()
EXPORT = ROOT / "source_data"
TMP = ROOT / "figure_4_panels"
CODE = Path(__file__).resolve().parents[1]
for folder in [TMP, EXPORT / "GAM_grids", EXPORT / "Figure_4c_and_Table_S20", EXPORT / "Supplementary_Figure_S10"]:
    folder.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE / "src"))
from decoding_bias.analysis import finetune
from decoding_bias.analysis.design import MODELS
from decoding_bias.catalog import BIOPHYS_14
from decoding_bias.analysis.gam_display import display_layers

DATA = args.source_data.resolve()
variance=pd.read_csv(DATA/"recomputed/pca/pca_variance.csv").set_index("PC").explained_variance_pct
pc1=f"PC1 ({variance.PC1:.1f}%)";pc2=f"PC2 ({variance.PC2:.1f}%)"
plt.rcParams.update({"font.family": "Arial", "font.size": 6.4, "axes.labelsize": 6.4, "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": .5, "savefig.facecolor": "white", "savefig.dpi": 600})
MM = 1 / 25.4
FT = ["ProteinMPNN", "AlkalineMPNN_020", "AcidophileMPNN_020"]
NAMES = {"WT": "WT", "ProteinMPNN": "ProteinMPNN", "AlkalineMPNN_020": "AlkSecMPNN", "AcidophileMPNN_020": "AcidSecMPNN"}
COLOURS = {"WT": "#333333", "ProteinMPNN": "#6E9ED0", "AlkalineMPNN_020": "#64AF62", "AcidophileMPNN_020": "#EF9849"}
LABEL_BOX={"facecolor":"white","edgecolor":"none","alpha":1.,"pad":.7}
SURFACE_LABELS={
    "A9CIG1":{"WT":(-2.55,1.48,"left"),"ProteinMPNN":(.45,1.26,"left"),"AlkalineMPNN_020":(2.75,-.60,"right"),"AcidophileMPNN_020":(-1.95,-1.69,"left")},
    "O67786":{"WT":(.70,2.48,"left"),"ProteinMPNN":(-.10,-.62,"center"),"AlkalineMPNN_020":(2.72,-.35,"right"),"AcidophileMPNN_020":(-1.35,.20,"right")}}

def common_transparent_crop(cells, padding=.04):
    """Remove common transparent margins, keeping all four models at the same scale."""
    bounds=[]
    for cell in cells:
        yy,xx=np.where(cell[:,:,3]>0)
        if not len(xx):raise ValueError("Empty structure render")
        bounds.append((xx.min(),yy.min(),xx.max()+1,yy.max()+1))
    left=min(b[0] for b in bounds);top=min(b[1] for b in bounds)
    right=max(b[2] for b in bounds);bottom=max(b[3] for b in bounds)
    pad=int(np.ceil(max(right-left,bottom-top)*padding))
    left=max(0,left-pad);right=min(cells[0].shape[1],right+pad)
    top=max(0,top-pad);bottom=min(cells[0].shape[0],bottom+pad)
    return [cell[top:bottom,left:right] for cell in cells],(left,top,right,bottom)

def surface_label(ax,uid,model,point):
    x,y,ha=SURFACE_LABELS[uid][model]
    return ax.annotate(NAMES[model],point,xytext=(x,y),textcoords="data",ha=ha,va="center",fontsize=6,bbox=LABEL_BOX,zorder=9,
                       arrowprops={"arrowstyle":"-","lw":.4,"color":".35","shrinkA":2,"shrinkB":3})
surface = pd.read_csv(CODE / "data/finetune/design_surface_features.csv")
projected, loads, explained = finetune._surface_pca(surface)
assert np.allclose(explained[:2] * 100, [64.23660709188265, 35.50910235324802], atol=1e-9)
per_protein, tests, _ = finetune.surface_shift_tests(surface)
assert np.isclose(tests.iloc[0].mean_delta_PC1, 1.935777507769559)
assert np.isclose(tests.iloc[1].mean_delta_PC1, -1.004373932692931)
centroids = projected.groupby(["uniprot_id", "model"])[["PC1", "PC2"]].mean().reset_index()
source4a = EXPORT / "Figure_4a"
source4a.mkdir(exist_ok=True)
projected.to_csv(source4a / "all_surface_pca_coordinates.csv", index=False)
centroids.to_csv(source4a / "template_model_centroids.csv", index=False)
loads.to_csv(source4a / "surface_pca_loadings.csv", index=False)
pd.DataFrame({"feature": finetune.SURFACE_FEATURES, "center": surface[finetune.SURFACE_FEATURES].mean(), "scale": surface[finetune.SURFACE_FEATURES].std(ddof=1)}).to_csv(source4a / "surface_pca_scaling.csv", index=False)
(source4a / "pca_definition.json").write_text(json.dumps({"n_fit_records": 1225, "n_templates": 25, "fit_models": surface.model.value_counts().to_dict(), "features": finetune.SURFACE_FEATURES, "scaling_ddof": 1, "PC1_positive": "higher surface acidic fraction", "explained_variance_percent": (100 * explained).tolist(), "plotted_templates": ["A9CIG1", "O67786"], "designs_per_template_model": 8}, indent=2) + "\n")

# Representative structures: the author's native-resolution renders and their selections.
assets = args.render_assets.resolve()
selection = pd.read_csv(assets / "mutations_summary.csv")
selection = selection[selection.uniprot_id.isin(["A9CIG1", "O67786"])].copy()
selection.to_csv(source4a / "representative_structure_selection.csv", index=False)
for uid in ["A9CIG1", "O67786"]:
    for name in ["WT", "ProteinMPNN", "AlkSecMPNN", "AcidSecMPNN"]:
        p = assets / "pdbs" / f"{uid}_{name}.pdb"
        (source4a / "structures").mkdir(exist_ok=True)
        shutil.copy2(p, source4a / "structures" / p.name)

fig = plt.figure(figsize=(180 * MM, 100 * MM))
fig.text(.01, .98, "a", weight="bold", fontsize=7, va="top")
fig.text(.045, .98, "Surface acid–base PCA and representative structures", weight="bold", fontsize=7, va="top")
for j, (uid, title) in enumerate([("A9CIG1", "Small ribosomal subunit protein uS14  A9CIG1"), ("O67786", "Enolase-phosphatase E1  O67786")]):
    left = .075 + .50 * j
    ax = fig.add_axes([left, .40, .37, .48])
    ax.set_title(title, fontsize=6.4, pad=5)
    sub = centroids[centroids.uniprot_id.eq(uid)].set_index("model")
    wild = sub.loc["WT", ["PC1", "PC2"]].to_numpy(float)
    ax.scatter(*wild, facecolor="white", edgecolor="black", s=23, zorder=5)
    surface_label(ax,uid,"WT",wild)
    for model in FT:
        end = sub.loc[model, ["PC1", "PC2"]].to_numpy(float)
        ax.annotate("", xy=end, xytext=wild, arrowprops={"arrowstyle": "-|>", "color": COLOURS[model], "lw": 1}, zorder=5)
        ax.scatter(*end, color=COLOURS[model], s=12, zorder=4)
        surface_label(ax,uid,model,end)
    ax.set(xlim=(-3, 3.3), ylim=(-2, 3), xlabel=f"Surface PC1 ({100*explained[0]:.1f}%)", ylabel=f"Surface PC2 ({100*explained[1]:.1f}%)")
    ax.set_axisbelow(True); ax.grid(color=".93", lw=.3, zorder=0)
    raw = np.asarray(Image.open(assets / "renders" / f"{uid}_muts_spheres_surface.png").convert("RGBA"))
    h, w = raw.shape[:2]
    cells = [raw[:h//2, :w//2], raw[:h//2, w//2:], raw[h//2:, :w//2], raw[h//2:, w//2:]]
    cells,crop=common_transparent_crop(cells)
    (source4a/f"{uid}_display_crop.json").write_text(json.dumps({"pixel_crop_per_3200x2000_cell":list(map(int,crop)),"common_to_four_model_renders":True,"native_pixels_edited":False},indent=2)+"\n")
    for k, (image, name) in enumerate(zip(cells, ["WT", "ProteinMPNN", "AlkSecMPNN", "AcidSecMPNN"])):
        a = fig.add_axes([.006 + .5*j + .123*k, .057, .121, .27])
        a.imshow(image); a.axis("off"); a.set_title(name, fontsize=6, pad=1)
fig.text(.5, .015, "Mutation spheres: acidic D/E red; basic K/R blue; neutral white", fontsize=6, ha="center")
fig.savefig(TMP / "Figure_4a.pdf"); plt.close(fig)

# Two fine-tuned-minus-base GAMs in the fixed global natural-protein PCA.
grids = []
for name in ["AlkSecMPNN", "AcidSecMPNN"]:
    table = pd.read_csv(DATA / "GAM_grids" / f"{name}_minus_base_grid.csv")
    assert len(table) == 160 * 160 and table.supported.any()
    grids.append((name, table))
limit = max(float(t.loc[t.supported, "predicted_score"].abs().quantile(.95)) for _, t in grids)
(EXPORT / "GAM_grids/figure4b_display_parameters.json").write_text(json.dumps({"colour_map": "viridis", "vmin": -limit, "vmax": limit, "symmetric_scale_rule": "maximum across the two panels of the 95th percentile of absolute fitted values in supported cells", "values_outside_colour_limits": "saturated; unmodified values retained in grid CSVs", "display_x_range": [-6, 6], "display_y_range": [-6, 6], "support": "at least 3 supporting proteins and inside observed convex hull; other predictions shown at alpha 0.22 without contours"}, indent=2) + "\n")
fig = plt.figure(figsize=(95 * MM, 80 * MM))
fig.text(.015, .98, "b", weight="bold", fontsize=7, va="top")
fig.text(.065, .98, "Fine-tuned score difference from ProteinMPNN", weight="bold", fontsize=6.6, va="top")
for i, (name, table) in enumerate(grids):
    ax = fig.add_axes([.12 + .44*i, .24, .37, .61])
    x,y,z,opacity,contours=display_layers(table)
    mesh = ax.pcolormesh(x,y,z,cmap="viridis",vmin=-limit,vmax=limit,alpha=opacity,shading="nearest")
    ax.contour(x,y,contours,colors="white",linewidths=.25,levels=8,alpha=.65)
    ax.set(xlim=(-6,6), ylim=(-6,6), xlabel=pc1, title=f"{name} − base")
    ax.set_xticks([-4,0,4]); ax.set_yticks([-4,0,4]); ax.set_aspect("equal")
    if i==0: ax.set_ylabel(pc2)
    else: ax.set_yticklabels([])
cax = fig.add_axes([.27,.10,.54,.045]); bar=fig.colorbar(mesh,cax=cax,orientation="horizontal")
bar.set_label("Difference between within-model z scores",fontsize=6)
fig.savefig(TMP / "Figure_4b.pdf"); plt.close(fig)

# Original seven-model centroid shifts, in the 14-feature global frame.
natural = pd.read_csv(CODE / "data/main_analysis.csv")
matrix = (natural[BIOPHYS_14] - natural[BIOPHYS_14].mean()) / natural[BIOPHYS_14].std(ddof=1)
_, _, vt = np.linalg.svd(matrix.to_numpy(), full_matrices=False)
rotation = vt[:2].T
if rotation[BIOPHYS_14.index("sequence_length"),0] < 0: rotation[:,0] *= -1
if rotation[BIOPHYS_14.index("instability_index"),1] < 0: rotation[:,1] *= -1
design = pd.read_csv(CODE / "data/design/designs_features.csv")
wild = pd.read_csv(CODE / "data/design/wt_features.csv").set_index("uniprot_id")
means = design[design.model.isin(MODELS)].groupby(["model","uniprot_id"])[BIOPHYS_14].mean()
delta = means.copy()
for key in means.index: delta.loc[key] = means.loc[key] - wild.loc[key[1],BIOPHYS_14]
coordinates = (delta / natural[BIOPHYS_14].std(ddof=1)).to_numpy() @ rotation
points = delta.reset_index()[["model","uniprot_id"]]; points[["dPC1","dPC2"]] = coordinates
points.to_csv(EXPORT / "Figure_4c_and_Table_S20/template_pca_shifts.csv",index=False)
avg = points.groupby("model")[["dPC1","dPC2"]].mean().reindex(MODELS)
avg.to_csv(EXPORT / "Figure_4c_and_Table_S20/plotted_model_centroids.csv")
fig, ax = plt.subplots(figsize=(85*MM,80*MM))
fig.subplots_adjust(left=.20,bottom=.18,right=.97,top=.88)
fig.text(.025,.98,"c",weight="bold",fontsize=7,va="top")
fig.text(.09,.98,"Design shifts relative to WT centroid",weight="bold",fontsize=6.6,va="top")
palette=["#76A1CC","#F39B45","#63B56B","#DF6A65","#A38AC3","#AA7F78","#DA87BA"]
offsets={"ProteinMPNN":(3,-9),"SolubleMPNN":(-24,5),"Caliby":(-23,0),"SolubleCaliby":(-43,1),"ESM-IF":(3,1),"MIF":(3,4),"MIF-ST":(-15,-11)}
for model,colour in zip(MODELS,palette):
    x,y=avg.loc[model]
    ax.annotate("",(x,y),(0,0),arrowprops={"arrowstyle":"-|>","lw":1.1,"color":colour},zorder=4)
    ax.scatter(x,y,s=13,color=colour,zorder=5)
    ax.annotate(model,(x,y),xytext=offsets[model],textcoords="offset points",fontsize=6,color=".15",bbox=LABEL_BOX,zorder=7)
ax.axhline(0,color=".78",lw=.4,zorder=1);ax.axvline(0,color=".78",lw=.4,zorder=1)
ax.scatter([0],[0],s=22,color="black",edgecolors="white",linewidths=.35,zorder=8)
ax.set(xlabel="Δ"+pc1,ylabel="Δ"+pc2,xlim=(-.5,.5),ylim=(-.5,.90))
ax.set_axisbelow(True);ax.grid(color=".94",lw=.3,zorder=0)
fig.savefig(TMP / "Figure_4c.pdf");plt.close(fig)
original=ROOT/"Figure_4.pdf"
doc=fitz.open();page=doc.new_page(width=180*72/25.4,height=180*72/25.4)
for filename,rect in [("Figure_4a.pdf",(0,0,180,100)),("Figure_4b.pdf",(0,100,95,180)),("Figure_4c.pdf",(95,100,180,180))]:
    part=fitz.open(TMP/filename);page.show_pdf_page(fitz.Rect(*(v*72/25.4 for v in rect)),part,0)
doc.save(original,garbage=4,deflate=True);doc.close()

# Supplementary Figure S9: show individual protein-level observations.
recovery=pd.read_csv(DATA/"Supplementary_Figure_S09/template_level_recovery.csv")
order=MODELS+["AlkSecMPNN","AcidSecMPNN"]
order=[m for m in order if m in set(recovery.model)]
assert len(order)==9
labels=[{"AlkalineMPNN":"AlkSecMPNN","AcidophileMPNN":"AcidSecMPNN"}.get(m,m) for m in order]
fig,axes=plt.subplots(1,2,figsize=(180*MM,85*MM));fig.subplots_adjust(left=.065,right=.98,bottom=.27,top=.88,wspace=.30)
for j,model in enumerate(order):
    sub=recovery[recovery.model.eq(model)];jit=np.linspace(-.07,.07,len(sub))
    for column,offset,colour in [("func_recovery",-.16,"#D89000"),("bg_recovery",.16,"#559BC8")]:
        y=sub[column].to_numpy();axes[0].scatter(j+offset+jit,y,s=5,color=colour,alpha=.65)
        axes[0].plot([j+offset-.08,j+offset+.08],[y.mean()]*2,color="black",lw=1)
    y=sub.functional_minus_background.to_numpy();axes[1].scatter(j+jit,y,s=7,color=".35",alpha=.7)
    axes[1].plot([j-.12,j+.12],[y.mean()]*2,color="black",lw=1.2)
axes[0].set(ylabel="WT-sequence recovery",ylim=(-.03,1.03));axes[1].set(ylabel="Functional minus background recovery")
axes[1].axhline(0,color=".65",lw=.5)
for ax,letter,title in zip(axes,["a","b"],["Functional and background positions","Paired template-level differences"]):
    ax.set_xticks(range(len(order)),labels,rotation=40,ha="right");ax.set_title(title,fontsize=7)
    ax.text(-.12,1.06,letter,transform=ax.transAxes,weight="bold",fontsize=7)
fig.text(.5,.985,"Orange: functional positions    Blue: background    Black bars: means",ha="center",va="top",fontsize=6.3)
fig.savefig(ROOT/"supplementary_figure_9.pdf");plt.close(fig)

# Supplementary Figure S10: use the same four-feature frame as main Figure 4a.
fig=plt.figure(figsize=(180*MM,150*MM));ax=fig.add_axes([.10,.64,.82,.29])
for model in ["WT"]+FT:
    sub=centroids[centroids.model.eq(model)]
    ax.scatter(sub.PC1,sub.PC2,s=9,facecolor="white" if model=="WT" else COLOURS[model],edgecolor=COLOURS[model],lw=.45,alpha=.65,label=NAMES[model])
global_means=centroids[centroids.model.isin(FT)].groupby("model")[["PC1","PC2"]].mean()
for model in FT[1:]: ax.annotate("",global_means.loc[model],global_means.loc[FT[0]],arrowprops={"arrowstyle":"-|>","lw":1.4,"color":COLOURS[model]})
ax.set(xlabel=f"Surface PC1 ({100*explained[0]:.1f}%)",ylabel=f"Surface PC2 ({100*explained[1]:.1f}%)")
ax.legend(loc="lower center",bbox_to_anchor=(.5,1.02),ncol=4,frameon=False);ax.grid(color=".93",lw=.35);ax.text(-.085,1.06,"a",transform=ax.transAxes,weight="bold",fontsize=7)
wt_surface=surface[surface.model.eq("WT")].set_index("uniprot_id")
ft_surface=surface[surface.model.isin(FT)].groupby(["uniprot_id","model"])[finetune.DIRECT_FEATURES].mean()
direct_source=[]
feature_labels=["Surface net charge","Surface acidic fraction","Surface basic fraction","Sequence isoelectric point","Sequence charge per residue","Surface ionisable fraction"]
for k,(feature,label) in enumerate(zip(finetune.DIRECT_FEATURES,feature_labels)):
    a=fig.add_axes([.09+.315*(k%3),.10+.24*(1-k//3),.235,.17])
    wide=ft_surface[feature].unstack("model")
    for j,model in enumerate(FT[1:]):
        values=(wide[model]-wide[FT[0]]).dropna()
        for uid,value in values.items():direct_source.append({"uniprot_id":uid,"model":model,"feature":feature,"fine_tuned_minus_base":value})
        a.scatter(j+np.linspace(-.08,.08,len(values)),values,s=5,color=COLOURS[model],alpha=.55)
        a.errorbar(j,values.mean(),yerr=values.std(ddof=1)/np.sqrt(len(values)),fmt="o",color="black",ms=2.5,capsize=2,lw=.6)
    a.axhline(0,color=".7",lw=.4);a.set_xticks([0,1],["AlkSec","AcidSec"]);a.set_title(label,fontsize=6.5)
    if k==0:a.text(-.27,1.18,"b",transform=a.transAxes,weight="bold",fontsize=7)
fig.text(.012,.34,"Fine-tuned minus base",rotation=90,fontsize=6.4,va="center")
fig.savefig(ROOT/"supplementary_figure_10.pdf");plt.close(fig)
pd.DataFrame(direct_source).to_csv(EXPORT/"Supplementary_Figure_S10/template_level_direct_shifts.csv",index=False)
centroids[centroids.model.isin(["WT"]+FT)].to_csv(EXPORT/"Supplementary_Figure_S10/plotted_surface_centroids.csv",index=False)

# Supplementary Figure S11: each plotted point is an independent template.
sc=pd.read_csv(DATA/"Supplementary_Figure_S11/template_level_structural_metrics.csv")
sc_models=finetune.SELF_CONSISTENCY_MODELS
fig,axes=plt.subplots(1,3,figsize=(180*MM,72*MM));fig.subplots_adjust(left=.07,right=.98,bottom=.25,top=.86,wspace=.39)
for ax,metric,title,letter in zip(axes,["scTM","scRMSD","pLDDT"],["Self-consistency TM-score","Cα RMSD (Å)","Mean pLDDT"],["a","b","c"]):
    values=[sc.loc[sc.model.eq(m),metric].to_numpy() for m in sc_models]
    assert all(len(v)==25 for v in values)
    ax.boxplot(values,positions=range(4),widths=.45,showfliers=False,medianprops={"color":"black","lw":.8},boxprops={"lw":.6},whiskerprops={"lw":.6},capprops={"lw":.6})
    for i,v in enumerate(values):ax.scatter(i+np.linspace(-.12,.12,len(v)),v,s=5,color=["#6E9ED0","#64AF62","#EF9849",".5"][i],alpha=.65,zorder=3)
    ax.set_xticks(range(4),["Base","AlkSec","AcidSec","WT"],rotation=30,ha="right");ax.set_title(title,fontsize=6.5);ax.text(-.20,1.10,letter,transform=ax.transAxes,weight="bold",fontsize=7)
fig.savefig(ROOT/"supplementary_figure_11.pdf");plt.close(fig)

print(f"Rendered Figure 4 and Supplementary Figures S9-S11 in {ROOT}; plotted tables in {EXPORT}")
