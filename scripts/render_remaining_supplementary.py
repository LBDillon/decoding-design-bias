"""Draw Supplementary Figures S1, S2, S4-S8 and S12 from the Source Data.

The plotted tables are exported under OUTPUT_DIR/source_data; the Source Data
directory is only read.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from decoding_bias.analysis.figure_data import MODEL_ORDER,LABELS,CLASSES
from decoding_bias.catalog import BIOPHYS_14
from decoding_bias.analysis.gam_display import display_layers
parser=argparse.ArgumentParser();parser.add_argument('--source-data',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
data=args.source_data.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);export=out/'source_data'
plt.rcParams.update({'font.family':'Arial','font.size':6,'axes.labelsize':6,'axes.titlesize':7,'xtick.labelsize':5.5,'ytick.labelsize':5.5,'legend.fontsize':5.5,'axes.linewidth':.5,'pdf.fonttype':42,'savefig.facecolor':'white'})
mm=1/25.4;domain_colours={'Archaea':'#D55E00','Bacteria':'#0072B2','Eukaryota':'#009E73'}
contexts={'Backbone-conditioned':'#2C7BB6','Structure + native sequence':'#7B3294','Sequence-only':'#1A9641'}
names=[LABELS[m] for m in MODEL_ORDER]
def save(fig,n):fig.savefig(out/f'supplementary_figure_{n}.pdf');plt.close(fig)
def source_folder(n):p=export/f'Supplementary_Figure_S{n:02d}';p.mkdir(parents=True,exist_ok=True);return p
# S1: retention bars with black correlation points. S2: paired species-variance bars.
original_order=['SolubleMPNN','SolubleCaliby','Caliby','ProteinMPNN','ESM-IF','ESM3-struct','TriFlow','MIF','MIF-ST','ESM3-seq','ProGen2-XL','ProtGPT2','CARP-640M','ESM2-15B']
vd=pd.read_csv(data/'recomputed/variance/score_variance_decomposition.csv').set_index('model').reindex(original_order);assert vd.R2_Full.notna().all()
vd.to_csv(source_folder(1)/'displayed_model_estimates.csv');vd.to_csv(source_folder(2)/'displayed_model_estimates.csv')
x=np.arange(len(vd));colours=['#2E7D32']*7+['#F9A825','#C0CA33']+['#00838F']*5
fig,ax=plt.subplots(figsize=(180*mm,105*mm));fig.subplots_adjust(left=.11,right=.98,bottom=.31,top=.87)
ax.bar(x,vd.species_effect_retention_given_family_biophys,color=colours,width=.72,label='Effect-size retention')
ax.scatter(x,vd.species_effect_corr_given_family_biophys,color='black',s=12,zorder=3,label='Effect correlation')
for i,v in enumerate(vd.species_effect_retention_given_family_biophys):
 corr=vd.species_effect_corr_given_family_biophys.iloc[i];label_y=max(v,corr)+.035 if abs(corr-v)<.09 else v+.025
 ax.text(i,label_y,f'{v:.2f}',ha='center',fontsize=6)
ax.axhline(1,color='.5',ls='--',lw=.6);ax.axvline(8.5,color='.65',ls=':',lw=.6)
ax.set(xticks=x,xticklabels=original_order,ylabel='Retention / Pearson correlation',ylim=(0,1.15),title='Species-effect retention within protein families')
plt.setp(ax.get_xticklabels(),rotation=55,ha='right');ax.legend(frameon=False,ncol=2,loc='upper left',fontsize=5.5)
ax.grid(axis='y',color='.92',lw=.4);ax.set_axisbelow(True);save(fig,1)
fig,ax=plt.subplots(figsize=(180*mm,105*mm));fig.subplots_adjust(left=.11,right=.98,bottom=.31,top=.87)
ax.bar(x-.18,vd.species_only_R2,width=.36,color='#FFCC80',label='Species-only R²')
ax.bar(x+.18,vd.residual_species_R2,width=.36,color='#EF6C00',label='Residual species increment')
upper=max(vd.species_only_R2.max(),vd.residual_species_R2.max())
for i,r in enumerate(vd.itertuples()):ax.text(i,r.species_only_R2+.025*upper,f'{100*r.species_attenuation:.0f}%',ha='center',fontsize=6)
ax.axvline(8.5,color='.65',ls=':',lw=.6)
ax.set(xticks=x,xticklabels=original_order,ylabel='Fraction of score variance',ylim=(0,upper*1.23),title='Species-signal attenuation after family and biophysical adjustment')
plt.setp(ax.get_xticklabels(),rotation=55,ha='right');ax.legend(frameon=False,ncol=2,loc='upper left',fontsize=5.5)
ax.grid(axis='y',color='.92',lw=.4);ax.set_axisbelow(True);save(fig,2)

# Species distributions: show every species observation and its domain density.
ratings=pd.read_csv(data/'Figure_2/all_models_species_ratings_long.csv');ratings.to_csv(source_folder(4)/'all_species_ratings.csv',index=False)
fig,axes=plt.subplots(4,4,figsize=(180*mm,175*mm));fig.subplots_adjust(left=.08,right=.98,bottom=.10,top=.94,wspace=.27,hspace=.45)
rng=np.random.default_rng(42)
for k,m in enumerate(MODEL_ORDER):
 ax=axes.flat[k];sub=ratings[ratings.model.eq(m)]
 for j,domain in enumerate(domain_colours):
  values=sub[sub.domain.eq(domain)].rating.to_numpy()-1500
  violin=ax.violinplot([values],positions=[j],widths=.8,showextrema=False)
  for artist in violin['bodies']:artist.set_facecolor(domain_colours[domain]);artist.set_edgecolor(domain_colours[domain]);artist.set_alpha(.2)
  ax.scatter(j+rng.uniform(-.22,.22,len(values)),values,s=.6,c=domain_colours[domain],alpha=.45,linewidths=0)
 ax.axhline(0,color='.4',lw=.4);ax.set(title=LABELS[m],xticks=[0,1,2],xticklabels=['A','B','E'],ylim=(-1000,1100),yticks=[-800,0,800])
 if k%4:ax.set_yticklabels([])
for k in [14,15]:axes.flat[k].axis('off')
fig.text(.5,.025,'A: Archaea (orange)     B: Bacteria (blue)     E: Eukaryota (green)',ha='center',fontsize=6)
fig.text(.015,.52,'Species Elo − 1500',rotation=90,va='center',fontsize=6);save(fig,4)

variance=pd.read_csv(data/'recomputed/pca/pca_variance.csv').set_index('PC').explained_variance_pct
pc1=f'PC1 ({variance.PC1:.1f}%)';pc2=f'PC2 ({variance.PC2:.1f}%)'
natural=pd.read_csv(data/'inputs/main_analysis.csv');coords=pd.read_csv(data/'GAM_grids/global_pca_coordinates.csv').merge(natural[['Entry','domain','species','protein_family','broad_function']],on='Entry',validate='one_to_one')
fig,axes=plt.subplots(1,2,figsize=(180*mm,80*mm));fig.subplots_adjust(left=.09,right=.98,bottom=.20,top=.92,wspace=.26)
kde_rows=[]
for k,pc in enumerate(['PC1','PC2']):
 grid=np.linspace(-7,8,400)
 for j,(domain,colour) in enumerate(domain_colours.items()):
  values=coords.loc[coords.domain.eq(domain),pc].to_numpy();density=gaussian_kde(values,bw_method='scott')(grid)
  axes[k].plot(grid,density,color=colour,lw=.9,label=f'{domain} (n={len(values)})');axes[k].fill_between(grid,0,density,color=colour,alpha=.15)
  kde_rows += [{'component':pc,'domain':domain,'x':a,'density':b,'n_proteins':len(values)} for a,b in zip(grid,density)]
 axes[k].set(xlabel=pc1 if k==0 else pc2,ylabel='Probability density',xlim=(-7,8));axes[k].grid(color='.94',lw=.35)
axes[0].legend(frameon=False,fontsize=5.5);pd.DataFrame(kde_rows).to_csv(source_folder(5)/'Scott_bandwidth_density_curves.csv',index=False);coords.to_csv(source_folder(5)/'protein_level_coordinates.csv',index=False);save(fig,5)

# Functional means and species/family centroids retain their underlying observations.
fig,axes=plt.subplots(2,2,figsize=(180*mm,170*mm));fig.subplots_adjust(left=.22,right=.98,bottom=.10,top=.94,wspace=.65,hspace=.36)
fun=coords.groupby('broad_function').agg(PC1=('PC1','mean'),PC2=('PC2','mean'),n=('Entry','size'),domain=('domain',lambda x:x.mode().iat[0])).reset_index();fun.to_csv(source_folder(6)/'functional_class_centroids.csv',index=False)
for k,pc in enumerate(['PC1','PC2']):
 table=fun.sort_values(pc);y=np.arange(len(table));ax=axes[0,k];ax.scatter(table[pc],y,s=6+table.n*.045,c=table.domain.map(domain_colours),alpha=.8,linewidths=.3,edgecolors='white');ax.axvline(0,color='.6',lw=.4)
 ax.set(yticks=y,yticklabels=table.broad_function,xlabel=f'Mean {pc}',title=f'Functional classes: {pc}')
for k,key in enumerate(['species','protein_family']):
 table=coords.groupby(key).agg(PC1=('PC1','mean'),PC2=('PC2','mean'),n=('Entry','size'),domain=('domain',lambda x:x.mode().iat[0])).reset_index();table.to_csv(source_folder(6)/f'{key}_centroids.csv',index=False)
 ax=axes[1,k];ax.scatter(table.PC1,table.PC2,s=3+table.n*.10,c=table.domain.map(domain_colours),alpha=.55,linewidths=.2,edgecolors='white');ax.set(xlabel=pc1,ylabel=pc2,title=f'{key.replace("_"," ").title()} centroids');ax.grid(color='.94',lw=.35);ax.set_axisbelow(True)
coords.to_csv(source_folder(6)/'underlying_protein_coordinates_and_groups.csv',index=False);save(fig,6)

# All primary GAMs use the same masks and full shared colour range.
fig,axes=plt.subplots(4,4,figsize=(180*mm,185*mm));fig.subplots_adjust(left=.08,right=.98,bottom=.12,top=.95,wspace=.24,hspace=.35)
grids=[]
for m in MODEL_ORDER:
 stem=m[:-6] if m.endswith('_score') else m
 grids.append(pd.read_csv(data/f'GAM_grids/{stem}_score_grid.csv'))
lo=float(np.floor(min(t.loc[t.supported,'predicted_score'].min() for t in grids)*2)/2);hi=float(np.ceil(max(t.loc[t.supported,'predicted_score'].max() for t in grids)*2)/2)
for k,(m,t) in enumerate(zip(MODEL_ORDER,grids)):
 ax=axes.flat[k];gx,gy,zz,opacity,contours=display_layers(t)
 mesh=ax.pcolormesh(gx,gy,zz,cmap='viridis',vmin=lo,vmax=hi,alpha=opacity,shading='nearest');ax.contour(gx,gy,contours,levels=np.linspace(lo,hi,9),colors='white',linewidths=.2,alpha=.6)
 ax.set(title=LABELS[m],xlim=(-6,6),ylim=(-6,6),xticks=[-4,0,4],yticks=[-4,0,4]);ax.set_box_aspect(1)
 if k%4:ax.set_yticklabels([])
for k in [14,15]:axes.flat[k].axis('off')
cax=fig.add_axes([.40,.057,.42,.012]);cb=fig.colorbar(mesh,cax=cax,orientation='horizontal');cb.set_label('Predicted within-model score z-score',labelpad=2)
fig.text(.02,.52,pc2,rotation=90,va='center',fontsize=6);fig.text(.23,.062,pc1,ha='center',fontsize=6)
(source_folder(7)/'plot_definition.json').write_text(json.dumps({'models':MODEL_ORDER,'shared_colour_limits':[lo,hi],'source':'GAM_grids/*_score_grid.csv','support':'at least 3 supporting proteins and inside observed convex hull; unsupported predictions at alpha 0.22, no contours','GAM_display_axes':{'PC1':[-6,6],'PC2':[-6,6]},'GAM_background_alpha':.22},indent=2)+'\n');save(fig,7)

importance=pd.read_csv(data/'recomputed/importance/property_importance_expanded.csv');importance=importance[importance.model.isin(names)&importance.property.isin(BIOPHYS_14)].copy();assert len(importance)==14*14
importance.to_csv(source_folder(8)/'displayed_importance_and_tests.csv',index=False)
pretty=['Length','Mass / residue','Isoelectric point','Acidic fraction','Basic fraction','Hydropathicity','Aromaticity','Instability index','Proline fraction','Ordered fraction','Helix–sheet contrast','Relative contact order','Mean Cβ distance','Surface exposure']
fig=plt.figure(figsize=(180*mm,135*mm))
for k,(column,title,cmap) in enumerate([('univariate_beta','Univariate association','RdBu_r'),('multivariate_beta','Joint coefficient','RdBu_r'),('rel_weight_pct','Relative weight (%)','viridis')]):
 pivot=importance.pivot(index='property',columns='model',values=column).reindex(index=BIOPHYS_14,columns=names);limit=float(np.abs(pivot).max().max())
 ax=fig.add_axes([.19+.265*k,.30,.245,.61]);mesh=ax.imshow(pivot,aspect='auto',cmap=cmap,vmin=0 if k==2 else -limit,vmax=limit,interpolation='none');ax.set(title=title,xticks=range(14),xticklabels=names,yticks=range(14),yticklabels=pretty if k==0 else ['']*14);ax.tick_params(length=0,pad=2);plt.setp(ax.get_xticklabels(),rotation=68,ha='right')
 if k==1:
  for j,feature in enumerate(BIOPHYS_14):
   for i,model in enumerate(names):
    r=importance[importance.property.eq(feature)&importance.model.eq(model)].iloc[0]
    if r.VIF>5:ax.add_patch(Rectangle((i-.5,j-.5),1,1,fill=False,hatch='///',edgecolor='.4',lw=0))
    if r.mv_p<.05:ax.plot(i,j,'.',color='black',ms=1.6)
 cax=fig.add_axes([.21+.265*k,.075,.20,.014]);fig.colorbar(mesh,cax=cax,orientation='horizontal')
fig.text(.5,.025,'Joint panel: hatching, VIF > 5; dots, unadjusted p < 0.05',ha='center',fontsize=6);save(fig,8)

matched=pd.read_csv(data/'inputs/finetune/surface_shift_matched.csv');assert matched.groupby('family').size().eq(10).all()
wide=matched.pivot(index='name',columns='family',values='case_minus_control');assert wide.notna().all().all() and len(wide)==10
summary=matched.groupby('family').case_minus_control.agg(['mean','sem','count']).reset_index();matched.to_csv(source_folder(12)/'per_target_observations.csv',index=False);summary.to_csv(source_folder(12)/'model_mean_sem.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(180*mm,85*mm));fig.subplots_adjust(left=.13,right=.97,bottom=.22,top=.88,wspace=.45)
families=list(wide.columns);assert set(families)=={'ProteinMPNN','ESM2-35M'}
axes[0].scatter(wide['ProteinMPNN'],wide['ESM2-35M'],s=16,color='#0072B2');axes[0].axhline(0,color='.6',lw=.4);axes[0].axvline(0,color='.6',lw=.4);axes[0].set(xlabel='ProteinMPNN steer',ylabel='ESM2-35M steer',title='a  Same ten secreted targets')
for i,family in enumerate(['ProteinMPNN','ESM2-35M']):
 values=matched.loc[matched.family.eq(family),'case_minus_control'];axes[1].scatter(i+np.linspace(-.12,.12,len(values)),values,s=12,c=['#D55E00','#0072B2'][i],alpha=.7);axes[1].errorbar(i,values.mean(),yerr=values.sem(),fmt='s',color='black',ms=3,capsize=2,lw=.7)
axes[1].axhline(0,color='.6',lw=.4);axes[1].set(xticks=[0,1],xticklabels=['ProteinMPNN','ESM2-35M'],ylabel='Fine-tuned minus matched-control\nsurface net charge',title='b  Target values and mean ± SEM',xlim=(-.5,1.5));save(fig,12)
print(f'Rendered Supplementary Figures S1, S2, S4-S8 and S12 in {out}; plotted tables in {export}')
