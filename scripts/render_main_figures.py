"""Draw Figures 2 and 3 from the Source Data and export the plotted tables.

The exported tables are written under OUTPUT_DIR/source_data; the Source Data
directory is only read.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from decoding_bias.analysis.figure_data import elo_panels, MODEL_ORDER, LABELS, CLASSES

parser=argparse.ArgumentParser()
parser.add_argument('--source-data',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
data=args.source_data.resolve(); out=args.output_dir.resolve()
export=out/'source_data'
out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':6,'axes.titlesize':7,'axes.labelsize':6,
                    'xtick.labelsize':5.6,'ytick.labelsize':5.6,'legend.fontsize':5.6,
                    'axes.linewidth':.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                    'savefig.facecolor':'white'})
mm=1/25.4
dom={'Archaea':'#D55E00','Bacteria':'#0072B2','Eukaryota':'#009E73'}
context={'Backbone-conditioned':'#2C7BB6','Structure + native sequence':'#7B3294','Sequence-only':'#1A9641'}
markers={'Backbone-conditioned':'o','Structure + native sequence':'s','Sequence-only':'^'}

ratings=pd.read_csv(data/'Figure_2/all_models_species_ratings_long.csv')
tax=pd.read_csv(data/'inputs/taxonomy.csv')
gaps,species,matrix,meta,z=elo_panels(ratings,tax)
assert ratings.species.nunique()==495 and len(gaps)==14
assert set(ratings.model)==set(MODEL_ORDER)
f2=export/'Figure_2';f2.mkdir(parents=True,exist_ok=True)
gaps.to_csv(f2/'panel_a_gap_and_2000_species_bootstrap_intervals.csv',index=False)
species.to_csv(f2/'panel_c_displayed_species_order.csv',index=False)
matrix.to_csv(f2/'panel_b_mean_species_z_by_phylum.csv')
meta.to_csv(f2/'panel_b_phylum_order_and_counts.csv',index=False)
z.to_csv(f2/'panel_b_species_z_before_aggregation.csv',index=False)
means=species.groupby(['model','domain']).agg(mean_elo=('rating','mean'),n_species=('species','nunique')).reset_index()
means['mean_minus_1500']=means.mean_elo-1500
means.to_csv(f2/'panel_c_domain_means.csv',index=False)
(f2/'plot_definition.json').write_text(json.dumps({'analysis':'unweighted Elo','n_species':495,'n_families':281,
 'n_species_by_model':ratings.groupby('model').species.nunique().to_dict(),
 'elo_permutations':50,'elo_seed':42,'bootstrap_seed':0,'bootstrap_draws':2000,
 'bootstrap_unit':'species; independently within Archaea and Eukaryota; conditional on fitted Elo ratings',
 'bootstrap_interval':'2.5 and 97.5 percentiles; no Elo refitting within bootstrap',
 'heatmap':'within-model species-level population z-scores, then equal-species mean within phylum',
 'phylum_order':'Eukaryota, Archaea, Bacteria; descending across-model cell mean within domain',
 'species_order':'Archaea, Bacteria, Eukaryota; ascending Elo within each domain',
 'model_order':MODEL_ORDER,'domain_colours':dom},indent=2)+'\n')

fig=plt.figure(figsize=(180*mm,185*mm))
fig.text(.012,.99,'a',weight='bold',fontsize=7,va='top')
fig.text(.06,.99,'Archaea − Eukaryota Elo gap',fontsize=7,weight='bold',va='top')
ax=fig.add_axes([.145,.58,.325,.35])
for i,r in gaps.iterrows():
 ax.errorbar(r.gap,i,xerr=[[r.gap-r.ci_lower],[r.ci_upper-r.gap]],fmt=markers[r.scoring_context],
             color=context[r.scoring_context],markersize=3,capsize=1.6,lw=.65)
ax.axvline(0,color='.5',ls='--',lw=.5)
ax.set(yticks=range(14),yticklabels=gaps.label,xlim=(-700,650),ylim=(13.7,-.7),xlabel='Archaea − Eukaryota Elo gap')
ax.set_xticks([-600,-300,0,300,600]);ax.tick_params(axis='y',length=0,pad=3)
for tick,(_,r) in zip(ax.get_yticklabels(),gaps.iterrows()):tick.set_color(context[r.scoring_context])
for j,(label,colour) in enumerate(context.items()):
 fig.text(.045,.538-j*.015,{'Backbone-conditioned':'● Backbone-conditioned','Structure + native sequence':'■ Structure + native sequence','Sequence-only':'▲ Sequence-only'}[label],color=colour,fontsize=5.6)

fig.text(.012,.492,'c',weight='bold',fontsize=7,va='top')
fig.text(.06,.492,'Species-level Elo ratings',weight='bold',fontsize=7,va='top')
for j,model in enumerate(['proteinmpnn_score','protgpt2_score']):
 ax=fig.add_axes([.11,.28-j*.225,.36,.17]);sub=species[species.model.eq(model)]
 for domain in ['Archaea','Bacteria','Eukaryota']:
  s=sub[sub.domain.eq(domain)];mean=s.rating_minus_1500.mean()
  ax.vlines(s.plot_position,0,s.rating_minus_1500,color=dom[domain],lw=.35,alpha=.65)
  ax.hlines(mean,s.plot_position.min(),s.plot_position.max(),color=dom[domain],ls='--',lw=.7)
  ax.text(s.plot_position.mean(),-785,f'{domain}\nμ={mean:+.0f}',color=dom[domain],fontsize=5.5,ha='center',va='bottom')
 ax.axhline(0,color='.4',lw=.5);ax.set(xlim=(-2,496),ylim=(-820,900),yticks=[-600,0,600],xticks=[],ylabel='Elo − 1500',title=LABELS[model])
 ax.grid(axis='y',color='.92',lw=.4);ax.set_axisbelow(True)
 if j==1:ax.set_xlabel('Species grouped by domain, sorted within domain',labelpad=4)

fig.text(.51,.99,'b',weight='bold',fontsize=7,va='top')
fig.text(.58,.99,'Phylum Elo patterns',weight='bold',fontsize=7,va='top')
ax=fig.add_axes([.70,.15,.275,.795]);limit=float(np.abs(matrix).max().max())
mesh=ax.imshow(matrix.values,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='none')
ax.set(yticks=range(len(meta)),yticklabels=meta.phylum,xticks=range(14),xticklabels=[LABELS[m] for m in matrix.columns])
ax.tick_params(axis='y',length=0,pad=2);ax.tick_params(axis='x',length=0,pad=3,labelrotation=68)
for tick,domain in zip(ax.get_yticklabels(),meta.domain):tick.set_color(dom[domain])
for tick,model in zip(ax.get_xticklabels(),matrix.columns):tick.set_color(context[CLASSES[model]]);tick.set_ha('right')
for k in range(1,len(meta)):
 if meta.domain.iloc[k]!=meta.domain.iloc[k-1]:ax.axhline(k-.5,color='.25',lw=.7)
ax.set_xticks(np.arange(-.5,14,1),minor=True);ax.set_yticks(np.arange(-.5,len(meta),1),minor=True)
ax.grid(which='minor',color='white',lw=.25);ax.tick_params(which='minor',length=0)
cax=fig.add_axes([.72,.044,.23,.012]);cb=fig.colorbar(mesh,cax=cax,orientation='horizontal');cb.set_ticks([-3,-1.5,0,1.5,3]);cb.set_label('Mean species Elo z-score',labelpad=1)
fig.savefig(out/'Figure_2.pdf');plt.close(fig)

# All Figure 3 panels use the R sample-scaled global PCA frame from the Source Data.
natural=pd.read_csv(data/'inputs/main_analysis.csv')
coords=pd.read_csv(data/'GAM_grids/global_pca_coordinates.csv').merge(natural[['Entry','domain','species','protein_name','sequence_length','avg_plddt']],on='Entry',validate='one_to_one')
assert len(coords)==len(natural)
variance_table=pd.read_csv(data/"recomputed/pca/pca_variance.csv")
transform=pd.read_csv(data/'GAM_grids/global_pca_transform.csv')
landmarks={'P0C0S5':('Histone H2A.Z',(-4.2,6.2)),'Q8YHN9':('Ribosomal protein uL4',(-4.1,3.9)),
 'P00355':('GAPDH',(-4.4,.6)),'P00512':('Phosphofructokinase',(-3.4,-3.5)),
 'P00942':('Triosephosphate isomerase',(-1.7,-5.8)),'P00408':('Cytochrome c oxidase II',(3.1,-4.8)),
 'P07437':('Beta-tubulin',(4.4,3.0)),'O73947':('PCNA sliding clamp',(4.5,-1.0)),
 'P62149':('Calmodulin',(1.7,6.2)),'O93627':('RuBisCO',(-.9,2.2))}
callouts=coords[coords.Entry.isin(landmarks)].copy();assert len(callouts)==10
assert (callouts.avg_plddt>=70).all() and callouts.sequence_length.between(50,600).all()
callouts['label']=callouts.Entry.map(lambda x:landmarks[x][0]);callouts['label_x']=callouts.Entry.map(lambda x:landmarks[x][1][0]);callouts['label_y']=callouts.Entry.map(lambda x:landmarks[x][1][1])
folder=export/'Figure_3bc_and_Table_S14';folder.mkdir(parents=True,exist_ok=True)
variance_table.to_csv(folder/'pca_variance.csv',index=False)
callouts.to_csv(folder/'landmark_coordinates_and_labels.csv',index=False)
coords.to_csv(folder/'exact_global_R_coordinates_and_metadata.csv',index=False)
loading=transform.copy();loading['magnitude']=np.hypot(loading.PC1,loading.PC2);loading=loading.nlargest(8,'magnitude').copy()
scale=float(coords[['PC1','PC2']].abs().max().max()*.45/loading.magnitude.max())
loading['arrow_scale']=scale;loading['display_x']=loading.PC1*scale;loading['display_y']=loading.PC2*scale
loading.to_csv(folder/'top_feature_loading_arrows.csv',index=False)
# reconstruct() draws Figure 3 from the Source Data copies of the tables exported above.
from decoding_bias.analysis.figure3 import reconstruct
report=reconstruct(data, out)
(folder/'figure3_display_parameters.json').write_text(json.dumps({'global_scaling_ddof':1,'GAM_models':report['models'],'GAM_shared_colour_limits':[-3,1.5],'GAM_support':'at least 3 proteins in density bin and inside observed convex hull; unsupported predictions at alpha 0.22, no contours','axes':{'PC1':[-6,6],'PC2':[-7,8]},'loading_arrow_scale':scale,'landmark_selection':'manually chosen visual landmarks, not class representatives; pLDDT >=70 and length 50–600',
 'loading_density_alpha':.45,'loading_density_edge_gray':.84,'loading_origin_marker':{'x':0,'y':0,'colour':'black'},
 'GAM_display_axes':{'PC1':[-6,6],'PC2':[-6,6]},'GAM_background_alpha':.22},indent=2)+'\n')
print(f'Rendered Figures 2 and 3 in {out}; plotted tables in {export}')

