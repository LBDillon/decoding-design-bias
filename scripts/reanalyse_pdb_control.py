"""Reanalyse the experimental-PDB control with rank-aware fits and exact cohorts.

No model scores are generated or modified. Defaults use the deposited input files.
"""
from pathlib import Path
import argparse,hashlib,json,platform,sys
import numpy as np
import pandas as pd
import scipy
from scipy import stats
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from decoding_bias.analysis import pdb_control as pc


def legacy_basis(x):
    a=np.column_stack([np.ones(len(x)),x]);q,_=np.linalg.qr(a)
    return q,a.shape[1]-1


def pool_without_ribosomal(frame):
    return frame.loc[~frame.broad_function.fillna('').str.contains('ribosom',case=False)].copy()


def write_fit(frame,panel,out,prefix,common=False,basis_function=pc.make_Q):
    values,ranks,members=pc.fit_table(frame,panel,common=common,basis_function=basis_function)
    values.to_csv(out/f'{prefix}_estimates.csv',index=False)
    ranks.to_csv(out/f'{prefix}_design_ranks.csv',index=False)
    members.to_csv(out/f'{prefix}_members.csv',index=False)
    return values


def correlation_rows(pdb,matched,full,scenario):
    rows=[]
    for metric in ['residual_species_raw','residual_species_adj','species_effect_retention_given_family_biophys']:
        for other,label in [(matched,'matched_AF_mean'),(full,'full_AF')]:
            left=pdb.set_index('model')[metric];right=other.set_index('model')[metric]
            pair=pd.concat([left.rename('pdb'),right.rename('af')],axis=1).dropna()
            a=stats.pearsonr(pair.pdb,pair.af);b=stats.spearmanr(pair.pdb,pair.af)
            rows.append({'scenario':scenario,'comparison':label,'metric':metric,'n_models':len(pair),
                         'pearson_r':float(a.statistic),'pearson_p_nominal':float(a.pvalue),
                         'spearman_rho':float(b.statistic),'spearman_p_nominal':float(b.pvalue)})
    return rows


def summarize_draws(values):
    columns=['n','n_archaea','n_bacteria','n_eukaryota','residual_species_raw','residual_species_adj',
             'species_effect_retention_given_family_biophys']
    rows=[]
    for model,group in values.groupby('model',sort=False):
        row={'model':model,'draws':group.draw.nunique()}
        for col in columns:
            row[col]=float(group[col].mean())
            row[col+'_sd']=float(group[col].std(ddof=0))
            row[col+'_q025']=float(group[col].quantile(.025))
            row[col+'_q975']=float(group[col].quantile(.975))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--main-input',type=Path,required=True)
    parser.add_argument('--pdb-input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--historical-main-input',type=Path)
    parser.add_argument('--archived-matched-input',type=Path)
    parser.add_argument('--draws',type=int,default=30)
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    original=pd.read_csv(args.historical_main_input or args.main_input,low_memory=False)
    main_input=pd.read_csv(args.main_input,low_memory=False)
    main_input=main_input.loc[~main_input.Entry.isin(['P19949','Q8TFG7'])].copy()
    pdb=pd.read_csv(args.pdb_input,low_memory=False)
    main_input.to_csv(out/'AF_analysis_input_excluded.csv',index=False)
    pdb.to_csv(out/'PDB_analysis_input.csv',index=False)
    provenance={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
                'draws':args.draws,'seeds':list(range(1000,1000+args.draws)),'excluded_AF_ids':['P19949','Q8TFG7'],
                'n_AF_after_exclusions':len(main_input),'n_PDB_universe':len(pdb),
                'panel':pc.PANEL,'excluded_unverified_duplicate':'SolubleCaliby',
                'data_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.main_input,args.pdb_input]}}
    # Isolate the kernel correction on exactly the same PDB response/input rows.
    pdb_correct=write_fit(pdb,pc.PANEL_WITH_DUPLICATE,out,'pdb_rank_corrected')
    pdb_legacy=write_fit(pdb,pc.PANEL_WITH_DUPLICATE,out,'pdb_legacy_qr',basis_function=legacy_basis)
    full=write_fit(main_input,pc.PANEL,out,'AF_full_rank_corrected')
    before_after=pdb_legacy[['model','n','residual_species_raw','residual_species_adj']].merge(
        pdb_correct[['model','residual_species_raw','residual_species_adj','rank_Full']],on='model',suffixes=('_legacy_qr','_corrected'))
    before_after['raw_change_from_kernel_only']=before_after.residual_species_raw_corrected-before_after.residual_species_raw_legacy_qr
    before_after.to_csv(out/'PDB_legacy_QR_vs_rank_corrected_same_inputs.csv',index=False)
    print('Corrected PDB and full AF fits finished',flush=True)
    # Replay the original sampling rule. Both score panels retain the duplicate for
    # exact historical complete-case selection, never as an independent comparison.
    historical_pool=pool_without_ribosomal(original)
    target={'Eukaryota':396,'Bacteria':364,'Archaea':116}
    replay_values=[];replay_corrected=[];replay_members=[];replay_ranks=[]
    seed0=pc.matched_sample(historical_pool,target,1000)
    if args.archived_matched_input:
        archived=pd.read_csv(args.archived_matched_input)
        provenance['seed1000_membership_matches_archived']=set(seed0.Entry)==set(archived.Entry)
        provenance['seed1000_order_matches_archived']=seed0.Entry.to_list()==archived.Entry.to_list()
        plot_columns=[c for model,(c,_) in pc.PANEL_WITH_DUPLICATE.items()
                      if model not in ['ESM3-struct','ESM3-seq','ESM2-15B']]
        plot_features=pc.BIOPHYS_14+['charge_at_ph7','small_residue_fraction']
        plot_pool=historical_pool.dropna(subset=plot_features+plot_columns+['species','protein_family'])
        plot_sample=pc.matched_sample(plot_pool,target,1000)
        provenance['archived_single_plot_cohort']={
            'matches_prefiltered_ten_model_pool_seed1000':set(plot_sample.Entry)==set(archived.Entry),
            'pool_n':len(plot_pool),'n':len(plot_sample),
            'explanation':'The single plotting cohort prefilters 16 features and ten score columns before sampling; the historical 30-draw control samples first, then filters all 13 scores.'}
    for draw,seed in enumerate(range(1000,1000+args.draws)):
        sampled=pc.matched_sample(historical_pool,target,seed)
        complete=pc.complete_frame(sampled,[c for c,_ in pc.PANEL_WITH_DUPLICATE.values()])
        for kernel,storage in [(legacy_basis,replay_values),(pc.make_Q,replay_corrected)]:
            vals,ranks,members=pc.fit_table(complete,pc.PANEL_WITH_DUPLICATE,common=True,basis_function=kernel)
            storage.append(vals.assign(draw=draw,seed=seed))
            if kernel is pc.make_Q:
                replay_ranks.append(ranks.assign(draw=draw,seed=seed))
        replay_members.append(sampled[['Entry','domain']].assign(draw=draw,seed=seed,selected=True,complete_case=sampled.Entry.isin(complete.Entry)))
    old=pd.concat(replay_values);new=pd.concat(replay_corrected)
    old.to_csv(out/'historical_rule_legacy_draw_estimates.csv',index=False)
    new.to_csv(out/'historical_rule_corrected_draw_estimates.csv',index=False)
    summarize_draws(old).to_csv(out/'historical_rule_legacy_summary.csv',index=False)
    summarize_draws(new).to_csv(out/'historical_rule_corrected_summary.csv',index=False)
    pd.concat(replay_members).to_csv(out/'historical_rule_all_draw_members.csv',index=False)
    pd.concat(replay_ranks).to_csv(out/'historical_rule_corrected_design_ranks.csv',index=False)
    print('Historical 30-draw replay finished',flush=True)
    # Primary controls: prefilter by each model's available score and match the
    # exact scored PDB cohort's domain marginal, not just the nominal 876 chains.
    af_pool=pool_without_ribosomal(main_input)
    pdb_unique=pdb_correct.loc[pdb_correct.model.isin(pc.PANEL)].copy()
    primary_values=[];primary_ranks=[];primary_members=[]
    for model,spec in pc.PANEL.items():
        pdb_sub=pc.complete_frame(pdb,[spec[0]])
        eligible=pc.complete_frame(af_pool,[spec[0]])
        counts={dom:int(pdb_sub.domain.eq(dom).sum()) for dom in pc.DOMAIN_ORDER}
        for draw,seed in enumerate(range(1000,1000+args.draws)):
            sampled=pc.matched_sample(eligible,counts,seed)
            vals,ranks,members=pc.fit_table(sampled,{model:spec})
            primary_values.append(vals.assign(draw=draw,seed=seed))
            primary_ranks.append(ranks.assign(draw=draw,seed=seed))
            primary_members.append(members.assign(draw=draw,seed=seed))
        print(f'Matched {model}: PDB and AF n={len(pdb_sub)}',flush=True)
    primary=pd.concat(primary_values);summary=summarize_draws(primary)
    primary.to_csv(out/'primary_matched_draw_estimates.csv',index=False)
    summary.to_csv(out/'primary_matched_summary.csv',index=False)
    pd.concat(primary_ranks).to_csv(out/'primary_matched_design_ranks.csv',index=False)
    pd.concat(primary_members).to_csv(out/'primary_matched_draw_members.csv',index=False)
    common_pdb=pc.complete_frame(pdb,[c for c,_ in pc.PANEL.values()])
    common_pool=pc.complete_frame(af_pool,[c for c,_ in pc.PANEL.values()])
    common_pdb_est=write_fit(common_pdb,pc.PANEL,out,'common_PDB',common=True)
    common_values=[];common_ranks=[];common_members=[]
    counts={dom:int(common_pdb.domain.eq(dom).sum()) for dom in pc.DOMAIN_ORDER}
    for draw,seed in enumerate(range(1000,1000+args.draws)):
        sample=pc.matched_sample(common_pool,counts,seed)
        vals,ranks,members=pc.fit_table(sample,pc.PANEL,common=True)
        common_values.append(vals.assign(draw=draw,seed=seed))
        common_ranks.append(ranks.assign(draw=draw,seed=seed))
        common_members.append(members.assign(draw=draw,seed=seed))
    common=pd.concat(common_values);common_summary=summarize_draws(common)
    common.to_csv(out/'common_matched_draw_estimates.csv',index=False)
    common_summary.to_csv(out/'common_matched_summary.csv',index=False)
    pd.concat(common_ranks).to_csv(out/'common_matched_design_ranks.csv',index=False)
    pd.concat(common_members).to_csv(out/'common_matched_draw_members.csv',index=False)
    provenance['common_PDB_n']=len(common_pdb);provenance['common_PDB_domain_counts']=counts
    comparisons=correlation_rows(pdb_unique,summary,full,'per_model_matched')
    comparisons+=correlation_rows(common_pdb_est,common_summary,full,'shared_PDB_complete_case')
    pd.DataFrame(comparisons).to_csv(out/'model_vector_correlations.csv',index=False)
    combined=pdb_unique.merge(summary,on='model',suffixes=('_PDB','_AFmatched')).merge(full[['model','residual_species_raw','residual_species_adj']],on='model')
    combined=combined.rename(columns={'residual_species_raw':'residual_species_raw_AFfull','residual_species_adj':'residual_species_adj_AFfull'})
    combined.to_csv(out/'primary_comparison.csv',index=False)
    (out/'analysis_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Common-case sensitivity and comparisons finished',flush=True)
    print(pdb_unique[['model','n','residual_species_raw','residual_species_adj']].round(6).to_string(index=False))

if __name__=='__main__':main()
