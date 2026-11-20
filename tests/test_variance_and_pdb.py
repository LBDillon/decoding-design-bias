"""Rank-aware variance calculations and matched experimental-PDB controls."""
import unittest

import numpy as np
import pandas as pd

from decoding_bias.analysis import pdb_control
from decoding_bias.analysis.variance import make_Q, r2_from_Q, ssr_from_Q

class TestVarianceAndPdb(unittest.TestCase):
    def test_duplicate_predictor_does_not_add_variance_or_degrees_of_freedom(self):
        x=np.array([-3.,-2.,-1.,0.,1.,2.,3.,4.])
        y=np.array([2.,-1.,0.,3.,1.,-2.,4.,-1.])
        q,p=make_Q(np.column_stack([x,2*x,np.ones(len(x))]))
        a=np.column_stack([np.ones(len(x)),x])
        residual=y-a@np.linalg.lstsq(a,y,rcond=None)[0]
        self.assertEqual(p,1)
        self.assertEqual(q.shape,(len(x),2))
        self.assertAlmostEqual(ssr_from_Q(y,q),float(residual@residual),places=11)
        expected=1-float(residual@residual)/float(((y-y.mean())**2).sum())
        self.assertAlmostEqual(r2_from_Q(y,q,p)[1],1-(1-expected)*(len(x)-1)/(len(x)-2),places=11)


    def test_family_species_aliases_give_order_invariant_species_increment(self):
        fam=np.repeat(np.arange(6),4)
        species=np.repeat(np.array([0,0,1,1,2,2]),4)
        rng=np.random.default_rng(40)
        z=rng.normal(size=(24,2));y=rng.normal(size=24)
        fd=np.eye(6)[fam][:,1:];sd=np.eye(3)[species][:,1:]
        reduced=np.column_stack([z,fd]);full=np.column_stack([z,fd,sd])
        qr,pr=make_Q(reduced);qf,pf=make_Q(full)
        qrev,prev=make_Q(full[:,::-1])
        self.assertEqual(pr,pf)
        self.assertEqual(pf,prev)
        self.assertAlmostEqual(ssr_from_Q(y,qr),ssr_from_Q(y,qf),places=11)
        self.assertAlmostEqual(ssr_from_Q(y,qf),ssr_from_Q(y,qrev),places=11)


    def test_matched_sample_is_exact_reproducible_and_never_pads_missing_domains(self):
        pool=pd.DataFrame({'Entry':[f'p{i}' for i in range(18)],
                           'domain':['Archaea']*6+['Bacteria']*6+['Eukaryota']*6})
        counts={'Eukaryota':3,'Bacteria':4,'Archaea':2}
        a=pdb_control.matched_sample(pool,counts,1000)
        b=pdb_control.matched_sample(pool,counts,1000)
        self.assertEqual(a.Entry.to_list(),b.Entry.to_list())
        self.assertEqual(a.domain.value_counts().to_dict(),counts)
        self.assertFalse(a.Entry.duplicated().any())
        with self.assertRaises(ValueError):
            pdb_control.matched_sample(pool,{'Archaea':7},1000)


    def test_score_missingness_is_recorded_in_actual_cohort(self):
        from decoding_bias.catalog import BIOPHYS_14
        rng=np.random.default_rng(18);n=80
        frame=pd.DataFrame(rng.normal(size=(n,14)),columns=BIOPHYS_14)
        frame['Entry']=[f'p{i}' for i in range(n)]
        frame['domain']=np.tile(['Archaea','Bacteria','Eukaryota','Bacteria'],20)
        frame['protein_family']=np.tile(['f0','f1','f2','f3'],20)
        frame['species']=np.repeat([f's{i}' for i in range(8)],10)
        frame['score']=rng.normal(size=n);frame.loc[:6,'score']=np.nan
        result,ranks,members=pdb_control.fit_table(frame,{'model':('score','sequence')})
        self.assertEqual(int(result.iloc[0]['n']),73)
        self.assertEqual(set(members.Entry),set(frame.loc[frame.score.notna(),'Entry']))
        self.assertTrue((ranks['rank']<=ranks.nominal_columns).all())


    def test_numerically_zero_species_effects_do_not_count_as_direction_flips(self):
        from decoding_bias.analysis.variance import species_effect_collapse, make_Q
        y=np.array([-1.,-1.,1.,1.]);Q,p=make_Q(np.array([[0.],[0.],[1.],[1.]]))
        bases={'Family':(Q,p),'Biophys+Family':(Q,p)}
        got=species_effect_collapse(y,np.arange(4),4,bases)
        self.assertEqual(got['species_effect_signflip_given_family'],0.)
        self.assertEqual(got['species_effect_signflip_given_family_biophys'],0.)

