"""CPU synthetic parity, durable resume, and checkpoint failure tests."""
import copy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from athena_research import AthenaData
from athena_research.core.base import cupy_enabled
from athena_research.operations.structure_functions import get_sf_mb
from athena_research.operations.sf_checkpoint import (
    checkpointed_sf,pair_work,pair_ordinal,
)


def save_checkpoint(ad,path,label):
    """Example caller-owned atomic callback, independent of problem drivers."""
    fd,temporary=tempfile.mkstemp(dir=path.parent,suffix='.h5data')
    os.close(fd)
    try:
        ad.save(str(temporary))
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


class ToyData:
    def __init__(self,n=4):
        self.n_mbs=n;self.nx1=self.nx2=self.nx3=2;self.time=1.
        self.x1min=self.x2min=self.x3min=0.
        self.x1max=float(n);self.x2max=self.x3max=1.
        self.mb_geometry=np.array([[i,i+1,0,1,0,1] for i in range(n)],float)
        self._header={'mesh':{'ix1_bc':'user','ix2_bc':'periodic','ix3_bc':'periodic'}}
        self.sf={'old_sf':{'tag':'keep'}};self.dist={'old_pdf':{'tag':'keep'}}
        rng=np.random.default_rng(7)
        self.fields={k:rng.uniform(1,2,(n,2,2,2)) for k in ('dens','velx')}
        self.fields['mask']=np.ones((n,2,2,2));self.fields['mask'][2]=0.
        self.calls=[]

    def data(self,key,mbl=None,mbh=None):
        if mbl is None or mbh is None:
            raise AssertionError('unbounded meshblock access')
        self.calls.append((key,mbl,mbh))
        return self.fields[key][mbl:mbh]


SETTINGS=dict(weights='mask',npairs=4096,nbins=8,max_order=2,simultaneous_blocks=2)


@unittest.skipIf(cupy_enabled,'CPU fixture suite; use validate_sf_checkpoint_gpu.py for CUDA')
class CheckpointTests(unittest.TestCase):
    def test_explicit_keys_selection_and_source_are_frozen(self):
        settings=dict(SETTINGS,product_keys=['density_probe','velocity_probe'],
                      selection_metadata={'mask_definition':'toy-v1'})
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source';source.write_text('original')
            ad=ToyData()
            self.assertFalse(checkpointed_sf(ad,'probe',['dens','velx'],source=source,
                checkpoint=lambda *a:None,stop_after_pairs=1,**settings))
            changed=dict(settings,selection_metadata={'mask_definition':'toy-v2'})
            with self.assertRaises(ValueError):
                checkpointed_sf(ad,'probe',['dens','velx'],source=source,checkpoint=lambda *a:None,**changed)
            changed=dict(settings,product_keys=['density_other','velocity_probe'])
            with self.assertRaises(ValueError):
                checkpointed_sf(ad,'probe',['dens','velx'],source=source,checkpoint=lambda *a:None,**changed)
            source.write_text('changed-length')
            with self.assertRaises(ValueError):
                checkpointed_sf(ad,'probe',['dens','velx'],source=source,checkpoint=lambda *a:None,**settings)

    def test_generic_publication_and_mpi_guard(self):
        ad=ToyData()
        self.assertTrue(checkpointed_sf(ad,'probe',['dens','velx'],
            product_keys=['density_probe','velocity_probe'],checkpoint=lambda *a:None,**SETTINGS))
        self.assertIn('density_probe',ad.sf)
        other=ToyData();other.has_full_data=False
        with self.assertRaises(ValueError):
            checkpointed_sf(other,'probe',['dens','velx'],checkpoint=lambda *a:None,**SETTINGS)

    def test_pair_coverage_and_cursor_suffix(self):
        for n in range(1,9):
            for batch in range(1,n+1):
                work=list(pair_work(n,batch))
                ids=[pair_ordinal(i,j,n) for a,b,i,j,_ in work]
                self.assertEqual(sorted(ids),list(range(n*(n+1)//2)))
                for index,item in enumerate(work):
                    self.assertEqual(list(pair_work(n,batch,item[-1])),work[index+1:])

    def test_legacy_and_repeated_resumes(self):
        legacy=ToyData();r,sums,w,c=get_sf_mb(legacy,['dens','velx'],**SETTINGS)
        fresh=ToyData();checkpointed_sf(fresh,'hot',['dens','velx'],checkpoint=lambda *a:None,**SETTINGS)
        resumed=ToyData();generations=[]
        while True:
            durable=[]
            done=checkpointed_sf(resumed,'hot',['dens','velx'],stop_after_pairs=2,
                checkpoint=lambda ad,label:durable.append(copy.deepcopy(ad.__dict__)),**SETTINGS)
            self.assertTrue(durable)
            state=durable[-1]
            generations.append(state['sf_checkpoints']['hot']['generation'])
            self.assertEqual(state['dist'],{'old_pdf':{'tag':'keep'}})
            self.assertEqual(state['sf']['old_sf'],{'tag':'keep'})
            if not done:
                self.assertNotIn('dens_hot',state['sf'])
            resumed=ToyData();resumed.__dict__.update(state)
            if done:
                break
        self.assertEqual(generations,sorted(set(generations)))
        for i,var in enumerate(['dens','velx']):
            for ad in (fresh,resumed):
                entry=ad.sf[var+'_hot']
                np.testing.assert_allclose(entry['r'],r,rtol=1e-10,atol=1e-12)
                np.testing.assert_allclose(entry['sf'],sums[i],rtol=1e-10,atol=1e-12)
                np.testing.assert_array_equal(entry['num_bin_points'],c)
                np.testing.assert_allclose(entry['weight_sum_bin'],w,rtol=1e-10,atol=1e-12)
        self.assertTrue(resumed.calls)

    def test_reject_changed_settings_source_and_bitmap(self):
        ad=ToyData();checkpointed_sf(ad,'hot',['dens','velx'],checkpoint=lambda *a:None,stop_after_pairs=1,**SETTINGS)
        for change in ('settings','geometry','bitmap','batch'):
            other=copy.deepcopy(ad);kw=dict(SETTINGS)
            if change=='settings':kw['npairs']=8192
            if change=='geometry':other.mb_geometry[0,0]=-.1
            if change=='bitmap':other.sf_checkpoints['hot']['completed_bitmap'][0]=0
            if change=='batch':kw['simultaneous_blocks']=3
            with self.assertRaises(ValueError):
                checkpointed_sf(other,'hot',['dens','velx'],checkpoint=lambda *a:None,**kw)

    def test_stop_before_work_does_not_publish_sf(self):
        ad=ToyData();saved=[]
        self.assertFalse(checkpointed_sf(ad,'hot',['dens','velx'],checkpoint=lambda ad,label:saved.append(label),
            should_stop=lambda:True,**SETTINGS))
        self.assertEqual(ad.sf_checkpoints['hot']['completed_pairs'],0)
        self.assertNotIn('dens_hot',ad.sf)
        self.assertEqual(saved,['SF hot partial'])

    def test_serialized_partial_resume_and_atomic_save_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'toy.h5data'
            toy=ToyData();ad=AthenaData();ad.time=toy.time;ad.sf=toy.sf;ad.dist=toy.dist
            def persist(model,label):
                ad.sf=model.sf;ad.dist=model.dist;ad.sf_checkpoints=model.sf_checkpoints
                save_checkpoint(ad,path,label)
            checkpointed_sf(toy,'hot',['dens','velx'],checkpoint=persist,stop_after_pairs=2,**SETTINGS)
            initial=path.read_bytes();checked=AthenaData().load(str(path),config=False)
            self.assertNotIn('dens_hot',checked.sf)
            resumed=ToyData();resumed.sf_checkpoints=checked.sf_checkpoints
            with patch.object(ad,'save',side_effect=RuntimeError('injected failure')):
                with self.assertRaises(RuntimeError):
                    save_checkpoint(ad,path,'failure')
            self.assertEqual(path.read_bytes(),initial)
            self.assertTrue(checkpointed_sf(resumed,'hot',['dens','velx'],checkpoint=persist,**SETTINGS))
            checked=AthenaData().load(str(path),config=False)
            self.assertIn('dens_hot',checked.sf)
            self.assertEqual(checked.dist['old_pdf']['tag'],'keep')


if __name__=='__main__':
    unittest.main()
