"""Tiny real-HDF5/fresh-process CUDA resume gate; no scientific data required."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['ATHENA_RESEARCH_CPU_ONLY']='0'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from athena_research import AthenaData
from athena_research.core.base import xp,asnumpy,cupy_enabled
from athena_research.operations.structure_functions import get_sf_mb
from athena_research.operations.sf_checkpoint import checkpointed_sf
from test_sf_checkpoint import ToyData,save_checkpoint


class GPUData(ToyData):
    def data(self,*args,**kwargs):
        return xp.asarray(super().data(*args,**kwargs))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--child',choices=('partial','resume'))
    args=parser.parse_args()
    if not cupy_enabled:raise SystemExit('CuPy/GPU backend required; refuse CPU fallback')
    args.work.mkdir(parents=True,exist_ok=True);path=args.work/'toy.h5data'
    variables=['dens','velx'];settings=dict(weights='mask',npairs=4096,nbins=8,
                                           max_order=3,simultaneous_blocks=2)
    ad=GPUData()
    if args.child:
        if path.exists():
            loaded=AthenaData().load(str(path),config=False)
            ad.sf=loaded.sf;ad.dist=loaded.dist;ad.sf_checkpoints=loaded.sf_checkpoints
        def save(model,label):
            stored=AthenaData();stored.time=model.time;stored.sf=model.sf;stored.dist=model.dist
            stored.sf_checkpoints=model.sf_checkpoints;save_checkpoint(stored,path,label)
        done=checkpointed_sf(ad,'gpu_probe',variables,checkpoint=save,
            stop_after_pairs=2 if args.child=='partial' else None,**settings)
        (args.work/(args.child+'.json')).write_text(json.dumps(dict(
            pid=os.getpid(),backend='cupy',complete=done,
            pairs=int(ad.sf_checkpoints['gpu_probe']['completed_pairs'])))+'\n')
        return 0 if done else 75
    if path.exists():raise SystemExit('use a fresh work directory; no overwrite')
    baseline=[asnumpy(v) for v in get_sf_mb(ad,variables,**settings)]
    command=[sys.executable,str(Path(__file__).resolve()),'--work',str(args.work)]
    env=dict(os.environ,ATHENA_RESEARCH_CPU_ONLY='0')
    for mode,expected in (('partial',75),('resume',0)):
        result=subprocess.run(command+['--child',mode],env=env,check=False)
        if result.returncode!=expected:raise RuntimeError((mode,result.returncode))
    loaded=AthenaData().load(str(path),config=False)
    errors=[]
    for i,var in enumerate(variables):
        entry=loaded.sf[var+'_gpu_probe']
        for name,reference in zip(('r','sf','weight_sum_bin','num_bin_points'),
                                 (baseline[0],baseline[1][i],baseline[2],baseline[3])):
            if name=='sf':np.testing.assert_allclose(entry[name],reference,rtol=1e-9,atol=1e-11)
            else:np.testing.assert_array_equal(entry[name],reference)
            errors.append(float(np.max(np.abs(entry[name]-reference))))
    assert loaded.sf['old_sf']['tag']=='keep' and loaded.dist['old_pdf']['tag']=='keep'
    records={mode:json.loads((args.work/(mode+'.json')).read_text()) for mode in ('partial','resume')}
    assert records['partial']['pid']!=records['resume']['pid']
    assert records['partial']['pairs']==2 and records['resume']['pairs']==10
    report=dict(passed=True,backend='cupy',completed_pairs=10,
        accepted_pairs=int(np.asarray(baseline[3]).sum()),children=records,max_absolute_delta=max(errors))
    (args.work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
