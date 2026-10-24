"""Opt-in serial CPU/single-GPU SF block-pair checkpoints.

Reuse legacy pair sampling and kernels. The caller supplies an atomic save
callback, serializes cache writers, and requests cooperative stopping. Partial
moments remain separate from final ad.sf products. MPI/xyz/sparse modes are
not implemented here; ordinary get_sf_mb/set_sf behavior is unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from ..core.base import xp,asnumpy,cupy_enabled
from . import structure_functions as legacy
from ..utils.batch_processing import determine_blocks_per_batch

SCHEMA='sf_pairs_v1'


def pair_ordinal(i,j,n):
    if not 0<=i<=j<n:
        raise ValueError('invalid global block pair')
    return i*n-i*(i-1)//2+j-i


def pair_work(n,batch,cursor=None):
    """Legacy intra/cross-batch order, resuming directly from a four-int cursor."""
    nb=(n+batch-1)//batch
    if cursor is None:
        cursor=(0,0,0,0)
    a0,b0,i0,j0=map(int,cursor)
    if a0==-1:
        return
    if not (0<=a0<=b0<nb and a0*batch<=i0<min(n,(a0+1)*batch)
            and max(i0,b0*batch)<=j0<min(n,(b0+1)*batch)):
        raise ValueError('invalid traversal cursor')
    for a in range(a0,nb):
        lo1=a*batch;hi1=min(n,lo1+batch)
        for b in range(b0 if a==a0 else a,nb):
            lo2=b*batch;hi2=min(n,lo2+batch)
            resumed=(a,b)==(a0,b0)
            for i in range(i0 if resumed else lo1,hi1):
                lo=max(i,lo2)
                for j in range(j0 if resumed and i==i0 else lo,hi2):
                    if j+1<hi2:
                        following=(a,b,i,j+1)
                    elif i+1<hi1:
                        following=(a,b,i+1,max(i+1,lo2))
                    elif b+1<nb:
                        following=(a,b+1,lo1,(b+1)*batch)
                    elif a+1<nb:
                        following=(a+1,a+1,(a+1)*batch,(a+1)*batch)
                    else:
                        following=(-1,-1,-1,-1)
                    yield a,b,i,j,following


CODE_HASH_KEYS=('kernel_source_sha256','engine_sha256')


def code_changed_only(saved_signature_json,signature):
    """True when a saved state differs from the current signature only in code
    hashes. The sampling tag (SF_SAMPLER_VERSION) changes whenever the
    sampling numerics change, so such a state can be continued."""
    saved=json.loads(saved_signature_json)
    current=json.loads(json.dumps(signature,sort_keys=True))
    strip=lambda sig:{k:v for k,v in sig.items() if k not in CODE_HASH_KEYS}
    return saved!=current and strip(saved)==strip(current)


def _signature(ad,source,name,variables,weights,npairs,nbins,order,log_bins,batch,minimum,metadata,product_keys,
               delta_edges=None):
    geom=np.asarray(ad.mb_geometry,dtype=np.float64)
    file_info={}
    if source is not None:
        path=Path(source).resolve();stat=path.stat()
        file_info=dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    return dict(schema=SCHEMA,source=file_info,physical_time=float(ad.time),name=name,
        variables=list(variables),weights=list(weights),npairs=float(npairs),nbins=int(nbins),
        order=int(order),log_bins=bool(log_bins),batch=int(batch),minimum=int(minimum),
        selection_metadata=metadata if metadata is not None else {},product_keys=list(product_keys),nmb=int(ad.n_mbs),
        dimensions=[int(ad.nx1),int(ad.nx2),int(ad.nx3)],sparse_data=False,
        backend='cupy' if cupy_enabled else 'numpy',precision='float64',
        geometry_sha256=hashlib.sha256(geom.tobytes()).hexdigest(),
        header_sha256=hashlib.sha256(json.dumps(ad._header,sort_keys=True,default=str).encode()).hexdigest(),
        kernel_source_sha256=hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest(),
        engine_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sampling=legacy.SF_SAMPLER_VERSION,
        delta_edges=None if delta_edges is None else np.asarray(asnumpy(delta_edges),dtype=float).tolist())


def _validate_state(state,signature,shape,total,pdf_shape=None):
    if state.get('signature_json')!=json.dumps(signature,sort_keys=True):
        raise ValueError('SF checkpoint source/settings/backend/algorithm mismatch; refusing resume')
    expected={'sf_sums':shape,'weight_sums':(shape[0],shape[-1]),
              'bin_counts':(shape[0],shape[-1]),'completed_bitmap':((total+7)//8,)}
    if pdf_shape is not None:
        expected['delta_pdf']=pdf_shape
    for name,dims in expected.items():
        value=np.asarray(state.get(name))
        if value.shape!=dims or not np.all(np.isfinite(value)):
            raise ValueError(f'invalid checkpoint array {name}')
    if state.get('status') not in ('partial','complete'):
        raise ValueError('invalid SF checkpoint status')
    if np.asarray(state['completed_bitmap']).dtype!=np.uint8 or np.asarray(state['cursor']).shape!=(4,):
        raise ValueError('invalid SF bitmap/cursor representation')
    if (np.any(state['bin_counts']<0) or np.any(state['weight_sums']<0) or np.any(state['sf_sums']<0)
            or (pdf_shape is not None and np.any(np.asarray(state['delta_pdf'])<0))):
        raise ValueError('negative checkpoint counts/weights')
    if np.any(np.asarray(state['bin_counts'])!=np.floor(state['bin_counts'])):
        raise ValueError('fractional checkpoint pair counts')
    bits=np.unpackbits(np.asarray(state['completed_bitmap'],dtype=np.uint8),bitorder='little')
    if np.any(bits[total:]) or int(bits[:total].sum())!=int(state['completed_pairs']):
        raise ValueError('SF checkpoint bitmap/count disagree')
    complete=int(state['completed_pairs'])==total
    if (state.get('status')=='complete')!=complete:
        raise ValueError('SF checkpoint status/pair coverage disagree')
    if complete and list(state['cursor'])!=[-1,-1,-1,-1]:
        raise ValueError('completed SF cursor is not terminal')


def checkpointed_sf(ad,name,variables,*,weights,npairs=1e10,nbins=100,max_order=3,
                     log_bins=True,simultaneous_blocks=None,nsamples_block_min=1000,
                     source=None,selection_metadata=None,checkpoint,checkpoint_seconds=60.,
                     should_stop=lambda:False,stop_after_pairs=None,redo=False,product_keys=None,
                     delta_edges=None):
    """Attach/save name state; return True only after all final SF keys publish.

    `checkpoint(ad, label)` must atomically persist the entire original cache
    or raise. A raised callback never advances any durable file. The caller
    must serialize production cache writers. stop_after_pairs is a validation
    interruption, not a scientific sampling limit.

    A saved state from another sampler version is discarded (its sums cannot
    be continued); one that differs only in code hashes is continued.
    delta_edges adds the per-bin increment histogram (see get_sf_mb), saved
    with the other accumulators and published as 'delta_pdf'.
    """
    if getattr(ad,'has_full_data',True) is False:
        raise ValueError('resumable SF currently requires serial/full-data ownership')
    variables=list(variables);weights=[weights] if isinstance(weights,str) else list(weights)
    if not isinstance(name,str) or not name or '/' in name or '\0' in name:
        raise ValueError('checkpoint name must be a nonempty HDF5-safe key')
    if not variables or not weights or nbins<2 or not 1<=max_order<=10 or npairs<=0:
        raise ValueError('invalid SF settings')
    if checkpoint_seconds<=0 or (stop_after_pairs is not None and stop_after_pairs<1):
        raise ValueError('invalid checkpoint interval/validation stop')
    keys=list(product_keys) if product_keys is not None else [v+f'_{name}' for v in variables]
    if len(keys)!=len(variables) or not all(isinstance(k,str) and k and '/' not in k and '\0' not in k for k in keys) or len(set(keys))!=len(keys):
        raise ValueError('one unique nonempty product key per variable required')
    ad.sf=dict(getattr(ad,'sf',{}) or {})
    if not redo and all(legacy.sf_entry_is_current(ad.sf.get(k)) for k in keys):
        return True
    states=dict(getattr(ad,'sf_checkpoints',{}) or {})
    previous=None if redo else states.get(name)
    if previous is not None and json.loads(previous['signature_json']).get('sampling')!=legacy.SF_SAMPLER_VERSION:
        # Sums from another sampler cannot be continued by this one.
        previous=None
    if delta_edges is not None:
        delta_edges=legacy._validated_delta_edges(delta_edges,len(variables))
    n=int(ad.n_mbs);total=n*(n+1)//2;points=int(ad.nx1*ad.nx2*ad.nx3)
    if n<1:
        raise ValueError('SF needs at least one meshblock')
    if previous is not None:
        frozen=json.loads(previous['signature_json'])['batch']
        if simultaneous_blocks is not None and min(int(simultaneous_blocks),n)!=frozen:
            raise ValueError('batch size differs from saved traversal')
        batch=frozen
    else:
        batch=min(int(simultaneous_blocks),n) if simultaneous_blocks is not None else min(n,
            determine_blocks_per_batch(n,len(variables)*3,len(weights),points))
    if batch<1:
        raise ValueError('invalid batch size')
    signature=_signature(ad,source,name,variables,weights,npairs,nbins,max_order,
                         log_bins,batch,nsamples_block_min,selection_metadata,keys,delta_edges)
    if previous is not None and code_changed_only(previous['signature_json'],signature):
        print(f'SF {name}: checkpoint written by an earlier revision of the same sampler; continuing',flush=True)
        previous={**previous,'signature_json':json.dumps(signature,sort_keys=True)}
    shape=(len(weights),len(variables),max_order,nbins)
    pdf_shape=None if delta_edges is None else (len(weights),len(variables),nbins,delta_edges.shape[1]+1)
    L=np.array([ad.x1max-ad.x1min,ad.x2max-ad.x2min,ad.x3max-ad.x3min],float)
    geom=np.asarray(ad.mb_geometry);first=geom[0]
    cell=(first[1::2]-first[::2])/np.array([ad.nx1,ad.nx2,ad.nx3])
    positive=cell[cell>0]
    if not positive.size:
        raise ValueError('invalid SF cell geometry')
    rmin=float(positive.min());rmax=float(np.sqrt(.25*np.sum(L**2)))
    edges,radial=legacy.radial_bin_edges(rmin,rmax,nbins,log_bins)
    periodic=[('periodic' in str(ad._header['mesh'][f'ix{i}_bc'])) for i in (1,2,3)]
    if previous is None:
        state=dict(signature_json=json.dumps(signature,sort_keys=True),status='partial',
            sf_sums=np.zeros(shape),weight_sums=np.zeros((len(weights),nbins)),
            bin_counts=np.zeros((len(weights),nbins)),completed_bitmap=np.zeros((total+7)//8,dtype=np.uint8),
            completed_pairs=0,cursor=np.array([0,0,0,0],dtype=np.int64),generation=0,
            elapsed_seconds=0.,r=asnumpy(radial),edges=asnumpy(edges))
        if pdf_shape is not None:
            state['delta_pdf']=np.zeros(pdf_shape)
    else:
        state={**previous}
        _validate_state(state,signature,shape,total,pdf_shape)
        if not np.array_equal(state['edges'],asnumpy(edges)):
            raise ValueError('SF radial edges differ from checkpoint')
    # Copies isolate active GPU/CPU accumulators from the last checkpoint.
    sums=xp.array(state['sf_sums'],copy=True);counts=xp.array(state['bin_counts'],copy=True)
    weight_sums=xp.array(state['weight_sums'],copy=True)
    delta_pdf=None if pdf_shape is None else xp.array(state['delta_pdf'],copy=True)
    bitmap=np.array(state['completed_bitmap'],dtype=np.uint8,copy=True)
    centers=.5*(geom[:,::2]+geom[:,1::2]);samples=min(int(np.sqrt(npairs)/n),points)
    cache={};resident=None;processed_this_call=0;started=last_save=time.monotonic()
    prior_elapsed=float(state['elapsed_seconds'])

    # Live progress stays in locals; `state` is only ever the last snapshot
    # handed to ad.sf_checkpoints, so an exception mid-loop leaves a
    # self-consistent state behind for an in-process retry.
    completed_pairs=int(state['completed_pairs']);cursor=np.array(state['cursor'],dtype=np.int64,copy=True)

    def save(complete=False):
        nonlocal last_save,state
        if cupy_enabled:
            xp.cuda.Stream.null.synchronize()
        if source is not None:
            # Code files may be edited while a long run is in flight; only the
            # data, settings and sampler must stay fixed.
            current=_signature(ad,source,name,variables,weights,npairs,nbins,max_order,log_bins,
                               batch,nsamples_block_min,selection_metadata,keys,delta_edges)
            if any(current[k]!=signature[k] for k in current if k not in CODE_HASH_KEYS):
                raise ValueError('source/config changed during SF; refusing checkpoint')
        snapshot=dict(state)
        snapshot.update(completed_pairs=completed_pairs,cursor=cursor.copy(),
            sf_sums=np.array(asnumpy(sums),copy=True),
            weight_sums=np.array(asnumpy(weight_sums),copy=True),bin_counts=np.array(asnumpy(counts),copy=True),
            completed_bitmap=bitmap.copy(),generation=int(state['generation'])+1,
            elapsed_seconds=prior_elapsed+time.monotonic()-started,status='complete' if complete else 'partial')
        if delta_pdf is not None:
            snapshot['delta_pdf']=np.array(asnumpy(delta_pdf),copy=True)
        _validate_state(snapshot,signature,shape,total,pdf_shape)
        state=snapshot;states[name]=state;ad.sf_checkpoints=states
        if complete:
            for i,key in enumerate(keys):
                if redo or not legacy.sf_entry_is_current(ad.sf.get(key)):
                    ad.sf[key]=dict(r=np.array(state['r'],copy=True),
                        sf=np.asarray(state['sf_sums'][:,i]).squeeze().copy(),
                        weight_sum_bin=np.asarray(state['weight_sums']).squeeze().copy(),
                        num_bin_points=np.asarray(state['bin_counts'],dtype=np.int64).squeeze().copy(),
                        sampler=legacy.SF_SAMPLER_VERSION)
                    if delta_pdf is not None:
                        ad.sf[key]['delta_pdf']=np.asarray(state['delta_pdf'][:,i]).squeeze().copy()
                        ad.sf[key]['delta_edges']=asnumpy(delta_edges[i]).copy()
        checkpoint(ad,f'SF {name} '+('complete' if complete else 'partial'))
        last_save=time.monotonic()
        print(f'SF {name}: durable pairs {state["completed_pairs"]}/{total}; '
              f'accepted={float(state["bin_counts"].sum()):.0f}; generation={state["generation"]}',flush=True)

    try:
        for a,b,i,j,following in pair_work(n,batch,cursor):
            if should_stop() or (stop_after_pairs is not None and processed_this_call>=stop_after_pairs):
                save();return False
            ordinal=pair_ordinal(i,j,n);byte,bit=divmod(ordinal,8)
            if bitmap[byte]&(1<<bit):
                raise ValueError('SF cursor revisits already committed pair; refusing double count')
            if resident!=(a,b):
                block1=list(range(a*batch,min(n,(a+1)*batch)))
                block2=list(range(b*batch,min(n,(b+1)*batch)))
                # Keep batch1 across its cross-batch phases, just as legacy
                # traversal does. Never retain all domain blocks on the GPU.
                for key in list(cache):
                    if resident is None or resident[0]!=a or key not in block1:
                        del cache[key]
                legacy._load_batch_data(ad,cache,block1,weights,variables)
                legacy._load_batch_data(ad,cache,block2,weights,variables)
                resident=(a,b)
            if i not in cache or j not in cache:
                raise ValueError('required SF block missing from cache')
            if should_stop():
                save();return False
            legacy._process_block_pair(ad,i,j,cache,centers,weights,variables,
                sums,counts,weight_sums,rmax,rmin,*L,edges,nbins,max_order,samples,points,
                nsamples_block_min,*periodic,sparse_data=False,use_mpi=False,
                delta_edges=delta_edges,delta_pdf=delta_pdf)
            bitmap[byte]|=np.uint8(1<<bit)
            completed_pairs+=1
            cursor=np.asarray(following,dtype=np.int64);processed_this_call+=1
            if completed_pairs<total and time.monotonic()-last_save>=checkpoint_seconds:
                save()
        if completed_pairs!=total:
            raise ValueError('SF traversal ended without complete pair coverage')
        save(complete=True);return True
    finally:
        cache.clear()
