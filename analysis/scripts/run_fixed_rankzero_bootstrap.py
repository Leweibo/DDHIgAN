"""Fixed-rank0 bootstrap of five-fold performance and development optimism.

Run on a frozen RIS release. All patient-level artifacts stay on RIS.
Existing scientific trainer is reused through process-local multiplicity hooks.
"""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from python.deephit.fixed_bootstrap import draw_counts, audit, signed_optimism
from scripts import run_ddhigan_refit_pilot as pilot

NAME = 'physv15_fixed_rankzero_bootstrap_20260915'
ROOT = Path('results')/NAME
SPEC = Path('config')/NAME/'run_spec.json'
BASE = Path('config/physv15_rankzero_standard_20260911/core.yaml')
NREP = 500
sha, write = pilot.sha, pilot.write


def task(rep, fold):
    return ROOT/'tasks'/f'rep{rep:04d}_fold{fold}'


def check():
    manifest = json.loads(Path('FIXED_BOOTSTRAP_CODE.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(name) == digest, name
    spec = json.loads(SPEC.read_text())
    assert spec['replicates'] == NREP and spec['new_fits'] == NREP*5
    cfg = yaml.safe_load(BASE.read_text())
    assert cfg['model']['loss']['ranking'] == spec['ranking_weight'] == 0
    assert cfg['model']['max_residual_time'] == 11
    assert cfg['training']['seed'] == 316 and cfg['training']['patience'] == 20
    assert sha(cfg['data']['release_manifest_path']) == spec['release_manifest_sha256']
    return cfg


def outcomes(cfg):
    frame = pd.read_csv(Path(cfg['data']['processed_dir'])/'igan_outcomes.csv', dtype={'patient_id': str})
    frame = frame.set_index('patient_id').sort_index()
    assert len(frame) == 9947 and frame.index.is_unique
    return frame


def preflight():
    from python.utils.data_utils import get_fold_split
    from python.utils.survival_data import split_patient_ids
    from python.deephit.train import enforce_external_confirmations
    cfg = check()
    enforce_external_confirmations(cfg)
    live = pilot.resources()
    out = outcomes(cfg)
    folds = [list(map(str, get_fold_split(cfg['data']['splits_dir'], f, 'test'))) for f in range(5)]
    assert set(sum(folds, [])) == set(out.index)
    partitions = []
    originals = []
    for f in range(5):
        development = list(map(str, get_fold_split(cfg['data']['splits_dir'], f, 'train')))
        assert set(development) == set(out.index)-set(folds[f])
        tr, va = split_patient_ids(development, .2, 316+f, out.loc[development, 'eskd_status'].astype(int).tolist())
        partitions.append(dict(train=tr, validation=va, test=folds[f], development=development))
        prefix = cfg['model']['name']
        ck = Path(cfg['output']['checkpoint_dir'])/f'{prefix}_ESKD_fold{f}.pt'
        norm = Path(cfg['output']['norm_stats_dir'])/f'fold_{f}_{prefix}_ESKD_norm_stats.json'
        originals.append(dict(fold=f, checkpoint=str(ck), checkpoint_sha256=sha(ck),
                              norm=str(norm), norm_sha256=sha(norm)))
    root = ROOT/'run'
    root.mkdir(parents=True, exist_ok=False)
    write(ROOT/'private_partitions.json', partitions)
    records = []
    for rep in range(NREP):
        counts = draw_counts(folds, rep)
        write(ROOT/'draws'/f'rep{rep:04d}.json', counts)
        for f, split in enumerate(partitions):
            tc = {p: counts[p] for p in sorted(split['train']) if p in counts}
            vc = {p: counts[p] for p in sorted(split['validation']) if p in counts}
            assert out.loc[list(tc), 'eskd_status'].nunique() == 2
            assert out.loc[list(vc), 'eskd_status'].nunique() == 2
            roster = dict(train_counts=tc, validation_counts=vc,
                          audit=audit(split['train'], split['validation'], split['test'], tc, vc))
            dest = task(rep, f)
            write(dest/'private_roster.json', roster)
            c = json.loads(json.dumps(cfg))
            c['model']['name'] = f'ddhigan_fixedboot_rep{rep:04d}'
            c['data'].pop('query_roster_preflight_path', None)
            for key, sub in [('norm_stats_dir','norm_stats'), ('checkpoint_dir','checkpoint'),
                             ('prediction_dir','prediction'), ('training_dir','training'), ('tensorboard_dir','tensorboard')]:
                c['output'][key] = str(dest/sub)
            with (dest/'config.yaml').open('x') as handle:
                yaml.safe_dump(c, handle, sort_keys=False)
            records.append(dict(rep=rep, fold=f, config_sha256=sha(dest/'config.yaml'),
                                roster_sha256=sha(dest/'private_roster.json')))
    write(root/'preflight.json', dict(status='VALID', resources=live, tasks=records,
          originals=originals, partitions_sha256=sha(ROOT/'private_partitions.json'),
          draw_hashes={str(r): sha(ROOT/'draws'/f'rep{r:04d}.json') for r in range(NREP)}))
    print('PREFLIGHT_VALID 500 shared draws; 2500 fixed-rank0 fits', flush=True)


def loadtask(rep, fold):
    check()
    pre = json.loads((ROOT/'run/preflight.json').read_text())
    assert sha(ROOT/'private_partitions.json') == pre['partitions_sha256']
    dp = ROOT/'draws'/f'rep{rep:04d}.json'
    assert sha(dp) == pre['draw_hashes'][str(rep)]
    record = pre['tasks'][rep*5+fold]
    assert (record['rep'], record['fold']) == (rep, fold)
    dest = task(rep, fold)
    assert sha(dest/'private_roster.json') == record['roster_sha256']
    assert sha(dest/'config.yaml') == record['config_sha256']
    roster = json.loads((dest/'private_roster.json').read_text())
    split = json.loads((ROOT/'private_partitions.json').read_text())[fold]
    counts = json.loads(dp.read_text())
    for key, label in [('train_counts','train'), ('validation_counts','validation')]:
        assert roster[key] == {p: counts[p] for p in split[label] if p in counts}
    return dest, yaml.safe_load((dest/'config.yaml').read_text()), roster


def hooks():
    pilot.ROOT = ROOT
    pilot.loadtask = loadtask
    pilot.audit_partitions = audit
    pilot.codecheck = check
    pilot.taskdir = task


def fit(rep, fold):
    hooks()
    pilot.fit(rep, fold)


def predict(rep, fold):
    import torch
    from torch.utils.data import DataLoader, Subset
    from python.deephit.data_loader import DynamicDeepHitDataset, collate_fn
    from python.deephit.predict import predict_landmark
    from python.deephit.train import build_model
    from python.utils.time_grid import HalfYearTimeGrid
    from python.utils.data_utils import load_norm_stats
    from scripts import evaluate_query12_recency as E

    base = check()
    pre = json.loads((ROOT/'run/preflight.json').read_text())
    split = json.loads((ROOT/'private_partitions.json').read_text())[fold]
    out = outcomes(base)
    reference = pilot.cache() if rep < 0 else None
    if rep < 0:
        dest = ROOT/'original'/f'fold{fold}'
        original = pre['originals'][fold]
        ck, norm = Path(original['checkpoint']), Path(original['norm'])
        assert sha(ck) == original['checkpoint_sha256'] and sha(norm) == original['norm_sha256']
        cfg = base
        counts = {p: 1 for p in out.index}
    else:
        dest, cfg, roster = loadtask(rep, fold)
        assert json.loads((dest/'FIT_VALIDATED.json').read_text())['status'] == 'VALID'
        prefix = cfg['model']['name']
        ck = dest/'checkpoint'/f'{prefix}_ESKD_fold{fold}.pt'
        norm = dest/'norm_stats'/f'fold_{fold}_{prefix}_ESKD_norm_stats.json'
        counts = json.loads((ROOT/'draws'/f'rep{rep:04d}.json').read_text())
    target = dest/'evaluation'
    target.mkdir(parents=True, exist_ok=False)
    state = torch.load(ck, map_location='cuda')
    # Existing outputs may have different output directory metadata; compare scientific settings.
    for field in ('architecture','gru_hidden','rnn_layers','attention_hidden','event_hidden','dropout','max_residual_time','interval_width','num_event_bins','loss'):
        assert state['config']['model'][field] == cfg['model'][field], field
    assert state['config']['data']['static_cols'] == cfg['data']['static_cols']
    assert state['config']['data']['long_cols'] == cfg['data']['long_cols']
    model = build_model(cfg).cuda()
    model.load_state_dict(state['model_state_dict'])
    grid = HalfYearTimeGrid(max_time=11., interval_width=.5)
    counts_vector = np.array([counts.get(p, 0) for p in out.index], dtype=int)
    dev_mask = out.index.isin(split['development']).astype(int)
    test_mask = out.index.isin(split['test']).astype(int)
    rows, inventory = [], []
    for q in range(6):
        ds = DynamicDeepHitDataset(processed_dir=cfg['data']['processed_dir'], patient_ids=out.index.tolist(),
             static_cols=cfg['data']['static_cols'], long_cols=cfg['data']['long_cols'], outcome='ESKD',
             max_visits=30, time_grid=grid, norm_stats=load_norm_stats(norm), evaluation_landmarks=[q],
             include_actual_queries=False, min_history_time=0., anchor_nearest_t0=True,
             append_missing_query_row=False, input_time_scale=10., preserve_source_time=True,
             actual_query_end_tolerance=1e-10)
        ix = np.flatnonzero(E.eligible_queries(ds.query_time.numpy(), ds.last_observation_time.numpy(), 1))
        pred = predict_landmark(model, DataLoader(Subset(ds, ix.tolist()), batch_size=32, shuffle=False, collate_fn=collate_fn),
                   q, [3,5,10], grid, torch.device('cuda'), residual_grid=list(range(1,11)), conditioning='query_survival').sort_values('patient_id')
        assert pred.patient_id.is_unique
        assert E.eligible_queries(pred.query_time, pred.last_observation_time, 1).all()
        for h in (3,5,10):
            np.testing.assert_allclose(pred[f'cond_surv_{h}y'], pred[f'resid_surv_{h:.1f}y'], rtol=0, atol=1e-12)
            pred[f'cond_surv_{h}y'] = pred[f'resid_surv_{h:.1f}y']
        dev = out.loc[split['development']]
        dev = dev[dev.eskd_time > q]
        repeat = np.array([counts.get(p,0) for p in dev.index], dtype=int)
        g = E._censoring_step(np.repeat(dev.eskd_time.to_numpy(float)-q, repeat), np.repeat(dev.eskd_status.to_numpy(int), repeat))
        cell = E.build_cell(pred, g, out.index, out, q)
        for role, weights in [('heldout', counts_vector*test_mask),
                              ('development_apparent', counts_vector*dev_mask),
                              ('development_original', dev_mask)]:
            values, _ = E.metric(cell, weights)
            assert np.isfinite(values).all(), (rep,fold,q,role)
            if rep < 0 and role == 'heldout':
                prior, _ = E.metric(reference['cells'][0,fold,q], np.ones(len(reference['ids']), dtype=int))
                np.testing.assert_allclose(values, prior, rtol=0, atol=1e-9)
            rows.append(dict(replicate=rep,fold=fold,query=q,role=role,
                        n=int(weights[cell['indices']].sum()), **dict(zip(E.METRICS, values.tolist()))))
        path = target/f'query{q}.csv'
        pred.to_csv(path, index=False)
        inventory.append(dict(query=q,n=len(pred),sha256=sha(path)))
    write(target/'VALID.json',dict(status='VALID',replicate=rep,fold=fold,rows=rows,inventory=inventory,
                                   checkpoint_sha256=sha(ck),norm_sha256=sha(norm)))
    print(f'EVALUATED rep={rep} fold={fold}', flush=True)


def summarize():
    from scripts import evaluate_query12_recency as E
    from scipy.stats import binom
    rows = []
    for rep in range(-1,NREP):
        for fold in range(5):
            dest = ROOT/'original'/f'fold{fold}' if rep < 0 else task(rep,fold)
            result = json.loads((dest/'evaluation/VALID.json').read_text())
            assert result['status'] == 'VALID' and len(result['rows']) == 18
            for item in result['inventory']:
                assert sha(dest/'evaluation'/f'query{item["query"]}.csv') == item['sha256']
            rows.extend(result['rows'])
    frame = pd.DataFrame(rows)
    metrics = list(E.METRICS)
    assert not frame.duplicated(['replicate','fold','query','role']).any()
    assert len(frame) == (NREP+1)*5*6*3
    per_query = frame.groupby(['replicate','role','query'])[metrics].mean()
    per_rep = per_query.groupby(['replicate','role'])[metrics].mean()
    summaries = []
    precision = []
    for q in [*range(6), 'mean']:
        table = per_rep if q == 'mean' else per_query.xs(q, level='query')
        get = lambda rep, role: table.loc[(rep,role)].to_numpy(float)
        held = np.stack([get(r,'heldout') for r in range(NREP)])
        apparent = np.stack([get(r,'development_apparent') for r in range(NREP)])
        original = np.stack([get(r,'development_original') for r in range(NREP)])
        correction = signed_optimism(apparent, original, get(-1,'development_apparent'))
        for j, metric in enumerate(metrics):
            summaries.append(dict(query=q,metric=metric,original_heldout=float(get(-1,'heldout')[j]),
                  refit_bootstrap_mean=float(held[:,j].mean()),refit_bootstrap_sd=float(held[:,j].std(ddof=1)),
                  percentile_low=float(np.quantile(held[:,j],.025)),percentile_high=float(np.quantile(held[:,j],.975)),
                  original_development_apparent=float(get(-1,'development_apparent')[j]),
                  signed_optimism=float(correction['signed_optimism'][j]),
                  optimism_corrected_development=float(correction['corrected'][j]),
                  optimism_mcse=float(correction['monte_carlo_se'][j])))
            sorted_values = np.sort(held[:,j])
            for p in (.025,.975):
                low, high = binom.ppf([.025,.975],NREP,p).astype(int)
                precision.append(dict(query=q,metric=metric,quantile=p,
                    mc_rank_low=int(low),mc_rank_high=int(high),
                    mc_endpoint_low=float(sorted_values[max(0,low-1)]),
                    mc_endpoint_high=float(sorted_values[min(NREP-1,high)]),
                    first_half=float(np.quantile(held[:NREP//2,j],p)),
                    second_half=float(np.quantile(held[NREP//2:,j],p))))
    target = ROOT/'aggregate_review'
    target.mkdir(exist_ok=False)
    frame.to_csv(target/'evaluation_cells.csv',index=False)
    per_query.to_csv(target/'per_query.csv')
    per_rep.to_csv(target/'per_replicate.csv')
    pd.DataFrame(summaries).to_csv(target/'summary.csv',index=False)
    pd.DataFrame(precision).to_csv(target/'monte_carlo_precision.csv',index=False)
    write(target/'COMPLETE.json',dict(status='COMPLETE_VALIDATED',replicates=NREP,new_fits=NREP*5,
          fixed_ranking_weight=0,model_selection_repeated=False,heldout_cells=NREP*30,
          uncertainty='Conditional on original folds, original inner partitions, fixed specification and algorithm seeds',
          optimism_target='Development apparent metric, not cross-validation heldout metric',
          corrected_estimator_ci=False))
    print('FIXED_BOOTSTRAP_COMPLETE',flush=True)


def controller():
    check()
    live = pilot.resources()
    assert (ROOT/'run/DEEP_PREFLIGHT.json').exists()
    (ROOT/'run/started').mkdir()
    write(ROOT/'run/runner.json',dict(pid=os.getpid(),host=platform.node(),started=time.time(),resources=live,new_fits=NREP*5))
    stop = threading.Event()
    def lane(gpu, items):
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        for rep, fold in items:
            if stop.is_set():
                return
            path = ROOT/'run'/f'rep{rep:04d}_fold{fold}.log'
            with path.open('x') as log:
                for stage in (['predict'] if rep < 0 else ['fit','predict']):
                    try:
                        subprocess.run([sys.executable,__file__,stage,'--replicate',str(rep),'--fold',str(fold)],
                                env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3*3600)
                    except Exception:
                        stop.set()
                        raise
    try:
        for items in ([(-1,f) for f in range(5)], [(r,f) for r in range(NREP) for f in range(5)]):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(lane,g,items[g::4]) for g in range(4)]
                for future in futures:
                    future.result()
            if items[0][0] == -1:
                write(ROOT/'run/REFERENCE_VALID.json',dict(status='VALID',original_models=5,heldout_metrics_match=True))
        summarize()
    except Exception as exc:
        write(ROOT/'run/FAILED.json',dict(error=repr(exc),time=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['preflight','deep_preflight','fit','predict','summarize','run'])
    parser.add_argument('--replicate',type=int,default=0)
    parser.add_argument('--fold',type=int,default=0)
    args = parser.parse_args()
    if args.stage == 'preflight': preflight()
    elif args.stage == 'deep_preflight':
        hooks()
        pilot.deep_preflight()
    elif args.stage == 'fit': fit(args.replicate,args.fold)
    elif args.stage == 'predict': predict(args.replicate,args.fold)
    elif args.stage == 'summarize': summarize()
    else: controller()
