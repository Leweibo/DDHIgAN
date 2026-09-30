"""Independent arithmetic validation of exported fixed-rank0 aggregate CSVs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    root = args.root
    complete = json.loads((root/'COMPLETE.json').read_text())
    assert complete['status'] == 'COMPLETE_VALIDATED'
    assert complete['replicates'] == 500 and complete['new_fits'] == 2500
    assert complete['fixed_ranking_weight'] == 0 and not complete['model_selection_repeated']
    cells = pd.read_csv(root/'evaluation_cells.csv')
    summary = pd.read_csv(root/'summary.csv',dtype={'query':str})
    metrics = summary.metric.unique().tolist()
    assert len(metrics) == 8 and len(cells) == 501*5*6*3
    keys = ['replicate','fold','query','role']
    assert not cells.duplicated(keys).any()
    expected = pd.MultiIndex.from_product([range(-1,500),range(5),range(6),
                 ['heldout','development_apparent','development_original']],names=keys)
    assert set(map(tuple,cells[keys].values)) == set(expected)
    assert np.isfinite(cells[metrics].values).all()
    maxima = []
    def equal(a,b):
        maxima.append(float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
    for query in ['0','1','2','3','4','5','mean']:
        subset = cells if query == 'mean' else cells[cells['query'] == int(query)]
        means = subset.groupby(['replicate','role'])[metrics].mean()
        held = means.xs('heldout',level='role').loc[range(500)]
        app = means.xs('development_apparent',level='role')
        orig = means.xs('development_original',level='role')
        diff = app.loc[range(500)]-orig.loc[range(500)]
        equal(app.loc[-1],orig.loc[-1])
        s = summary[summary['query']==query].set_index('metric').loc[metrics]
        equal(s.original_heldout,means.loc[(-1,'heldout')])
        equal(s.refit_bootstrap_mean,held.mean())
        equal(s.refit_bootstrap_sd,held.std(ddof=1))
        equal(s.percentile_low,held.quantile(.025))
        equal(s.percentile_high,held.quantile(.975))
        equal(s.original_development_apparent,app.loc[-1])
        equal(s.signed_optimism,diff.mean())
        equal(s.optimism_corrected_development,app.loc[-1]-diff.mean())
        equal(s.optimism_mcse,diff.std(ddof=1)/np.sqrt(500))
    forbidden = ('patient_id','name','身份证','姓名')
    assert not any(v in str(c).lower() for c in cells.columns for v in forbidden)
    print(json.dumps(dict(status='VALID',replicates=500,new_fits=2500,
          independently_rebuilt_summaries=len(summary),maximum_absolute_error=max(maxima),
          fixed_ranking_weight=0,optimism_not_subtracted_from_heldout=True),indent=2))


if __name__ == '__main__':
    main()
