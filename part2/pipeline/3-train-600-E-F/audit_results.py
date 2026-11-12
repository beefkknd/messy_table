"""Independent arithmetic/provenance checks; cannot select or change a policy."""
from common_round import *
from collections import Counter
import numpy as np

def check_result(result,expected):
    c=result['counts'];m=result['metrics']
    assert c['ground_truth']==expected['ground_truth']
    assert c['images']==expected['images'] and c['shared_objects']==expected['shared_objects']
    assert c['localized']+c['unmapped']==c['detections']
    assert c['shared_localized_both']<=c['shared_objects']
    # The assignment is recomputed at each IoU cutoff; credited identity sets
    # need not be nested under the historical maximum-IoU-sum policy.
    assert 0<=c['localized_at_iou75']<=min(c['ground_truth'],c['detections'])
    assert m['recall']==c['localized']/c['ground_truth']
    assert m['precision']==c['localized']/c['detections']
    assert m['both_view_coverage']==c['shared_localized_both']/c['shared_objects']
    total=Counter()
    for setup,rows in result['setups'].items():total.update(rows)
    assert dict(total)==c
    cases=Counter()
    for row in result['cases']:cases.update(row['counts'])
    assert dict(cases)==c
    for size in ('lt32','32_to_lt96','ge96'):assert c.get('tp_'+size,0)<=c.get('gt_'+size,0)
    assert sum(c['gt_'+s] for s in ('lt32','32_to_lt96','ge96'))==c['ground_truth']

def main():
    p=verify_protocol();checks=[]
    dev=read(HERE/'development-comparison.json');full=read(HERE/'full-test-comparison.json')
    for name,result in dev.items():check_result(result,{'ground_truth':4735,'images':200,'shared_objects':2173})
    checks.append('All development counts, setup/case sums, ratio denominators and short-side strata agree')
    for name,result in full.items():check_result(result,{'ground_truth':408937,'images':16749,'shared_objects':1486614})
    checks.append('All full-test counts, setup/case sums, ratio denominators and short-side strata agree')
    config=full['reference']['config']
    if config['family']=='G' and config.get('full_threshold',.25)==.25:
        old=read(ROOT/'evaluation/messytable-detection-full-test/results.json')['arms']['fine_tuned_tiles_G']['counts']
        for key in ('localized','ground_truth','detections','unmapped','shared_objects','shared_localized_both'):
            assert full['reference']['counts'][key]==old[key],key
        checks.append('Historical gated full-test reference replays all primary counts exactly')
    stage=read(HERE/'stage1-selection.json')
    assert stage['frozen_before_development_unix']<read(HERE/'final-test-freeze.json')['created_unix']
    for arm in ('frames600','sliced600'):
        selection=read(HERE/f'{arm}-selection.json')
        feasible=[r for r in selection['entries'] if r['result']['counts']['unmapped']<=stage['calibration_unmapped_cap']]
        selected=selection['selected']
        assert selected==max(feasible,key=lambda x:(x['result']['metrics']['both_view_coverage'],x['result']['metrics']['precision'] or 0,-x['result']['counts']['detections']))
        assert dev[arm]['config']==selected['config']==full[arm]['config']
        log=read(HERE/f'training/{arm}/training.json');assert log['completed_updates']==3000
        assert len(log['steps'])==3000 and all(np.isfinite(r['loss']) for r in log['steps'])
        assert sha(HERE/f'training/{arm}/checkpoint.pt')==log['checkpoint_sha256']
    checks.append('Training budgets matched; all losses finite; thresholds selected on calibration and reused unchanged')
    plans=[read(HERE/f'training/{arm}/run-plan.json') for arm in ('frames600','sliced600')]
    assert plans[0]['images']==plans[1]['images'];assert plans[0]['input_checkpoint_sha256']==plans[1]['input_checkpoint_sha256']
    checks.append('Same starting checkpoint and exact ordered photo exposures in both training arms')
    for population,expected in [('development',{'ground_truth':4735,'images':200,'shared_objects':2173}),
                                ('full-test',{'ground_truth':408937,'images':16749,'shared_objects':1486614})]:
        controls=read(HERE/f'{population}-matched-controls.json')
        for family in ('base','raw','G'):
            assert controls['frames600'][family]['config']==controls['sliced600'][family]['config']
            for arm in controls:check_result(controls[arm][family],expected)
    checks.append('Fixed-inference sliced/full-frame controls use identical policies and confidence thresholds')
    if (HERE/'control-followup-freeze.json').exists():
        followup=read(HERE/'control-followup-freeze.json');done=read(HERE/'control-followup-completion.json')
        assert followup['primary_completion_sha256']==sha(HERE/'completion.json')
        assert followup['controls_sha256']==sha(HERE/'development-matched-controls.json')
        assert done['full_test_controls_sha256']==sha(HERE/'full-test-matched-controls.json')
        assert followup['predeclared_controls_amendment']['created_unix']<read(HERE/'frames600-selection.json')['frozen_before_development_unix']
        assert read(HERE/'completion.json')['status']=='stopped_by_development_gate'
        selected=followup['selected'];assert selected['arm']=='sliced600' and selected['full_threshold']==.25 and selected['family']=='raw'
        checks.append('Primary failure preserved; qualifying secondary control was prespecified before validation and frozen before full-test inference')
    assert sha(ROOT/'evaluation/messytable-detection-full-test/results.json')==p['historical_full_test_results_sha256']
    for name,digest in p['part_one_sha256'].items():assert sha(ROOT/name)==digest
    checks.append('Historical full-test results and frozen Part One files are unchanged')
    # IoU-sum does not universally maximize match cardinality: disclose, rather than changing the primary metric.
    assert 'IoU-sum' in p['scoring']
    save(HERE/'final-audit.json',{'status':'passed','checks':checks,'protocol_sha256':sha(HERE/'protocol.json'),
        'full_test_sha256':sha(HERE/'full-test-comparison.json'),'created_unix':time.time(),
        'limitations':['Independent arithmetic/provenance audit, not an independent reimplementation of all detector inference.',
                      'Historical maximum-IoU-sum metric remains primary; no guarantee it maximizes cardinality in every frame.',
                      'One training seed; full test previously exposed; no identity matching evaluated.']})
    emit({'phase':'audit','status':'passed','checks':len(checks)})
if __name__=='__main__':main()
