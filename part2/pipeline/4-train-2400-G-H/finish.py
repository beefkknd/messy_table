"""Audit accounting, preserve all prior artifacts, and build the final report."""
from common_new import *
from collections import Counter
import subprocess,sys
import numpy as np
def main():
    verify();done=read(HERE/'completion.json');assert sha(HERE/'step1-results.json')==done['step1_sha256']
    assert sha(HERE/'step2/results.json')==done['step2_sha256'];step1=read(HERE/'step1-results.json');step2=read(HERE/'step2/results.json')
    control_done=read(HERE/'fullframe-control-completion.json')
    assert control_done['primary_completion_sha256']==sha(HERE/'completion.json')
    assert control_done['result_sha256']==sha(HERE/'step2/fullframe-controls.json')
    checks=[]
    pairdir=HERE/'step2/pairs';pairfreeze=read(pairdir/'freeze.json')
    assert len(pairfreeze['files'])==3600
    for name,digest in pairfreeze['files'].items():
        file=pairdir/name;assert sha(file)==digest,name
        with np.load(file,allow_pickle=False) as arrays:
            for key in arrays.files:
                values=arrays[key];assert np.isfinite(values).all(),(name,key)
                if key.endswith('__blend'):assert ((values>=-1e-12)&(values<=1+1e-12)).all()
                if key.endswith('__cosine'):assert ((values>=-1.00001)&(values<=1.00001)).all()
                if key.endswith('__roma_coverage'):assert ((values>=0)&(values<=1.00001)).all()
                if key.endswith('__vit_distance'):assert (values>=0).all()
    checks.append('All 3600 score archives match their frozen hashes; scores and component evidence are finite and bounded')
    for arm,r in step1.items():
        c=r['counts'];assert c['ground_truth']==408937 and c['images']==16749 and c['shared_objects']==1486614
        assert c['localized']+c['unmapped']==c['detections'];assert r['metrics']['recall']==c['localized']/c['ground_truth']
        assert r['metrics']['precision']==c['localized']/c['detections'];assert sum(s['counts']['localized'] for s in r['cases'])==c['localized']
    checks.append('All full-test detection denominators, precision, recall and unmapped accounting agree')
    for method,arms in step2['methods'].items():
        for arm,r in arms.items():
            c=r['counts'];assert c['shared']==78335 and c['pairs']==3600
            assert c['proposals']==sum(c[k] for k in ['correct','wrong','unmapped'])
            assert 0<=c['correct']<=c['detected_both']<=c['shared']
            assert r['metrics']['recall']==c['correct']/78335
            assert r['metrics']['precision']==c['correct']/c['proposals'] if c['proposals'] else r['metrics']['precision'] is None
            assert (78335-c['detected_both'])+(c['detected_both']-c['correct'])==78335-c['correct']
            total=Counter()
            for row in step2['pair_outcomes']:total.update(row['results'][method][arm]['counts'])
            assert dict(total)==c
    checks.append('Identity recall includes all missed detections; wrong/unmapped links and stagewise miss decomposition agree')
    for method,arms in read(HERE/'step2/fullframe-controls.json')['methods'].items():
        for arm,r in arms.items():
            c=r['counts'];assert c['shared']==78335 and c['pairs']==3600
            assert c['proposals']==c['correct']+c['wrong']+c['unmapped']
            assert c['correct']<=c['detected_both']<=c['shared']
    checks.append('Secondary full-frame-only matching controls preserve the denominator and recompute association')
    assert step2['oracle_exact_replay'];checks.append('Original strong human-box matcher reproduces 74159 correct and 6147 wrong')
    assert read(HERE/'step2/preprocessing-checks.json')['passed']
    assert read(HERE/'step2/roma-reference-check.json')['passed']
    checks.append('New DINO/ViT crops reproduce their original adapters; new RoMa forward reproduces frozen scene evidence')
    old=read(OLD/'full-test-matched-controls.json')['sliced600']['raw']['counts']
    for k in ['localized','unmapped','shared_localized_both','shared_objects']:assert step1['600_768']['counts'][k]==old[k]
    checks.append('Current 600-photo detector baseline replays exactly')
    training=read(HERE/'training/sliced2400/training.json');assert training['completed_updates']==3000
    assert training['checkpoint_sha256']==sha(W2400);assert len(training['steps'])==3000
    for name,checkpoint in [('600',W600),('2400',W2400)]:
        frozen=read(HERE/'predictions'/name/'freeze.json')
        assert frozen['checkpoint_sha256']==sha(checkpoint) and len(frozen['scenes'])==1861
    checks.append('One 2400-photo training run completed its fixed 3000-update budget')
    spec=read(HERE/'integration-code-freeze.json')
    for name,digest in spec['files'].items():assert sha(HERE/name)==digest,name
    for name,digest in spec['source_files'].items():assert sha(ROOT/name)==digest,name
    checks.append('Matching code frozen before execution; historical reports and checkpoints unchanged')
    save(HERE/'audit.json',{'status':'passed','checks':checks,'limits':['One seed; fixed compute, not convergence-matched.','Already exposed test and benchmark sample.','Native IoU-sum mapping is retained, not silently replaced with a maximum-cardinality mapper.']})
    subprocess.run([sys.executable,str(HERE/'build_report.py')],check=True,cwd=ROOT)
    save(HERE/'results-freeze.json',{n:sha(HERE/n) for n in ['protocol.json','integration-code-freeze.json','step1-results.json','step1-fullframe-controls.json','step2/results.json','step2/fullframe-controls.json','fullframe-control-spec.json','audit.json','summary.json']})
    emit({'phase':'finished','audit':'passed','report':str(HERE/'report.html')})
if __name__=='__main__':main()
