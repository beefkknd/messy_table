"""Freeze scope, source boundaries, 600 complete training frames, and recipe."""
from common_round import *
from collections import defaultdict, Counter

def main():
    assert not (HERE/'protocol.json').exists()
    train=read(DATA/'labels/train.json')['scenes']
    trusted=read(PROGRAM/'data/cohorts/trusted150.json')['scenes']
    chosen={v['path'] for s in trusted for v in s['views']}
    pools=defaultdict(list)
    for sid,s in train.items():
        for cam,v in s['cameras'].items():
            path='data/messytable-full/MessyTable/images/'+v['pathname']
            if path not in chosen: pools[sid.rsplit('-',1)[0]].append(path)
    key=lambda s: hashlib.sha256(('small-object-round-v1:'+s).encode()).hexdigest()
    for rows in pools.values(): rows.sort(key=key)
    setup_order=sorted(pools,key=key)
    while len(chosen)<600:
        for setup in setup_order:
            if pools[setup] and len(chosen)<600: chosen.add(pools[setup].pop())
    by_path={v['pathname']:(sid,cam,v) for sid,s in train.items() for cam,v in s['cameras'].items()}
    frames=[]; clipping=0
    for path in sorted(chosen):
        sid,cam,v=by_path[Path(path).name]; boxes=[]
        for obj in v['instances'].values():
            b=obj['pos']; clipped=[max(0,min(1920,b[0])),max(0,min(1080,b[1])),max(0,min(1920,b[2])),max(0,min(1080,b[3]))]
            clipping+=b!=clipped
            assert clipped[2]>clipped[0] and clipped[3]>clipped[1]
            boxes.append(clipped)
        frames.append({'image':path,'image_sha256':sha(ROOT/path),'scene':sid,'setup':sid.rsplit('-',1)[0],
                       'camera':cam,'scope':'complete_frame','partition':'train','boxes_xyxy':boxes})
    train_setups={r['setup'] for r in frames}; train_hashes={r['image_sha256'] for r in frames}
    boundary={}
    for name in ('calibration30','development100'):
        scenes=read(PROGRAM/f'data/cohorts/{name}.json')['scenes']
        assert not train_setups & {s['setup'] for s in scenes}
        assert not train_hashes & {v['sha256'] for s in scenes for v in s['views']}
        boundary[name]={'cohort_sha256':sha(PROGRAM/f'data/cohorts/{name}.json'),
                        'truth_sha256':sha(PROGRAM/f'data/scoring/{name}.json'),
                        'pairs':len(scenes),'setup_overlap':0,'image_overlap':0}
    scenes=read(ROOT/'evaluation/messytable-detection-full-test/images.json')['scenes']
    assert not train_setups & {s['setup'] for s in scenes}
    assert not train_hashes & {v['sha256'] for s in scenes for v in s['views']}
    save(HERE/'train600.json',{'frames':frames,'summary':{'photos':600,'boxes':sum(len(r['boxes_xyxy']) for r in frames),
                'setups':len(train_setups),'includes_original150':True,'native_boxes_clipped':clipping}})
    save(HERE/'boundary-audit.json',{'train':read(HERE/'train600.json')['summary'],'evaluation':boundary,
                                   'full_test_setup_overlap':0,'full_test_image_overlap':0})
    paths=['common_round.py','prepare.py','tile_policy.py','train.py','evaluate.py','run_round.py']
    protocol={'created_unix':time.time(),'scope':'YOLO Step 0/1 only; no teacher/student or identity matching',
      'hard_cap_seconds':21600,'selection':'Calibration30 only tunes rules and thresholds; Development100 chooses among frozen arms once; full official test evaluated once after final freeze',
      'exposure_warning':'Calibration and development are historical exposed validation cohorts; full official test is also previously exposed, not a fresh holdout.',
      'success_rule':{'both_view_gain_pp':3,'unmapped_cap':'Historical G on the same cohort','reference':'Best feasible preexisting base/raw/G arm under G unmapped cap'},
      'training':{'arms':['frames600','sliced600'],'photos':600,'additional_updates':3000,'batch':4,'imgsz':1024,
       'start':'Existing A300; fresh optimizer identically in both arms','seed':20261111,'backbone':'unfrozen except fixed DFL projection',
       'optimizer':'AdamW, weight decay .0005, gradient clip 10','peak_lr':.0001,'warmup_updates':100,'final_lr':.000001,
       'schedule':'linear warmup from .1*peak, then cosine decay','loss':'stock end-to-end mixture .8 one2many + .2 one2one',
       'sliced600':'same image schedule; two full frames and two fixed 768px crops per batch, alternating slots; all intersecting GT clipped and retained; no pseudo labels',
       'checkpoint_selection':'final update only; no early selection on development'},
      'inference':{'resolution':768,'head':'one2many','nms_iou':.7,'max_det':300,'full_floor':.001,'tile_floor':.25,'dtype':'float32','rect':True,'tile_size':768,'tile_stride':576,'merge_iou':.55,
          'note':'All old/new arms use the same 768 inference resolution; 1024 is training only; native tile pixels unchanged.'},
      'threshold_grid':[.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80],
      'new_tile_grid':{'small_cutoff':[.4,.55,.7],'large_cutoff':[.65,.8],'support':['none','low_confidence'],'clearance':[4,12],'fragment_ios_max':[.85,1.01]},
      'tile_policy':'Preserve full-frame prefix; filter before merging. Reject artificial-edge cuts. Small/large means predicted native short side <96/≥96. Nonedge other-tile support IoU≥.55 and confidence≥.4 when required below .8; correlated evidence only. Seam containment rejects IoS≥limit before normal IoU .55 dedup.',
      'scoring':'Frozen class-agnostic maximum eligible IoU-sum assignment at IoU≥.50; micro recall/precision; U=predictions−localized; both-view endpoint coverage, not identity accuracy. Native short-side size strata. Setup bootstrap 20,000 draws; one seed exploratory.',
      'starting_checkpoint_sha256':sha(A),'train_manifest_sha256':sha(HERE/'train600.json'),
      'native_train_sha256':sha(DATA/'labels/train.json'),
      'historical_full_test_results_sha256':sha(ROOT/'evaluation/messytable-detection-full-test/results.json'),
      'part_one_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'evaluation/messytable-part-one').glob('*') if p.is_file()},
      'code_sha256':{n:sha(HERE/n) for n in paths}}
    save(HERE/'protocol.json',protocol)
    save(HERE/'protocol-freeze.json',{n:sha(HERE/n) for n in ['protocol.json','train600.json','boundary-audit.json']})
    emit({'phase':'prepared','train':read(HERE/'train600.json')['summary'],'protocol_sha256':sha(HERE/'protocol.json')})
if __name__=='__main__':main()
