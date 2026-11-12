"""Freeze fixed-compute data scaling, resolution factorial, and end-to-end test."""
from common_new import *
from collections import defaultdict
def main():
    assert not (HERE/'protocol.json').exists()
    old=read(OLD/'train600.json');chosen={r['image'] for r in old['frames']}
    native=read(DATA/'labels/train.json')['scenes'];lookup={};pools=defaultdict(list)
    key=lambda s:hashlib.sha256(('scale2400-20261111:'+s).encode()).hexdigest()
    for sid,s in native.items():
        for cam,v in s['cameras'].items():
            path='data/messytable-full/MessyTable/images/'+v['pathname'];lookup[path]=(sid,cam,v)
            if path not in chosen:pools[sid.rsplit('-',1)[0]].append(path)
    for values in pools.values():values.sort(key=key)
    order=sorted(pools,key=key)
    while len(chosen)<2400:
        for setup in order:
            if pools[setup] and len(chosen)<2400:chosen.add(pools[setup].pop())
    frames=[]
    for path in sorted(chosen):
        sid,cam,v=lookup[path];boxes=[o['pos'] for o in v['instances'].values()]
        assert all(0<=b[0]<b[2]<=1920 and 0<=b[1]<b[3]<=1080 for b in boxes)
        frames.append({'image':path,'image_sha256':sha(ROOT/path),'scene':sid,'setup':sid.rsplit('-',1)[0],
                       'camera':cam,'scope':'complete_frame','partition':'train','boxes_xyxy':boxes})
    setups={r['setup'] for r in frames};hashes={r['image_sha256'] for r in frames};boundary={}
    for name,path in [('full_test',ROOT/'evaluation/messytable-detection-full-test/images.json'),
                       ('development100',PROGRAM/'data/cohorts/development100.json'),('calibration30',PROGRAM/'data/cohorts/calibration30.json')]:
        scenes=read(path)['scenes'];assert not setups & {s['setup'] for s in scenes}
        assert not hashes & {v['sha256'] for s in scenes for v in s['views']}
        boundary[name]={'setup_overlap':0,'exact_image_overlap':0,'manifest_sha256':sha(path)}
    save(HERE/'train2400.json',{'frames':frames,'summary':{'photos':2400,'setups':len(setups),
        'boxes':sum(len(r['boxes_xyxy']) for r in frames),'includes_previous600':True}})
    save(HERE/'boundary-audit.json',boundary)
    sources=[A,W600,OLD/'train600.json',OLD/'full-test-matched-controls.json',OLD/'evaluate.py',OLD/'tile_policy.py',OLD/'train.py',
      ROOT/'evaluation/messytable-detection-full-test/images.json',SAMPLE/'selection.json',SAMPLE/'vit/pair-index.json',
      SAMPLE/'vit/app_dist.npy',SAMPLE/'step2/scores/roma_mask.npy',SAMPLE/'step2/scores/blend_roma_vit_reference_dense_first.npy',
      SAMPLE/'step2/metrics/blend_roma_vit_reference_dense_first.json',REPRO/'variant_dense.py',REPRO/'variant_views.py',
      ENG/'partial_loss.py',ENG/'score_detector.py',ROOT/'evaluation/run_full_scene.py']
    # Freeze all existing report files; neither study is rebuilt by this round.
    for folder in [ROOT/'evaluation/messytable-part-one',OLD]:
        sources.extend(p for p in folder.glob('*') if p.is_file() and p.suffix in ('.py','.html','.txt'))
    protocol={'created_unix':time.time(),'hard_cap_seconds':21600,
      'scope':'One 2400-photo supervised run; full-frame inference resolution; frozen Step 2 end-to-end comparison. No teacher/student.',
      'training':{**read(OLD/'protocol.json')['training'],'arms':['sliced2400'],'photos':2400,'start':'Same original A300 as sliced600; fresh optimizer',
        'comparison':'Fixed 3000 updates and 12000 original-photo exposures: 20 epochs for 600 vs 5 for 2400, not convergence-equated. Same seed, crop mixture, schedule and resolution.'},
      'detector_arms':['old_G','600_768','600_1024','2400_768','2400_1024'],
      'inference':{'full_resolutions':[768,1024],'tile_resolution':768,'tile_windows':'same six 768px native windows, stride576',
          'cutoff':.25,'head':'one2many','nms_iou':.7,'merge_iou':.55,'rule':'raw tiles; preserve full-frame prefix; reject artificial edge <=4px; confidence-sorted dedup',
          'comparison':'Reuse exactly the same cached tile observations when changing full-frame resolution. No threshold calibration or tuning.'},
      'step1_population':'Full official test:16749 photos/408937 GT/66996 view pairs/1486614 shared links. Also frozen Development100, descriptive only.',
      'step2':{'population':'Frozen Part One sample100 scenes/900 photos/3600 pairs/78335 true links. Original legacy blind-pair cohort is not scored.',
        'arms':['human_oracle','old_G','600_768','600_1024','2400_768','2400_1024'],
        'methods':['DINOv2 tight CLS cosine, Hungarian max positive margin, cutoff .70','RoMa+SAM/ViT-P3DE 75/25'],
        'strong_score':'Freeze percentile transforms to the original human-box candidate reference distributions; never rerank each new detector population.',
        'strong_assignment':'Exact author quantized KMSolver in pinned Docker. Original reference distance p95 and .70 KM threshold, unchanged.',
        'oracle_replay':'Original human-box distance array must reproduce 74159 correct/6147 wrong before detector-arm scores accepted.',
        'primary':'Correct identity links / all 78335 GT shared links, including undetected endpoints; precision, mapped wrong and unmapped links separately. No candidate-conditional AP presented as end-to-end recall.',
        'score_freeze':'All matrices freeze before applying GT identity mapping for scoring. SAM prompts and crops use predicted coordinates only.'},
      'selection':'All factorial and matching arms fixed now; no dev/test-based model or threshold selection. Final checkpoint only.',
      'limits':['Native test and Part One sample are already exposed, not pristine holdouts.','One training seed.','Native train/val/test split is setup-disjoint, not guaranteed unseen physical products.'],
      'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in set(sources)},
      'code_sha256':{n:sha(HERE/n) for n in ['common_new.py','prepare.py','train2400.py','detect.py','run.py']}}
    save(HERE/'protocol.json',protocol)
    save(HERE/'protocol-freeze.json',{n:sha(HERE/n) for n in ['protocol.json','train2400.json','boundary-audit.json']})
    emit({'phase':'prepared','training':read(HERE/'train2400.json')['summary']})
if __name__=='__main__':main()
