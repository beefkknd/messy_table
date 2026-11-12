"""Calibration-only tuning, one development comparison, then frozen test replay."""
from common_round import *
from collections import Counter, defaultdict
from itertools import combinations
import argparse, copy, time
import numpy as np
from tile_policy import features, apply, grid, windows
from score_detector import view_score

def compact(result):return {k:v for k,v in result.items() if k not in ('cases',)}
def ratios(c):
    return {'recall':c['localized']/c['ground_truth'],'precision':c['localized']/c['detections'] if c['detections'] else None,
            'both_view_coverage':c['shared_localized_both']/c['shared_objects'],'unmapped_per_photo':c['unmapped']/c['images']}

def cached(cohort,minimum=.25):
    bp=ENG/f'round2/inference/A/{cohort}'
    tp=PROGRAM/f'step0-tiles/engineering/inference/{cohort}'
    for directory in (bp,tp):
        for name,digest in read(directory/'freeze.json').items():assert sha(directory/name)==digest
    bases=read(bp/'predictions.json')['scenes'];raw=read(PROGRAM/f'step0-tiles/engineering/raw/{cohort}.json')['scenes']
    raw_by={s['id']:s for s in raw};merged={s['id']:s for s in read(tp/'predictions.json')['scenes']}
    output=[]
    for scene in bases:
        views={}
        for side,v in scene['views'].items():
            base=[d for d in v['detections'] if d['confidence']>.25]
            f=features(base,raw_by[scene['id']]['views'][side])
            # Raw admission replays exact coordinates/order against the immutable cache.
            actual=apply(f,{'family':'raw'})
            expected=merged[scene['id']]['views'][side]['detections']
            assert [(d['box_xyxy'],d['confidence']) for d in actual]==[(d['box_xyxy'],d['confidence']) for d in expected]
            views[side]={'sha256':v['sha256'],'base':[d for d in v['detections'] if d['confidence']>minimum],
                         'tiles':raw_by[scene['id']]['views'][side]}
            if minimum==.25:views[side]['features']=f
        output.append({'id':scene['id'],'setup':scene['setup'],'views':views})
    return output

def make_predictions(cache,config):
    out=[]
    for s in cache:
        views={}
        for side,v in s['views'].items():
            base=[d for d in v['base'] if d['confidence']>config.get('full_threshold',.25)]
            f=v.get('features') if base==v['base'] else None
            if f is None:f=features(base,v['tiles'])
            views[side]={'sha256':v['sha256'],'detections':apply(f,config)}
        out.append({'id':s['id'],'setup':s['setup'],'views':views})
    return out

def score(predictions,cohort,full=False):
    if full:
        scenes=read(ROOT/'evaluation/messytable-detection-full-test/images.json')['scenes']
        labels=read(DATA/'labels/test.json')['scenes']
        truths={sid:{'views':{cam:{'objects_xyxy':{oid:o['pos'] for oid,o in v['instances'].items()}}
                 for cam,v in s['cameras'].items()}} for sid,s in labels.items()}
    else:
        scenes=read(PROGRAM/f'data/cohorts/{cohort}.json')['scenes']
        truths={s['id']:s for s in read(PROGRAM/f'data/scoring/{cohort}.json')['scenes']}
    manifests={s['id']:s for s in scenes};total=Counter();by_setup=defaultdict(Counter);cases=[]
    assert {s['id'] for s in predictions}==set(manifests)
    for pred in predictions:
        sid=pred['id'];scene=manifests[sid];truth=truths[sid]
        assert pred['setup']==scene['setup']
        counts=Counter(images=len(scene['views']),scenes=1);found={};annotated={};perview={}
        for v in scene['views']:
            side=v['id'];raw=pred['views'][side]
            c,ids,allids=view_score(raw,truth['views'][side],v,-1)
            counts.update(c);found[side]=ids;annotated[side]=allids
            for oid,b in truth['views'][side]['objects_xyxy'].items():
                short=min(b[2]-b[0],b[3]-b[1]);size='lt32' if short<32 else '32_to_lt96' if short<96 else 'ge96'
                counts['gt_'+size]+=1;counts['tp_'+size]+=int(oid) in ids
            perview[side]={'localized_ids':sorted(ids),'counts':c}
        for a,b in combinations(found,2):
            shared=annotated[a]&annotated[b]
            counts.update(view_pairs=1,shared_objects=len(shared),shared_localized_both=len(shared&found[a]&found[b]))
        assert counts['localized']+counts['unmapped']==counts['detections']
        total.update(counts);by_setup[scene['setup']].update(counts)
        cases.append({'id':sid,'setup':scene['setup'],'counts':dict(counts),'views':perview})
    return {'counts':dict(total),'metrics':ratios(total),'setups':dict(by_setup),'cases':cases}

def choose(entries,cap):
    feasible=[x for x in entries if x['result']['counts']['unmapped']<=cap]
    assert feasible,'No feasible candidate; do not silently exceed budget'
    return max(feasible,key=lambda x:(x['result']['metrics']['both_view_coverage'],x['result']['metrics']['precision'] or 0,-x['result']['counts']['detections']))

def stage1():
    verify_protocol();assert not (HERE/'stage1-selection.json').exists()
    cache=cached('calibration30');entries=[]
    for c in grid():
        result=score(make_predictions(cache,c),'calibration30')
        entries.append({'config':c,'result':compact(result)})
    cap=next(x['result']['counts']['unmapped'] for x in entries if x['config']['name']=='G')
    reference=choose([x for x in entries if x['config']['family']!='smart'],cap)
    smart=[x for x in entries if x['config']['family']=='smart']
    selected=choose(smart,cap)
    selection={'calibration_unmapped_cap':cap,'reference':reference,'selected':selected,'entries':entries,
               'frozen_before_development_unix':time.time(),'selection_data':'Calibration30 only'}
    save(HERE/'stage1-selection.json',selection)
    save(HERE/'stage1-selection-freeze.json',{'stage1-selection.json':sha(HERE/'stage1-selection.json')})
    dev=cached('development100');evaluation={}
    for label,c in [('reference',reference['config']),('smart',selected['config']),('G',{'family':'G','name':'G'}),('base',{'family':'base','name':'base'}),('raw',{'family':'raw','name':'raw'})]:
        evaluation[label]={'config':c,**score(make_predictions(dev,c),'development100')}
    save(HERE/'stage1-development.json',evaluation)
    emit({'phase':'stage1','calibration_selection':selected['config'],'development':{k:compact(v) for k,v in evaluation.items()}})

def calibrate_existing():
    """Apply the same confidence-grid opportunity to the preexisting detector."""
    verify_protocol();assert not (HERE/'existing-calibrated-reference.json').exists()
    cap=read(HERE/'stage1-selection.json')['calibration_unmapped_cap'];cache=cached('calibration30',.15);entries=[]
    for threshold in read(HERE/'protocol.json')['threshold_grid']:
        for family in ('base','raw','C','G','F','L'):
            cfg={'name':family,'family':family,'full_threshold':threshold}
            entries.append({'config':cfg,'result':compact(score(make_predictions(cache,cfg),'calibration30'))})
    selected=choose(entries,cap)
    save(HERE/'existing-calibrated-reference.json',{'selected':selected,'entries':entries,
        'calibration_unmapped_cap':cap,'frozen_unix':time.time(),
        'reason':'Same calibration threshold grid for old/new weights; no development-driven tuning; original stage1 reference preserved'})
    save(HERE/'existing-calibrated-reference-development.json',{'config':selected['config'],
        **score(make_predictions(cached('development100',.15),selected['config']),'development100')})
    emit({'phase':'existing-reference-calibrated','selected':selected['config'],'calibration_metrics':selected['result']['metrics']})

def infer(arm,cohort,full=False):
    verify_protocol()
    import cv2,torch
    from ultralytics import YOLO
    pin_ultralytics();torch.set_num_threads(8);cv2.setNumThreads(1)
    checkpoint=A if arm=='existingA' else HERE/f'training/{arm}/checkpoint.pt'
    model=YOLO(str(checkpoint))
    scenes=read(ROOT/'evaluation/messytable-detection-full-test/images.json' if full else PROGRAM/f'data/cohorts/{cohort}.json')['scenes']
    folder=HERE/'predictions'/arm/cohort;folder.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    def predict(pixels,floor):
        results=model.predict(pixels,imgsz=768,conf=floor,iou=.7,max_det=300,device='mps',half=False,rect=True,verbose=False,nms=True)
        assert not model.predictor.model.end2end
        assert next(model.model.parameters()).dtype==torch.float32
        return [[{'box_xyxy':b,'class':r.names[int(c)],'confidence':float(s)}
          for b,c,s in zip(r.boxes.xyxy.cpu().numpy().round().astype(int).tolist(),r.boxes.cls.tolist(),r.boxes.conf.tolist())
          if s>.15] for r in results]
    for index,s in enumerate(scenes):
        path=folder/(s['id']+'.json')
        if path.exists():continue
        images=[]
        for v in s['views']:
            assert sha(ROOT/v['path'])==v['sha256']
            im=cv2.imread(str(ROOT/v['path']),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION);assert im.shape[:2]==(v['height'],v['width'])
            images.append(im)
        full_predictions=[]
        for k in range(0,len(images),6):full_predictions.extend(predict(images[k:k+6],.001))
        views={}
        previous=read(ROOT/'evaluation/messytable-detection-full-test/predictions'/(s['id']+'.json')) if arm=='existingA' else None
        for v,im,base in zip(s['views'],images,full_predictions,strict=True):
            if previous is not None:
                oldbase=previous['views'][v['id']]['arms']['fine_tuned']
                # Preserve every archived >.25 box byte-for-byte. Replay is only
                # supplying the previously unavailable .15-.25 suffix.
                extended=list(oldbase)+[d for d in base if d['confidence']<=.25]
                views[v['id']]={'sha256':v['sha256'],'base':extended,'tiles':previous['views'][v['id']]['tile_observations']}
                continue
            regions=windows(v['width'],v['height']);tiles=[]
            predictions=predict([im[t:b,l:r] for l,t,r,b in regions],.25)
            for region,ds in zip(regions,predictions,strict=True):
                l,t,r,b=region;rows=[]
                for d in ds:
                    if d['confidence']<=.25:continue
                    x0,y0,x1,y1=d['box_xyxy'];local=[min(r-l,max(0,x0)),min(b-t,max(0,y0)),min(r-l,max(0,x1)),min(b-t,max(0,y1))]
                    mapped={**d,'box_xyxy':[l+local[0],t+local[1],l+local[2],t+local[3]]}
                    rows.append({'local':d,'mapped':mapped})
                tiles.append({'xyxy':list(region),'detections':rows})
            views[v['id']]={'sha256':v['sha256'],'base':base,'tiles':tiles}
        save(path,{'id':s['id'],'setup':s['setup'],'views':views})
        if (index+1)%50==0 or index+1==len(scenes):emit({'phase':'infer','arm':arm,'cohort':cohort,'scenes':index+1,'total':len(scenes),'seconds':time.monotonic()-start})
        p=read(HERE/'protocol.json')
        if time.time()-p['created_unix']>p['hard_cap_seconds']:raise RuntimeError('Whole-round time cap reached')
    save(folder/'freeze.json',{'checkpoint_sha256':sha(checkpoint),
                             'scenes':{p.name:sha(p) for p in sorted(folder.glob('*.json')) if p.name!='freeze.json'}})

def load_inference(arm,cohort):
    folder=HERE/'predictions'/arm/cohort;freeze=read(folder/'freeze.json')
    rows=[]
    for name,digest in freeze['scenes'].items():
        assert sha(folder/name)==digest;rows.append(read(folder/name))
    return rows

def calibrate(arm):
    verify_protocol();assert not (HERE/f'{arm}-selection.json').exists()
    cache=load_inference(arm,'calibration30');stage=read(HERE/'stage1-selection.json');cap=stage['calibration_unmapped_cap']
    configs=[{'name':'base','family':'base'},{'name':'G','family':'G'},stage['selected']['config'],{'name':'raw','family':'raw'}]
    entries=[]
    for threshold in read(HERE/'protocol.json')['threshold_grid']:
        for cfg in configs:
            c={**cfg,'full_threshold':threshold}
            entries.append({'config':c,'result':compact(score(make_predictions(cache,c),'calibration30'))})
    selected=choose(entries,cap)
    save(HERE/f'{arm}-selection.json',{'selected':selected,'entries':entries,'calibration_unmapped_cap':cap,'frozen_before_development_unix':time.time()})
    emit({'phase':'calibrated','arm':arm,'selection':selected})

def development():
    verify_protocol();assert not (HERE/'development-comparison.json').exists()
    result=read(HERE/'stage1-development.json')
    result['original_reference']=result['reference']
    result['reference']=read(HERE/'existing-calibrated-reference-development.json')
    for arm in ['frames600','sliced600']:
        cfg=read(HERE/f'{arm}-selection.json')['selected']['config']
        result[arm]={'config':cfg,**score(make_predictions(load_inference(arm,'development100'),cfg),'development100')}
    controls={}
    for arm in ('frames600','sliced600'):
        cache=load_inference(arm,'development100');controls[arm]={}
        for family in ('base','raw','G'):
            config={'name':family,'family':family,'full_threshold':.25}
            controls[arm][family]={'config':config,**score(make_predictions(cache,config),'development100')}
    save(HERE/'development-matched-controls.json',controls)
    cap=result['G']['counts']['unmapped'];ref=result['reference']['metrics']['both_view_coverage']
    # No development adjustment or threshold retuning after this comparison.
    for arm in ('smart','frames600','sliced600'):
        r=result[arm];gain=100*(r['metrics']['both_view_coverage']-ref)
        r['decision']={'gain_pp_vs_reference':gain,'within_unmapped_cap':r['counts']['unmapped']<=cap,
                       'meets_preregistered_target':gain>=3 and r['counts']['unmapped']<=cap}
    save(HERE/'development-comparison.json',result)
    selected={a:result[a]['config'] for a in ('reference','smart','frames600','sliced600')}
    save(HERE/'final-test-freeze.json',{'created_unix':time.time(),'configs':selected,
        'matched_controls':[{'family':f,'full_threshold':.25} for f in ('base','raw','G')],
        'development_sha256':sha(HERE/'development-comparison.json'),'protocol_sha256':sha(HERE/'protocol.json'),
        'warning':'Previously exposed official test; one final evaluation, no feedback tuning'})
    emit({'phase':'development-complete','arms':{k:compact(v) for k,v in result.items()},'full_test_settings_frozen':True})
    if not any(result[a]['decision']['meets_preregistered_target'] for a in ('smart','frames600','sliced600')):
        save(HERE/'completion.json',{'status':'stopped_by_development_gate','completed_unix':time.time(),
            'reason':'None gained >=3 pp coverage within historical G unmapped budget. Original stop rule; no new full-test inference or scoring.',
            'development_sha256':sha(HERE/'development-comparison.json')})
        emit({'phase':'stopped','reason':'No approach passed the development gate','new_full_test_evaluation':False})
        raise SystemExit(10)

def full_test():
    verify_protocol();assert not (HERE/'full-test-comparison.json').exists()
    frozen=read(HERE/'final-test-freeze.json');assert frozen['development_sha256']==sha(HERE/'development-comparison.json')
    manifest=read(ROOT/'evaluation/messytable-detection-full-test/images.json')['scenes'];cache=[]
    old_folder=ROOT/'evaluation/messytable-detection-full-test/predictions'
    sourcefreeze=read(ROOT/'evaluation/messytable-detection-full-test/prediction-freeze.json')['scenes']
    reference_config=frozen['configs']['reference']
    needs_raw_full=reference_config.get('full_threshold',.25)<.25
    # Historical full-test files retained only >.25 full-frame predictions. If
    # calibration selects a lower floor, inference must be explicitly replayed;
    # never pretend that the absent .15-.25 predictions are available.
    if needs_raw_full:
        if not (HERE/'predictions/existingA/full-test/freeze.json').exists():infer('existingA','full-test',True)
        cache=load_inference('existingA','full-test')
    else:cache=[]
    for s in manifest:
        path=old_folder/(s['id']+'.json');assert sha(path)==sourcefreeze[path.name]
        old=read(path);views={}
        for side,v in old['views'].items():
            views[side]={'sha256':v['sha256'],'base':v['arms']['fine_tuned'],'tiles':v['tile_observations']}
        if not needs_raw_full:cache.append({'id':s['id'],'setup':s['setup'],'views':views})
    result={}
    for arm in ('reference','smart'):
        result[arm]={'config':frozen['configs'][arm],**score(make_predictions(cache,frozen['configs'][arm]),'full-test',full=True)}
        emit({'phase':'full-test-scored','arm':arm,'metrics':result[arm]['metrics']})
    del cache
    controls={}
    for arm in ('frames600','sliced600'):
        cache=load_inference(arm,'full-test')
        result[arm]={'config':frozen['configs'][arm],**score(make_predictions(cache,frozen['configs'][arm]),'full-test',full=True)}
        controls[arm]={}
        for config in frozen['matched_controls']:
            controls[arm][config['family']]={'config':config,**score(make_predictions(cache,config),'full-test',full=True)}
        emit({'phase':'full-test-scored','arm':arm,'metrics':result[arm]['metrics']});del cache
    save(HERE/'full-test-matched-controls.json',controls)
    save(HERE/'full-test-comparison.json',result)
    save(HERE/'full-test-scoring-freeze.json',{'full-test-comparison.json':sha(HERE/'full-test-comparison.json'),
                                           'final-test-freeze.json':sha(HERE/'final-test-freeze.json')})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['stage1','reference','infer','calibrate','development','full-test']);ap.add_argument('--arm');ap.add_argument('--cohort');args=ap.parse_args()
    if args.phase=='stage1':stage1()
    elif args.phase=='reference':calibrate_existing()
    elif args.phase=='infer':infer(args.arm,args.cohort,args.cohort=='full-test')
    elif args.phase=='calibrate':calibrate(args.arm)
    elif args.phase=='development':development()
    else:full_test()
