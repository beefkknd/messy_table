"""Same full official test, same IoU policy and fixed confidence for every arm."""
from common_new import *
import sys
sys.path[:0]=[str(OLD),str(ENG)]
from detect import sources
from evaluate import score
def main():
    verify();manifest=read(ROOT/'evaluation/messytable-detection-full-test/images.json')['scenes'];results={}
    for arm in read(HERE/'protocol.json')['detector_arms']:
        predictions=[{'id':s['id'],'setup':s['setup'],'views':sources(arm,s['id'])} for s in manifest]
        results[arm]=score(predictions,'full-test',full=True)
        emit({'phase':'step1-scored','arm':arm,'metrics':results[arm]['metrics']})
    old=read(OLD/'full-test-matched-controls.json')['sliced600']['raw']['counts']
    for key in ['localized','ground_truth','detections','unmapped','shared_localized_both','shared_objects']:
        assert results['600_768']['counts'][key]==old[key],key
    save(HERE/'step1-results.json',results)
    controls={}
    for arm in ['600_768','600_1024','2400_768','2400_1024']:
        predictions=[]
        variant,res=arm.split('_')
        for s in manifest:
            if arm=='600_768':
                raw=read(OLD/'predictions/sliced600/full-test'/(s['id']+'.json'))
                views={k:{'sha256':v['sha256'],'detections':[d for d in v['base'] if d['confidence']>.25]} for k,v in raw['views'].items()}
            else:
                raw=read(HERE/'predictions'/variant/(s['id']+'.json'))
                views={k:{'sha256':v['sha256'],'detections':v['full_'+res]} for k,v in raw['views'].items()}
            predictions.append({'id':s['id'],'setup':s['setup'],'views':views})
        controls[arm]=score(predictions,'full-test',full=True)
    save(HERE/'step1-fullframe-controls.json',controls)
    save(HERE/'step1-scoring-freeze.json',{n:sha(HERE/n) for n in ['step1-results.json','step1-fullframe-controls.json']})
if __name__=='__main__':main()
