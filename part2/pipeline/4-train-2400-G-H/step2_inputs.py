"""Export predicted boxes on the preserved Part One sample; no oracle repair."""
from common_new import *
import sys
sys.path.insert(0,str(HERE))
from detect import sources
def main():
    verify();selection=read(SAMPLE/'selection.json');manifest={s['id']:s for s in read(ROOT/'evaluation/messytable-detection-full-test/images.json')['scenes']}
    scenes=[];arms=read(HERE/'protocol.json')['detector_arms']
    for selected in selection['scenes']:
        sid=selected['scene'];s=manifest[sid];values={arm:sources(arm,sid) for arm in arms};views={}
        for v in s['views']:
            cam=v['id'];ds={arm:values[arm][cam]['detections'] for arm in arms}
            assert all(values[arm][cam]['sha256']==v['sha256'] for arm in arms)
            views[cam]={**v,'detections':ds}
        scenes.append({'id':sid,'setup':s['setup'],'difficulty':selected['difficulty'],'views':views})
    save(HERE/'step2/boxes.json',{'arms':arms,'scenes':scenes,'source_selection_sha256':sha(SAMPLE/'selection.json')})
    save(HERE/'step2/box-freeze.json',{'boxes.json':sha(HERE/'step2/boxes.json')})
    emit({'phase':'step2-inputs','scenes':len(scenes),'views':sum(len(s['views']) for s in scenes)})
if __name__=='__main__':main()
