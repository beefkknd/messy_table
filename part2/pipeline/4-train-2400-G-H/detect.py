"""Full-frame resolution factorial; native tiles stay fixed at 768px."""
from common_new import *
import argparse,sys
import numpy as np
sys.path[:0]=[str(OLD),str(ENG)]
from tile_policy import overlaps,windows,clearance

def merge(base,tiles):
    kept=[d for d in base if d['confidence']>.25]
    candidates=[]
    for ti,t in enumerate(tiles):
        for di,row in enumerate(t['detections']):
            d=row['mapped'];b=d['box_xyxy']
            if d['confidence']<=.25 or b[0]>=b[2] or b[1]>=b[3] or clearance(b,t['xyxy'],1920,1080)<=4:continue
            candidates.append((ti,di,d))
    candidates.sort(key=lambda x:(-x[2]['confidence'],x[0],x[1]))
    for _,_,d in candidates:
        if kept and (overlaps([d['box_xyxy']],[p['box_xyxy'] for p in kept])[0]>=.55).any():continue
        kept.append(d)
    return kept

def sources(arm,sid):
    if arm=='old_G':
        p=ROOT/'evaluation/messytable-detection-full-test/predictions'/(sid+'.json')
        s=read(p);return {k:{'sha256':v['sha256'],'detections':v['arms']['fine_tuned_tiles_G']} for k,v in s['views'].items()}
    old=read(OLD/'predictions/sliced600/full-test'/(sid+'.json')) if arm.startswith('600_') else None
    if arm=='600_768':return {k:{'sha256':v['sha256'],'detections':merge(v['base'],v['tiles'])} for k,v in old['views'].items()}
    variant='600' if arm.startswith('600_') else '2400';s=read(HERE/'predictions'/variant/(sid+'.json'));res=arm.split('_')[1]
    return {k:{'sha256':v['sha256'],'detections':merge(v['full_'+res],old['views'][k]['tiles'] if old else v['tiles'])} for k,v in s['views'].items()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['600','2400']);args=ap.parse_args()
    p=verify();runtime()
    import cv2,torch,logging
    from common import pin_ultralytics
    from ultralytics import YOLO
    pin_ultralytics();logging.getLogger('ultralytics').setLevel(logging.ERROR)
    cv2.setNumThreads(1);torch.set_num_threads(8)
    checkpoint=W600 if args.arm=='600' else W2400;model=YOLO(str(checkpoint))
    scenes=read(ROOT/'evaluation/messytable-detection-full-test/images.json')['scenes']
    folder=HERE/'predictions'/args.arm;folder.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    def predict(images,size,floor):
        values=model.predict(images,imgsz=size,conf=floor,iou=.7,max_det=300,device='mps',half=False,rect=True,verbose=False,nms=True)
        assert not model.predictor.model.end2end and next(model.model.parameters()).dtype==torch.float32
        return [[{'box_xyxy':b,'confidence':float(c),'class':r.names[int(cls)]}
           for b,c,cls in zip(r.boxes.xyxy.cpu().numpy().round().astype(int).tolist(),r.boxes.conf.tolist(),r.boxes.cls.tolist()) if c>.25] for r in values]
    for index,s in enumerate(scenes):
        dest=folder/(s['id']+'.json')
        if dest.exists():continue
        pixels=[]
        for v in s['views']:
            path=ROOT/v['path'];assert sha(path)==v['sha256'];im=cv2.imread(str(path),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            assert im.shape[:2]==(1080,1920);pixels.append(im)
        output={res:[] for res in ([1024] if args.arm=='600' else [768,1024])}
        for res in output:
            for start in range(0,len(pixels),6):output[res].extend(predict(pixels[start:start+6],res,.001))
        views={}
        for k,(v,im) in enumerate(zip(s['views'],pixels,strict=True)):
            view={'sha256':v['sha256'],**{'full_'+str(res):rows[k] for res,rows in output.items()}}
            if args.arm=='2400':
                regions=windows(1920,1080);ds=predict([im[t:b,l:r] for l,t,r,b in regions],768,.25);tiles=[]
                for region,detections in zip(regions,ds,strict=True):
                    l,t,r,b=region;rows=[]
                    for d in detections:
                        x0,y0,x1,y1=d['box_xyxy'];local=[max(0,min(r-l,x0)),max(0,min(b-t,y0)),max(0,min(r-l,x1)),max(0,min(b-t,y1))]
                        rows.append({'local':d,'mapped':{**d,'box_xyxy':[l+local[0],t+local[1],l+local[2],t+local[3]]}})
                    tiles.append({'xyxy':list(region),'detections':rows})
                view['tiles']=tiles
            views[v['id']]=view
        save(dest,{'id':s['id'],'setup':s['setup'],'views':views})
        if (index+1)%100==0 or index+1==len(scenes):emit({'phase':'detect','arm':args.arm,'scenes':index+1,'total':len(scenes),'seconds':time.monotonic()-started})
        assert time.time()-p['created_unix']<p['hard_cap_seconds']
    save(folder/'freeze.json',{'checkpoint_sha256':sha(checkpoint),'scenes':{f.name:sha(f) for f in sorted(folder.glob('*.json')) if f.name!='freeze.json'}})

if __name__=='__main__':main()
