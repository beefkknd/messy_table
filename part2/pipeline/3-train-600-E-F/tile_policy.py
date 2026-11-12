"""Label-blind cached tile admission: geometry, support, then protected merging."""
import itertools
import numpy as np

def overlaps(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    inter=np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),0).prod(2)
    aa=np.maximum(a[:,2:]-a[:,:2],0).prod(1);bb=np.maximum(b[:,2:]-b[:,:2],0).prod(1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-9),inter/np.maximum(np.minimum(aa[:,None],bb[None,:]),1e-9)

def windows(width,height):
    def axis(n):
        stop=max(n-768,0);r=list(range(0,stop+1,576))
        if r[-1]!=stop:r.append(stop)
        return r
    return [(x,y,min(x+768,width),min(y+768,height)) for y in axis(height) for x in axis(width)]

def clearance(box,tile,width,height):
    l,t,r,b=tile;x0,y0,x1,y1=box;d=[]
    if l>0:d.append(x0-l)
    if t>0:d.append(y0-t)
    if r<width:d.append(r-x1)
    if b<height:d.append(b-y1)
    return min(d,default=1e9)

def features(base,tiles,width=1920,height=1080):
    rows=[]
    for ti,tile in enumerate(tiles):
        for di,raw in enumerate(tile['detections']):
            d=raw['mapped'];b=d['box_xyxy'];w,h=b[2]-b[0],b[3]-b[1]
            if w<=0 or h<=0:continue
            rows.append({'det':d,'tile':ti,'ordinal':di,'short':min(w,h),'aspect':max(w,h)/min(w,h),
                         'area':w*h/(width*height),'clear':clearance(b,tile['xyxy'],width,height)})
    rows.sort(key=lambda r:(-r['det']['confidence'],r['tile'],r['ordinal']))
    boxes=[r['det']['box_xyxy'] for r in rows]
    iou,ios=overlaps(boxes,boxes);biou,bios=overlaps(boxes,[d['box_xyxy'] for d in base])
    for i,r in enumerate(rows):
        r['support']=max((x['det']['confidence'] for j,x in enumerate(rows)
                if x['tile']!=r['tile'] and x['clear']>4 and iou[i,j]>=.55),default=0)
        r['support_L']=max((x['det']['confidence'] for j,x in enumerate(rows)
                if x['tile']!=r['tile'] and x['clear']>4 and iou[i,j]>=.6),default=0)
    return {'base':base,'rows':rows,'iou':iou,'ios':ios,'biou':biou,'bios':bios}

def grid():
    configs=[{'name':x,'family':x} for x in ('base','raw','C','G','F','L')]
    for sm,lg,support,edge,ios in itertools.product((.4,.55,.7),(.65,.8),('none','low_confidence'),(4,12),(.85,1.01)):
        configs.append({'name':f'smart_s{sm}_l{lg}_{support}_e{edge}_ios{ios}',
            'family':'smart','small':sm,'large':lg,'support':support,'clearance':edge,'ios':ios})
    return configs

def admit(f,config):
    family=config['family'];base=f['base']
    if family=='base':return list(base)
    candidates=[]
    for i,r in enumerate(f['rows']):
        c=r['det']['confidence']
        if r['clear']<=4:continue
        geom=r['short']>=8 and r['aspect']<=12 and r['area']<=.25 and r['clear']>=12
        if family=='raw':ok=True
        elif family=='C':ok=c>=.8
        elif family=='G':ok=c>=.8 and geom
        elif family=='F':ok=c>=.8 and geom and not (f['bios'][i]>=.85).any()
        elif family=='L':ok=geom and not (f['bios'][i]>=.85).any() and (c>=.8 or (c>=.5 and r['support_L']>=.5))
        else:
            cutoff=config['small'] if r['short']<96 else config['large']
            ok=(c>=cutoff and r['short']>=8 and r['aspect']<=12 and r['area']<=.25
                and r['clear']>=config['clearance'] and not (f['bios'][i]>=config['ios']).any()
                and (config['support']=='none' or c>=.8 or r['support']>=.4))
        if ok:candidates.append(i)
    kept=[]
    for i in candidates:
        if (f['biou'][i]>=.55).any():continue
        if kept and (f['iou'][i,kept]>=.55).any():continue
        if family=='smart' and kept and (f['ios'][i,kept]>=config['ios']).any():continue
        kept.append(i)
    return list(base)+[f['rows'][i]['det'] for i in kept]

def historical(f,kind):
    """Historical gates filter AFTER raw merge, preserving frozen selection order."""
    raw=admit(f,{'family':'raw'});n=len(f['base'])
    if kind=='raw':return raw
    if kind=='base':return list(f['base'])
    by_key={(tuple(r['det']['box_xyxy']),r['det']['confidence']):i for i,r in enumerate(f['rows'])}
    extra=[]
    for d in raw[n:]:
        i=by_key[(tuple(d['box_xyxy']),d['confidence'])];r=f['rows'][i];c=d['confidence']
        geom=r['short']>=8 and r['aspect']<=12 and r['area']<=.25 and r['clear']>=12
        fragment=not (f['bios'][i]>=.85).any()
        ok=(c>=.8 if kind=='C' else c>=.8 and geom if kind=='G'
            else c>=.8 and geom and fragment if kind=='F'
            else geom and fragment and (c>=.8 or (c>=.5 and r['support_L']>=.5)))
        if ok:extra.append(d)
    return list(f['base'])+extra

def apply(f,c):return historical(f,c['family']) if c['family']!='smart' else admit(f,c)
