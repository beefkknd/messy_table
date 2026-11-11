"""Frozen no-training anchors, calibrated geometry, and local perspective study."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parent
ROOT = HERE.parents[3]
STAGED = VALIDATION / 'staged-verifier'
CACHE = VALIDATION / 'edge-mask-study/engineering/full'
sys.path.insert(0, str(STAGED / 'data'))
from load_evidence import load_cohort, verify
from verifier_policies import appearance_weights, assign_weights, SAM

def read(p):
    return json.loads(Path(p).read_bytes())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def hom(points):
    return np.column_stack((points, np.ones(len(points))))

def project(H, xy):
    out = hom(xy) @ H.T
    return out[:, :2] / out[:, 2:3]

def fundamental(cameras):
    def pose(c):
        v = c['camera_to_reference']
        out = np.eye(4)
        out[:3, :3] = Rotation.from_euler('xyz', v[3:]).as_matrix()
        out[:3, 3] = v[:3]
        return out
    motion = np.linalg.solve(pose(cameras['B']), pose(cameras['A']))
    x, y, z = motion[:3, 3]
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    ka, kb = [np.asarray(cameras[s]['intrinsics']).reshape(3, 3) for s in 'AB']
    f = np.linalg.inv(kb).T @ skew @ motion[:3, :3] @ np.linalg.inv(ka)
    return f / np.linalg.norm(f)

def corners(box, pad=0):
    x1, y1, x2, y2 = np.asarray(box, float) + [-pad, -pad, pad, pad]
    return np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]])

def box_geometry(F, a, b):
    if F is None:
        return {'status':'unknown_no_calibration', 'reject':False}
    pads = [max(16., .10*np.linalg.norm(np.asarray(box)[2:]-np.asarray(box)[:2])) for box in (a,b)]
    values = hom(corners(a, pads[0])) @ F.T @ hom(corners(b, pads[1])).T
    low, high = float(values.min()), float(values.max())
    reject = low > 1e-10 or high < -1e-10
    return {'status':'contradiction' if reject else 'compatible_not_identity', 'reject':reject,
            'padding_px':pads, 'bilinear_range':[low,high]}

def dedup(matches):
    result=[]
    for m in sorted(matches, key=lambda r:(max(r['descriptor_ratio_A_to_B'],r['descriptor_ratio_B_to_A']),r['A_keypoint'],r['B_keypoint'])):
        if any(np.linalg.norm(np.array(m['A_native_xy'])-p['A_native_xy'])<2 or
               np.linalg.norm(np.array(m['B_native_xy'])-p['B_native_xy'])<2 for p in result):
            continue
        result.append(m)
    return result

def support(xy, size):
    w,h=size
    xy=np.asarray(xy,np.float32)
    if len(xy)<3:
        return {'spread':[0.,0.], 'hull':0.}
    return {'spread':(np.ptp(xy,axis=0)/[w,h]).tolist(),
            'hull':float(cv2.contourArea(cv2.convexHull(xy))/(w*h))}

def spread_ok(info):
    return min(info['spread'])>=.20 and info['hull']>=.02

def detail_info(matches, sizes):
    unique=dedup(matches)
    good=[m for m in unique if m['inlier']]
    spatial={s:support([m[s+'_native_xy'] for m in good],sizes[s]) for s in 'AB'}
    valid=len(good)>=6 and len(good)/max(1,len(unique))>=.45 and all(spread_ok(x) for x in spatial.values())
    return {'eligible':valid,'unique_matches':len(unique),'unique_affine_inliers':len(good),'support':spatial}

def epipolar_points(F, matches, boxes):
    if F is None or not matches:
        return None
    a,b=[hom(np.array([m[s+'_native_xy'] for m in matches])+np.array(boxes[s][:2])) for s in 'AB']
    lb,la=a@F.T,b@F
    numerator=np.abs(np.sum(lb*b,axis=1))
    distances=np.maximum(numerator/np.maximum(np.linalg.norm(lb[:,:2],axis=1),1e-12),
                         numerator/np.maximum(np.linalg.norm(la[:,:2],axis=1),1e-12))
    return {'n':len(matches),'median_max_point_line_px':float(np.median(distances)),
            'p90_px':float(np.quantile(distances,.90)),'fraction_within16px':float(np.mean(distances<=16))}

def fit_surface(matches, sizes, key):
    ms=dedup(matches)
    result={'status':'too_few_unique_matches','unique_matches':len(ms)}
    if len(ms)<12:
        return result
    xy={s:np.asarray([m[s+'_native_xy'] for m in ms],float) for s in 'AB'}
    scale={s:224/max(sizes[s]) for s in 'AB'}
    norm={s:xy[s]*scale[s] for s in 'AB'}
    seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
    order=np.random.default_rng(seed).permutation(len(ms))
    validation=order[::4];train=np.array([i for j,i in enumerate(order) if j%4!=0])
    cv2.setRNGSeed(0)
    H,flags=cv2.findHomography(norm['B'][train],norm['A'][train],cv2.RANSAC,4.,maxIters=2000,confidence=.995)
    result.update(train_indices=train.tolist(),validation_indices=validation.tolist(),
                  matches=[{'A':m['A_native_xy'],'B':m['B_native_xy']} for m in ms])
    if H is None or flags is None or not np.isfinite(H).all() or abs(np.linalg.det(H))<1e-12:
        return {**result,'status':'fit_unavailable'}
    flags=flags.ravel().astype(bool)
    if flags.sum()<6 or flags.mean()<.60:
        return {**result,'status':'insufficient_training_support','train_inliers':int(flags.sum())}
    err=np.maximum(np.linalg.norm(project(H,norm['B'][validation])-norm['A'][validation],axis=1),
                   np.linalg.norm(project(np.linalg.inv(H),norm['A'][validation])-norm['B'][validation],axis=1))
    result.update(train_inliers=int(flags.sum()), validation_errors_224=err.tolist(),
                  validation_pass_fraction=float(np.mean(err<=4)),validation_median_224=float(np.median(err)))
    if not np.isfinite(err).all() or np.mean(err<=4)<.75 or np.median(err)>3:
        return {**result,'status':'heldout_validation_failed'}
    accepted=np.concatenate((train[flags],validation[err<=4]))
    spatial={s:support(xy[s][accepted],sizes[s]) for s in 'AB'}
    result['support']=spatial
    if not all(spread_ok(x) for x in spatial.values()):
        return {**result,'status':'support_too_local'}
    native=np.diag([1/scale['A'],1/scale['A'],1])@H@np.diag([scale['B'],scale['B'],1])
    native/=native[2,2]
    bc=corners([0,0,*sizes['B']]);denom=(hom(bc)@native.T)[:,2]
    if not ((denom>1e-8).all() or (denom < -1e-8).all()):
        return {**result,'status':'projective_horizon_in_crop'}
    projected=project(native,bc).astype(np.float32)
    area=cv2.contourArea(projected,oriented=True)/(sizes['A'][0]*sizes['A'][1])
    if not np.isfinite(projected).all() or not cv2.isContourConvex(projected) or not .05<=area<=20:
        return {**result,'status':'unstable_projected_crop','projected_area_ratio':float(area)}
    return {**result,'status':'geometry_valid','H_B_to_A':native.tolist(),'accepted_indices':accepted.tolist(),
            'projected_area_ratio':float(area)}

@lru_cache(maxsize=8)
def source(path, digest):
    assert sha(ROOT/path)==digest
    return np.array(Image.open(ROOT/path).convert('RGB'))

def native_crop(row, masks, side, index):
    v=row['views'][side];i=v['global_indices'].index(index);box=v['boxes_xyxy'][i]
    meta=masks[side]['crops'][i]['sam']
    if meta['fallback_original_rgb'] or meta['status']!='usable':
        return None,None
    path=CACHE/meta['mask_path']; assert sha(path)==meta['mask_sha256']
    mask=np.array(Image.open(path))==255
    x1,y1,x2,y2=box;pixels=source(v['path'],v['sha256'])[y1:y2,x1:x2].copy()
    assert pixels.shape[:2]==mask.shape
    return pixels,mask

def patch_views(row,masks,a,b,fit):
    pixels={};m={}
    for s,i in zip('AB',(a,b)):
        pixels[s],m[s]=native_crop(row,masks,s,i)
    if any(v is None for v in pixels.values()):
        return None,{'status':'unusable_mask'}
    h=np.asarray(fit['H_B_to_A']);accepted=fit['accepted_indices']
    supports={}
    for s in 'AB':
        xy=np.asarray([fit['matches'][i][s] for i in accepted],np.float32)
        hull=np.round(cv2.convexHull(xy)).astype(np.int32)
        polygon=np.zeros(m[s].shape,np.uint8);cv2.fillConvexPoly(polygon,hull,1)
        supports[s]=(polygon.astype(bool)&m[s]).astype(np.uint8)
    size_a=pixels['A'].shape[1::-1];size_b=pixels['B'].shape[1::-1]
    warped_support=cv2.warpPerspective(supports['B'],h,size_a,flags=cv2.INTER_NEAREST,borderValue=0)
    common_a=supports['A']&warped_support
    common_b=supports['B']&cv2.warpPerspective(common_a,np.linalg.inv(h),size_b,flags=cv2.INTER_NEAREST,borderValue=0)
    common_a&=cv2.warpPerspective(common_b,h,size_a,flags=cv2.INTER_NEAREST,borderValue=0)
    coverage=[float(common_a.sum()/m['A'].sum()),float(common_b.sum()/m['B'].sum())]
    if min(coverage)<.10 or min(common_a.sum(),common_b.sum())<64:
        return None,{'status':'insufficient_common_surface','foreground_coverage':coverage}
    boxes={}
    for s,mask in zip('AB',(common_a,common_b)):
        yy,xx=np.where(mask);boxes[s]=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
        if min(boxes[s][2]-boxes[s][0],boxes[s][3]-boxes[s][1])<16:
            return None,{'status':'common_patch_too_small','patch_boxes':boxes,'foreground_coverage':coverage}
    warped_rgb=cv2.warpPerspective(pixels['B'],h,size_a,flags=cv2.INTER_LINEAR,borderValue=(128,128,128))
    images=[]
    for rgb,mask,s in ((pixels['A'],common_a,'A'),(pixels['B'],common_b,'B'),(warped_rgb,common_a,'A')):
        canvas=np.where(mask[:,:,None].astype(bool),rgb,np.uint8(128)).astype(np.uint8)
        x1,y1,x2,y2=boxes[s];images.append(canvas[y1:y2,x1:x2].copy())
    return images,{'status':'patch_valid','foreground_coverage':coverage,'patch_boxes':boxes,
                   'common_pixels':[int(common_a.sum()),int(common_b.sum())]}

def unpack_box(row,side,index):
    return row['views'][side]['boxes_xyxy'][row['views'][side]['global_indices'].index(index)]

def pick(ids,scores,occupied=()):
    return assign_weights(ids,{edge:score-.70 for edge,score in scores.items() if score>.70},occupied)

def exact_count(ids,scores,k):
    if not k:return []
    n,m=len(ids['A']),len(ids['B']);positions={s:{i:p for p,i in enumerate(ids[s])} for s in 'AB'}
    matrix=np.full((n,m),-1e6,float)
    for (a,b),value in scores.items():matrix[positions['A'][a],positions['B'][b]]=value
    weights=np.concatenate((matrix,np.full((n,n-k),3.)),axis=1)
    aa,bb=linear_sum_assignment(weights,maximize=True)
    pairs=[(ids['A'][a],ids['B'][b]) for a,b in zip(aa,bb) if b<m]
    assert len(pairs)==k and all(p in scores for p in pairs)
    return pairs

def decisions(evidence, features):
    results={}
    for cid,c in evidence.items():
        edge_rows=features['cases'][cid]['edges'];local={tuple(e['edge']):e for e in edge_rows}
        ids=c['ids'];sam=c['residual_matrices'][SAM]
        full={(a,b):float(sam[i,j]) for i,a in enumerate(ids['A']) for j,b in enumerate(ids['B'])}
        slate={e:full[e] for e in local};allowed={e:v for e,v in slate.items() if not local[e]['geometry']['reject']}
        appearance=appearance_weights(c['appearance'],'easy',{'floor':.70,'margin':.06})
        anchors=assign_weights(ids,{e:w for e,w in appearance.items() if local[e]['detail']['eligible']})
        surviving=[e for e in anchors if not local[e]['geometry']['reject']]
        refill=pick(ids,allowed,surviving);round2=surviving+refill
        warped={e:r['warped_cosine'] for e,r in local.items() if 'warped_cosine' in r}
        native={e:r['unwarped_patch_cosine'] for e,r in local.items() if 'unwarped_patch_cosine' in r}
        geometry_warp={e:v for e,v in warped.items() if not local[e]['geometry']['reject']}
        geometry_native={e:v for e,v in native.items() if not local[e]['geometry']['reject']}
        joint=dict(allowed)
        for e,v in geometry_warp.items():joint[e]=max(joint[e],v)
        results[cid]={'arms':{
            'historical_only':[], 'sam_full':pick(ids,full), 'sam_slate':pick(ids,slate),
            'appearance_anchors':assign_weights(ids,appearance),'round1_anchors':anchors,
            'anchors_sam_no_geometry':anchors+pick(ids,slate,anchors),
            'geometry_sam_no_anchors':pick(ids,allowed),'round2_geometry':round2,
            'round3_perspective':round2+pick(ids,geometry_warp,round2),
            'round3_unwarped_patch':round2+pick(ids,geometry_native,round2),
            'joint_geometry_perspective_no_locks':pick(ids,joint),
            'perspective_only':pick(ids,warped),
            'joint_matched_round2_count':exact_count(ids,joint,len(round2))},
            'revoked_anchors':[e for e in anchors if e not in surviving],
            'round2_refill':refill,'round3_added':pick(ids,geometry_warp,round2),
            'unused_indices':ids,'counts':{'full_edges':len(full),'slate':len(slate),'geometry_rejected':len(slate)-len(allowed),
                'valid_surface_edges':len(warped),'geometry_allowed_valid_surfaces':len(geometry_warp)}}
    return results

def tests():
    # Known pinhole cameras observing random positive-depth 3D points.
    ka=np.array([[800,0,320],[0,800,240],[0,0,1.]])
    cam={'A':{'intrinsics':ka.ravel().tolist(),'camera_to_reference':[0,0,0,0,0,0]},
         'B':{'intrinsics':ka.ravel().tolist(),'camera_to_reference':[1,0,0,0,0,0]}}
    f=fundamental(cam);rng=np.random.default_rng(42);xyz=rng.uniform([-1,-1,4],[1,1,8],(100,3))
    p=xyz@ka.T;q=(xyz-[1,0,0])@ka.T;p=p[:,:2]/p[:,2:];q=q[:,:2]/q[:,2:]
    assert np.max(np.abs(np.sum((hom(p)@f.T)*hom(q),axis=1)))<1e-8
    for a,b in zip(p,q):assert not box_geometry(f,[*a,*a],[*b,*b])['reject']
    assert box_geometry(f,[0,0,50,50],[0,300,50,350])['reject']
    assert not box_geometry(None,[0,0,1,1],[3,3,5,5])['reject']
    # Planar warp recovered from independent train/validation coordinates.
    grid=np.array([(x,y) for y in [20,60,100,140,180] for x in [20,60,100,140,180]],float)
    h=np.array([[.88,.07,8],[-.03,.92,7],[.0002,.0001,1.]])
    target=project(h,grid)
    matches=[{'A_native_xy':a.tolist(),'B_native_xy':b.tolist(),'A_keypoint':i,'B_keypoint':i,
              'descriptor_ratio_A_to_B':.2,'descriptor_ratio_B_to_A':.2,'inlier':True} for i,(a,b) in enumerate(zip(target,grid))]
    fit=fit_surface(matches,{'A':(224,224),'B':(224,224)},'synthetic')
    assert fit['status']=='geometry_valid',fit
    assert set(fit['train_indices']).isdisjoint(fit['validation_indices'])
    assert np.max(np.abs(project(np.array(fit['H_B_to_A']),grid)-target))<1e-3
    duplicate=matches+[dict(matches[0],A_keypoint=99,B_keypoint=99)]
    assert len(dedup(duplicate))==len(matches)
    # Rectangular exact-K diagnostic against brute-force enumeration.
    import itertools
    checked=0
    for n,m in ((2,3),(3,2),(3,3)):
        ids={'A':list(range(n)),'B':list(range(m))}
        for _ in range(12):
            a=rng.uniform(-1,1,(n,m));scores={(i,j):float(a[i,j]) for i in range(n) for j in range(m)}
            for k in range(min(n,m)+1):
                got=exact_count(ids,scores,k)
                optimum=max(sum(a[i,j] for i,j in zip(rows,cols)) for rows in itertools.combinations(range(n),k) for cols in itertools.permutations(range(m),k))
                assert abs(sum(scores[p] for p in got)-optimum)<1e-9
                checked+=1
    return {'synthetic_3d_points':100,'known_homography_recovered':True,'train_validation_disjoint':True,
            'spatial_deduplication':True,'exact_k_bruteforce_fixtures':checked}

def run():
    out=HERE/'run';out.mkdir(exist_ok=False)
    cv2.setNumThreads(1);started=time.monotonic()
    protocol=read(HERE/'PROTOCOL.json')
    files=[HERE/'PROTOCOL.json',Path(__file__),STAGED/'PROTOCOL.json',
           STAGED/'data/load_evidence.py',STAGED/'data/verifier_policies.py',
           STAGED/'engineering/local-evidence/original100.json',ROOT/'data/messytable-100/camera-calibration.json']
    freeze={str(p.relative_to(ROOT)):sha(p) for p in files}
    save(out/'implementation-freeze.json',freeze)
    checks=tests();save(out/'synthetic-checks.json',checks)
    inventory,evidence,verified,_=load_cohort('original100',sha(STAGED/'PROTOCOL.json'))
    rows={r['id']:r for r in inventory['scenes']}
    masks=read(CACHE/'mask-index.json')['scenes']
    local=read(STAGED/'engineering/local-evidence/original100.json')
    cameras=read(ROOT/'data/messytable-100/camera-calibration.json')
    doc={'protocol_sha256':sha(HERE/'PROTOCOL.json'),'source_freezes':verified,'labels_read':False,'training':False,'cases':{}}
    images=[];image_keys=[];counts=Counter();native_checks=[]
    for scene in local['scenes']:
        cid=scene['id'];row=rows[cid];cal=cameras.get(str(int(cid.split('-')[1])));F=fundamental(cal) if cal else None
        records=[]
        for e in scene['shortlist_edges']:
            a,b=e['A'],e['B'];boxes={s:unpack_box(row,s,i) for s,i in zip('AB',(a,b))}
            sizes={s:(boxes[s][2]-boxes[s][0],boxes[s][3]-boxes[s][1]) for s in 'AB'}
            geo=box_geometry(F,boxes['A'],boxes['B']);detail=detail_info(e['local']['matches'],sizes)
            r={'edge':[a,b],'geometry':geo,'detail':detail,'cosines':e['cosines']}
            if detail['eligible']:
                ms=[m for m in dedup(e['local']['matches']) if m['inlier']]
                r['calibration_residual']=epipolar_points(F,ms,boxes)
                if r['calibration_residual'] is not None:native_checks.append({'case':cid,'edge':[a,b],**r['calibration_residual']})
            fit=fit_surface(e['local']['matches'],sizes,f'{cid}-{a}-{b}')
            if fit['status']=='geometry_valid':
                patches,patch_info=patch_views(row,masks[cid],a,b,fit)
                fit['patch']=patch_info
                if patches is not None:
                    folder=out/'patches'/f'{cid}-a{a}-b{b}';folder.mkdir(parents=True)
                    paths=[]
                    for label,pixels in zip(('A_surface','B_native_surface','B_rectified_surface'),patches):
                        path=folder/f'{label}.png';Image.fromarray(pixels).save(path);paths.append(str(path.relative_to(HERE)))
                        images.append(Image.fromarray(pixels));image_keys.append((cid,a,b,label))
                    r['patch_paths']=paths
                    r['patch_hashes']=[sha(HERE/p) for p in paths]
            r['surface']=fit;records.append(r);counts[fit['status']]+=1
        doc['cases'][cid]={'calibration_available':cal is not None,'F':F.tolist() if F is not None else None,'edges':records}
    doc['pre_encoding_seconds']=time.monotonic()-started
    save(out/'geometry-and-surfaces.json',doc)
    save(out/'calibration-residual-audit.json',{'readout':'max of two point-to-line distances in native pixels; unverified SIFT correspondences, no truth selection','cases':native_checks})
    print('Surface fit statuses',dict(counts),'patch triples',len(images)//3,flush=True)
    if images:
        # Import the unchanged inference entry point without generating SAM masks.
        sys.path.insert(0,str(ROOT/'everyday-identity/scripts'))
        sys.path.insert(0,str(ROOT/'evaluation'))
        import torch
        from transformers import AutoImageProcessor,AutoModel
        from runtime_assets import dinov2_source,dinov2_weights
        from run_full_scene import encode_crops
        torch.set_num_threads(8)
        assert sha(dinov2_weights())==read(CACHE/'plan.json')['dino_weights_sha256']
        device='mps' if torch.backends.mps.is_available() else 'cpu'
        processor=AutoImageProcessor.from_pretrained(dinov2_source(),local_files_only=True)
        model=AutoModel.from_pretrained(dinov2_source(),local_files_only=True).to(device).eval().requires_grad_(False)
        feature_start=time.monotonic()
        vectors=np.concatenate([encode_crops(images[i:i+64],processor,model,device) for i in range(0,len(images),64)])
        if device=='mps':torch.mps.synchronize()
        doc['encoding_seconds']=time.monotonic()-feature_start
        np.savez_compressed(out/'surface-embeddings.npz',vectors=vectors)
        assert vectors.shape==(len(images),384) and np.isfinite(vectors).all()
        for i in range(0,len(images),3):
            cid,a,b,_=image_keys[i];r=next(x for x in doc['cases'][cid]['edges'] if x['edge']==[a,b])
            r['unwarped_patch_cosine']=float(vectors[i]@vectors[i+1]);r['warped_cosine']=float(vectors[i]@vectors[i+2]);r['embedding_start_row']=i
        save(out/'embedding-index.json',{'keys':image_keys,'weights_sha256':sha(dinov2_weights()),'device':device})
    doc['wall_seconds']=time.monotonic()-started
    save(out/'features.json',doc)
    frozen_decisions=decisions(evidence,doc)
    assert len(frozen_decisions)==100
    for cid,d in frozen_decisions.items():
        for arm,edges in d['arms'].items():
            assert len(set(edges))==len(edges)==len({a for a,b in edges})==len({b for a,b in edges})
            assert all(a in evidence[cid]['ids']['A'] and b in evidence[cid]['ids']['B'] for a,b in edges)
    save(out/'decisions.json',{'protocol_sha256':sha(HERE/'PROTOCOL.json'),'labels_read':False,'cases':frozen_decisions})
    outputs=[out/p for p in ('implementation-freeze.json','features.json','decisions.json','geometry-and-surfaces.json','calibration-residual-audit.json','synthetic-checks.json')]
    outputs += [p for p in (out/'surface-embeddings.npz',out/'embedding-index.json') if p.exists()]
    save(out/'freeze.json',{p.name:sha(p) for p in outputs})
    print('Frozen outputs, no evaluation truth read.',round(doc['wall_seconds'],2),'seconds',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--test',action='store_true');args=parser.parse_args()
    if args.test:print(tests())
    else:run()
