"""Matched supervised recipe; one seed; no validation-based checkpoint selection."""
from common_round import *
import argparse, copy, math, random
import cv2, numpy as np, torch
from ultralytics import YOLO
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils import DEFAULT_CFG
from train_detector import sample_stream, atomic_checkpoint
from partial_loss import PartialDetectionLoss
from tile_policy import windows

def preprocess(row,size,tile=None):
    im=cv2.imread(str(ROOT/row['image']),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
    assert im is not None and im.shape[:2]==(1080,1920)
    boxes=np.asarray(row['boxes_xyxy'],np.float32).reshape(-1,4)
    if tile is not None:
        l,t,r,b=tile;im=im[t:b,l:r]
        boxes=boxes-[l,t,l,t]
        boxes=boxes.clip([0,0,0,0],[r-l,b-t,r-l,b-t])
        boxes=boxes[((boxes[:,2:]-boxes[:,:2])>0).all(1)]
    h,w=im.shape[:2];scale=size/max(h,w);rw,rh=round(w*scale),round(h*scale)
    l,t=(size-rw)//2,(size-rh)//2
    pixels=cv2.resize(im,(rw,rh),interpolation=cv2.INTER_LINEAR)
    pixels=cv2.copyMakeBorder(pixels,t,size-rh-t,l,size-rw-l,cv2.BORDER_CONSTANT,value=(114,114,114))
    boxes=boxes*[rw/w,rh/h,rw/w,rh/h]+[l,t,l,t]
    xywh=np.column_stack(((boxes[:,:2]+boxes[:,2:])/2,boxes[:,2:]-boxes[:,:2]))/size
    return pixels[:,:,::-1].transpose(2,0,1).copy(),xywh.astype(np.float32)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['frames600','sliced600']);args=ap.parse_args()
    p=verify_protocol();pin_ultralytics();cfg=p['training'];out=HERE/'training'/args.arm
    assert not out.exists();out.mkdir(parents=True)
    started=time.monotonic();torch.set_num_threads(8);cv2.setNumThreads(1);torch.manual_seed(cfg['seed'])
    rows=read(HERE/'train600.json')['frames'];stream=sample_stream(rows,cfg['seed'],cfg['additional_updates']*4)
    rng=random.Random(cfg['seed']+1);tile_schedule=[rng.randrange(6) for _ in stream]
    save(out/'run-plan.json',{'protocol_sha256':sha(HERE/'protocol.json'),'arm':args.arm,'input_checkpoint_sha256':sha(A),
      'images':[[r['image'] for r in stream[i:i+4]] for i in range(0,len(stream),4)],'tile_indices':tile_schedule,
      'tile_slots':'(slot+step)%2==0 for sliced; same original-image schedule in both arms','selection':'final checkpoint only'})
    model=YOLO(str(A)).model.to('mps').float().train();model.end2end=True
    # YOLO's checkpoint loader converts args to a dict for inference; training
    # losses require the attribute-style namespace used by the original trainer.
    model.args=copy.deepcopy(DEFAULT_CFG)
    for name,param in model.named_parameters():param.requires_grad_('dfl.conv.weight' not in name)
    criterion=E2ELoss(model,loss_fn=PartialDetectionLoss)
    opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=cfg['peak_lr'],weight_decay=.0005)
    logs=[];regions=windows(1920,1080)
    for step in range(cfg['additional_updates']):
        began=time.monotonic();offset=4*step;batchrows=stream[offset:offset+4]
        samples=[preprocess(row,1024,regions[tile_schedule[offset+j]] if args.arm=='sliced600' and (j+step)%2==0 else None)
                 for j,row in enumerate(batchrows)]
        images,boxes=zip(*samples);idx=np.repeat(np.arange(4),[len(b) for b in boxes])
        batch={'img':torch.from_numpy(np.stack(images)).to('mps',dtype=torch.float32)/255,
          'bboxes':torch.from_numpy(np.concatenate(boxes)).to('mps'),'batch_idx':torch.from_numpy(idx).to('mps'),
          'cls':torch.zeros((len(idx),1),device='mps'),'incomplete':torch.zeros(4,device='mps',dtype=torch.bool)}
        if step<100:lr=cfg['peak_lr']*(.1+.9*(step+1)/100)
        else:
            fraction=(step-100)/(cfg['additional_updates']-101)
            lr=cfg['final_lr']+.5*(cfg['peak_lr']-cfg['final_lr'])*(1+math.cos(math.pi*fraction))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);loss,parts=criterion(model(batch['img']),batch);objective=loss.sum()/4
        assert torch.isfinite(objective),'Nonfinite loss'
        objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);opt.step();torch.mps.synchronize()
        logs.append({'step':step+1,'loss':float(objective.detach()),'lr':lr,'seconds':time.monotonic()-began})
        if step<3 or (step+1)%100==0:emit({'phase':'train','arm':args.arm,**logs[-1],'elapsed':time.monotonic()-started})
        if time.time()-p['created_unix']>p['hard_cap_seconds']:
            save(out/'budget-limited.json',{'completed_updates':step+1,'intended_updates':3000});raise RuntimeError('Whole-round hard time cap reached')
    atomic_checkpoint(out/'checkpoint.pt',model,opt,3000,{'arm':args.arm,**cfg},sha(HERE/'train600.json'))
    save(out/'training.json',{'arm':args.arm,'completed_updates':3000,'checkpoint_sha256':sha(out/'checkpoint.pt'),
                            'seconds':time.monotonic()-started,'steps':logs})
    emit({'phase':'trained','arm':args.arm,'seconds':time.monotonic()-started})
if __name__=='__main__':main()
