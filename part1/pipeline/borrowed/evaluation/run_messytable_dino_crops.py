"""Frozen crop/context ensembles and first-match-wins DINO ladder, no training."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch,transformers
from transformers import AutoImageProcessor,AutoModel
from scipy.optimize import linear_sum_assignment
from run_full_scene import encode_boxes
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'evaluation';SOURCE=HERE/'messytable-ladder-dino';OUT=HERE/'messytable-dino-crop-ladder';BLIND={41,60,67}
sys.path.insert(0,str(ROOT/'everyday-identity/scripts'))
from runtime_assets import dinov2_source,dinov2_weights,dinov2_commit
SCALES=[1.,1.25,1.5,2.];BRANCHES=['tight_1x','context_125','context_150','context_200']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def assign(similarity,available_a=None,available_b=None):
 a=np.array(list(range(similarity.shape[0])) if available_a is None else available_a,dtype=int)
 b=np.array(list(range(similarity.shape[1])) if available_b is None else available_b,dtype=int)
 sub=similarity[np.ix_(a,b)];r,c=linear_sum_assignment(np.maximum(sub-.70,0),maximize=True)
 return [{'A':int(a[i]),'B':int(b[j]),'cosine':float(sub[i,j])} for i,j in zip(r,c) if sub[i,j]>.70]
def expanded(boxes,factor,width,height):
 if factor==1.:return boxes.copy()
 centers=(boxes[:,:2]+boxes[:,2:])/2;half=(boxes[:,2:]-boxes[:,:2])*factor/2
 values=np.column_stack([centers-half,centers+half]).round().astype(int);values[:,[0,2]]=np.clip(values[:,[0,2]],0,width);values[:,[1,3]]=np.clip(values[:,[1,3]],0,height)
 return values

def main():
 OUT.mkdir(exist_ok=False);(OUT/'features').mkdir();(OUT/'predictions').mkdir()
 sf=json.loads((SOURCE/'inference-freeze.json').read_text())
 for p,h in sf.items():assert sha(SOURCE/p)==h
 historical={str(p.relative_to(ROOT)):sha(p) for p in SOURCE.rglob('*') if p.is_file()}
 source_predictions=json.loads((SOURCE/'predictions/stage-09.json').read_text())['predictions'];assert len(source_predictions)==97 and not BLIND&{r['case'] for r in source_predictions}
 methods=BRANCHES+['mean_scores','max_scores','two_crop_support','ladder_through_125','ladder_through_150','ladder_through_200']
 weights=dinov2_weights();plan={'training':False,'scored_pairs':97,'blind_excluded_from_inference_and_scoring':sorted(BLIND),
  'detector_boxes':'Exact final stage9 progressive YOLO ladder boxes; never change or score expanded boxes as detections.',
  'crop_scales':SCALES,'crop_expansion':'Center-preserving width and height multiplier; round to integers then clip to original image boundaries. Original RGB pixels only, no masks or color edits.',
  'embedder':'facebook/dinov2-small','commit':dinov2_commit(),'weights_sha256':sha(weights),'preprocessing':'Final CLS, L2 normalized; resize224x224, no center crop, same processor as baseline.',
  'tight_control':'Reuse checksum-frozen tight features from previous97-pair run; verify a fresh tight encoding on first pair. Verify all tight assignments against saved final-stage links.',
  'matching_cutoff':.70,'assignment':'Same maximum-weight one-to-one Hungarian assignment on max(score-0.70,0), retain score>0.70.',
  'combination_rules':{'mean_scores':'Arithmetic mean of four same-scale cosine matrices; equivalent to cosine of equally weighted concatenated unit embeddings.',
                       'max_scores':'Maximum of four same-scale cosine scores; optimistic multi-attempt rule. Same numeric cutoff is NOT the same false-positive operating point.',
                       'two_crop_support':'Second-highest score among four same-scale cosines; >0.70 requires at least two crop sizes to exceed cutoff.',
                       'ladder':'Start with tight assignment, lock endpoints of retained links. At1.25,1.5,2.0, match only remaining unmatched endpoints. Never replace earlier links; this also locks earlier mistakes.'},
  'methods':methods,'score_calibration':'Cutoff remains0.70 for every rule; means/order statistics alter score distributions and correlations. No claim of equivalent false-positive rates.',
  'scoring':'Freeze all features, predictions and rules before label access; original detector boxes define annotation mappings. Report correct/wrong/unmapped and true-link recall for all methods.',
  'development_limits':['Repeatedly inspected97 development pairs, no fresh holdout.','Wider crops can encode neighboring lookalikes or scene background rather than target identity.',
                        'Same detector boxes impose identical detection ceiling593/1647 on every variant.'],
  'historical_hashes_before':historical,'source_freeze_sha256':sha(SOURCE/'inference-freeze.json'),'torch':torch.__version__,'transformers':transformers.__version__,'device':'mps','script_sha256':sha(Path(__file__))}
 save(OUT/'plan.json',plan)
 processor=AutoImageProcessor.from_pretrained(dinov2_source(),local_files_only=True);encoder=AutoModel.from_pretrained(dinov2_source(),local_files_only=True).to('mps').eval().requires_grad_(False)
 all_predictions={m:[] for m in methods};controls=[];start=time.monotonic();fresh_control=[]
 for n,pair in enumerate(source_predictions,1):
  case=pair['case'];features={};crop_metadata={}
  for side in 'AB':
   cache=np.load(SOURCE/'features'/f'case-{case:03}-{side}.npz');boxes=cache['boxes'];tight=cache['features'];assert boxes.tolist()==[d['box_xyxy'] for d in pair['detections'][side]]
   path=ROOT/'data/messytable-100'/f'case-{case:03}'/f'{side}.jpg'
   with Image.open(path) as image:width,height=image.size
   crops={name:expanded(boxes,factor,width,height) for name,factor in zip(BRANCHES,SCALES)}
   if n==1:
    fresh=encode_boxes(path,boxes,processor,encoder,'mps');delta=float(np.max(np.abs(fresh-tight))) if tight.size else 0.;fresh_control.append({'case':case,'side':side,'max_feature_delta':delta});assert delta<1e-4
   encoded={'tight_1x':tight}
   for name in BRANCHES[1:]:encoded[name]=encode_boxes(path,crops[name],processor,encoder,'mps')
   features[side]=encoded;crop_metadata[side]={k:v.tolist() for k,v in crops.items()}
   np.savez_compressed(OUT/'features'/f'case-{case:03}-{side}.npz',**encoded,**{name+'_boxes':value for name,value in crops.items()})
  sims={name:features['A'][name]@features['B'][name].T for name in BRANCHES};stack=np.stack(list(sims.values()),axis=0)
  matrices={**sims,'mean_scores':stack.mean(0),'max_scores':stack.max(0),'two_crop_support':np.sort(stack,axis=0)[-2]}
  links={name:assign(values) for name,values in matrices.items()}
  expected={(l['A'],l['B']):l for l in pair['links']};actual={(l['A'],l['B']):l for l in links['tight_1x']}
  same=expected.keys()==actual.keys();delta=max([abs(expected[k]['cosine']-actual[k]['cosine']) for k in expected.keys()&actual.keys()]+[0.]);assert same and delta<1e-4
  controls.append({'case':case,'tight_assignment_reproduced':same,'max_cosine_delta':delta})
  retained=[{**link,'accepted_crop':'tight_1x'} for link in links['tight_1x']]
  for name,method in zip(BRANCHES[1:],['ladder_through_125','ladder_through_150','ladder_through_200']):
   used_a={l['A'] for l in retained};used_b={l['B'] for l in retained};available_a=[i for i in range(sims[name].shape[0]) if i not in used_a];available_b=[j for j in range(sims[name].shape[1]) if j not in used_b]
   retained.extend({**link,'accepted_crop':name} for link in assign(sims[name],available_a,available_b));links[method]=[dict(l) for l in retained]
  for method in methods:
   annotated=[]
   for link in links[method]:
    a,b=link['A'],link['B'];scores={name:float(sim[a,b]) for name,sim in sims.items()};annotated.append({**link,'crop_cosines':scores,'supporting_crops_over_cutoff':sum(v>.70 for v in scores.values())})
   all_predictions[method].append({'case':case,'detections':pair['detections'],'links':annotated})
  save(OUT/'features'/f'case-{case:03}-crops.json',{'case':case,'crop_boxes':crop_metadata})
  if n%10==0 or n==97:print(f'{n}/97 context features encoded; tight control exact={same}; links tight{len(links["tight_1x"])} mean{len(links["mean_scores"])} max{len(links["max_scores"])} support2{len(links["two_crop_support"])} ladder{len(links["ladder_through_200"])}',flush=True)
 for method,pairs in all_predictions.items():save(OUT/'predictions'/f'{method}.json',{'method':method,'predictions':pairs})
 save(OUT/'regression-check.json',{'tight_assignments':controls,'fresh_feature_control':fresh_control,'passed':True})
 assert all(sha(ROOT/p)==h for p,h in historical.items()),'Historical DINO run changed'
 save(OUT/'baseline-integrity.json',{'unchanged':True,'checked_files':len(historical),'hashes':historical})
 save(OUT/'inference.json',{'pairs':97,'images':194,'methods':methods,'crop_scales':SCALES,'elapsed_seconds':time.monotonic()-start})
 files=list((OUT/'features').glob('*'))+list((OUT/'predictions').glob('*.json'))+[OUT/p for p in ['plan.json','regression-check.json','baseline-integrity.json','inference.json']]
 save(OUT/'inference-freeze.json',{str(p.relative_to(OUT)):sha(p) for p in files});print('Frozen all crop features and ten prediction methods before scoring.',flush=True)

if __name__=='__main__':main()
