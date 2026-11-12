"""Pure paths and atomic artifacts; model runtimes live in separate processes."""
from pathlib import Path
import hashlib,json,os,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'research/detector-small-object-round'
PROGRAM=ROOT/'research/self-learning-program'
ENG=PROGRAM/'engineering'
DATA=ROOT/'data/messytable-full/MessyTable'
REPRO=ROOT/'evaluation/messytable-official-reproduction'
SAMPLE=REPRO/'sample100'
A=ENG/'round2/train/A/checkpoint.pt'
W600=OLD/'training/sliced600/checkpoint.pt'
W2400=HERE/'training/sliced2400/checkpoint.pt'
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def emit(v):print(json.dumps(v,allow_nan=False),flush=True)
def runtime():
    os.environ['YOLO_CONFIG_DIR']=str(HERE/'.runtime');os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false'
    os.environ['MPLCONFIGDIR']=str(HERE/'.runtime/matplotlib');os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
def verify():
    for name,digest in read(HERE/'protocol-freeze.json').items():assert sha(HERE/name)==digest,name
    p=read(HERE/'protocol.json')
    for name,digest in p['code_sha256'].items():assert sha(HERE/name)==digest,name
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert time.time()-p['created_unix']<p['hard_cap_seconds'],'Six-hour compute cap exceeded'
    return p
