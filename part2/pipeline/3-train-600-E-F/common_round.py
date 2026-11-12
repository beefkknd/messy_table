"""Isolated, reproducible Step 0/1 experiment; never writes historical artifacts."""
from pathlib import Path
import hashlib, json, os, sys, time
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROGRAM = ROOT/'research/self-learning-program'
ENG = PROGRAM/'engineering'
sys.path.insert(0, str(ENG))
from common import initialize_runtime, pin_ultralytics
initialize_runtime()
A = ENG/'round2/train/A/checkpoint.pt'
DATA = ROOT/'data/messytable-full/MessyTable'
def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); os.replace(tmp,path)
def emit(value): print(json.dumps(value,allow_nan=False),flush=True)
def verify_protocol():
    frozen=read(HERE/'protocol-freeze.json')
    for name,digest in frozen.items(): assert sha(HERE/name)==digest,name
    p=read(HERE/'protocol.json')
    codes=dict(p['code_sha256'])
    if (HERE/'protocol-amendments.json').exists():
        for amendment in read(HERE/'protocol-amendments.json')['amendments']:
            codes.update(amendment['code_sha256'])
    for name,digest in codes.items(): assert sha(HERE/name)==digest,name
    assert sha(A)==p['starting_checkpoint_sha256']
    if (HERE/'source-cache-freeze.json').exists():
        for name,digest in read(HERE/'source-cache-freeze.json')['files'].items():assert sha(ROOT/name)==digest,name
    return p
