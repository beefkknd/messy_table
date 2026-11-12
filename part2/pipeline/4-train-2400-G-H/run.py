"""Sequential MPS experiments; no threshold or test-driven choice."""
from common_new import *
import subprocess,sys
def run(name,*args):
    verify();emit({'phase':'dispatch','script':name,'args':args})
    subprocess.run([sys.executable,str(HERE/name),*args],check=True,cwd=ROOT)
def main():
    run('detect.py','600')
    run('train2400.py','sliced2400')
    run('detect.py','2400')
    save(HERE/'detector-completion.json',{'created_unix':time.time(),'protocol_sha256':sha(HERE/'protocol.json')})
if __name__=='__main__':main()
