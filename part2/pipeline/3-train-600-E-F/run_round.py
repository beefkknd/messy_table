"""Sequential MPS jobs; no test labels before all test choices/predictions freeze."""
from common_round import *
import subprocess

def run(script,*args):
    p=verify_protocol()
    assert time.time()-p['created_unix']<p['hard_cap_seconds'],'Round budget expired'
    emit({'phase':'dispatch','script':script,'args':args})
    subprocess.run([sys.executable,str(HERE/script),*args],check=True,cwd=ROOT)

def main():
    if (HERE/'development-comparison.json').exists():
        dev=read(HERE/'development-comparison.json')
        if not any(dev[a]['decision']['meets_preregistered_target'] for a in ('smart','frames600','sliced600')):
            emit({'phase':'stopped','reason':'Primary development gate failed; resuming must not bypass the stop. The separately frozen secondary control follow-up has its own runner.'})
            return
    if not (HERE/'stage1-development.json').exists():run('evaluate.py','stage1')
    if not (HERE/'existing-calibrated-reference-development.json').exists():run('evaluate.py','reference')
    for arm in ('frames600','sliced600'):
        if not (HERE/f'training/{arm}/training.json').exists():run('train.py',arm)
        if not (HERE/f'predictions/{arm}/calibration30/freeze.json').exists():run('evaluate.py','infer','--arm',arm,'--cohort','calibration30')
        if not (HERE/f'{arm}-selection.json').exists():run('evaluate.py','calibrate','--arm',arm)
        if not (HERE/f'predictions/{arm}/development100/freeze.json').exists():run('evaluate.py','infer','--arm',arm,'--cohort','development100')
    if not (HERE/'development-comparison.json').exists():run('evaluate.py','development')
    for arm in ('frames600','sliced600'):
        if not (HERE/f'predictions/{arm}/full-test/freeze.json').exists():run('evaluate.py','infer','--arm',arm,'--cohort','full-test')
    if not (HERE/'full-test-comparison.json').exists():run('evaluate.py','full-test')
    save(HERE/'completion.json',{'completed_unix':time.time(),'elapsed_seconds':time.time()-read(HERE/'protocol.json')['created_unix'],
        'full_test_sha256':sha(HERE/'full-test-comparison.json'),'no_teacher_student':True})
if __name__=='__main__':main()
