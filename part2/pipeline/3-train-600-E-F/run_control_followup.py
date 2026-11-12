"""Validate a qualifying prespecified fixed control; preserve primary failure."""
from common_round import *
import subprocess
def main():
    p=verify_protocol();followup=read(HERE/'control-followup-freeze.json')
    assert followup['primary_completion_sha256']==sha(HERE/'completion.json')
    assert followup['controls_sha256']==sha(HERE/'development-matched-controls.json')
    for arm in ('frames600','sliced600'):
        assert time.time()-p['created_unix']<p['hard_cap_seconds']
        if not (HERE/f'predictions/{arm}/full-test/freeze.json').exists():
            subprocess.run([sys.executable,str(HERE/'evaluate.py'),'infer','--arm',arm,'--cohort','full-test'],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(HERE/'evaluate.py'),'full-test'],check=True,cwd=ROOT)
    save(HERE/'control-followup-completion.json',{'completed_unix':time.time(),
        'total_round_elapsed_seconds':time.time()-p['created_unix'],'followup_freeze_sha256':sha(HERE/'control-followup-freeze.json'),
        'full_test_sha256':sha(HERE/'full-test-comparison.json'),
        'full_test_controls_sha256':sha(HERE/'full-test-matched-controls.json'),
        'primary_failure_preserved':sha(HERE/'completion.json')==followup['primary_completion_sha256']})
if __name__=='__main__':main()
