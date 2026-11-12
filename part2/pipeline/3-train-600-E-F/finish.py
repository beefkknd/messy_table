"""Seal secondary-control scoring, then audit and build the isolated report."""
from common_round import *
import subprocess
def main():
    verify_protocol();done=read(HERE/'control-followup-completion.json')
    done['full_test_controls_sha256']=sha(HERE/'full-test-matched-controls.json')
    done['controls_scoring_sealed_unix']=time.time()
    save(HERE/'control-followup-completion.json',done)
    save(HERE/'final-results-freeze.json',{n:sha(HERE/n) for n in [
        'development-comparison.json','development-matched-controls.json','control-followup-freeze.json',
        'full-test-comparison.json','full-test-matched-controls.json','control-followup-completion.json']})
    for script in ('audit_results.py','build_report.py'):
        subprocess.run([sys.executable,str(HERE/script)],check=True,cwd=ROOT)
if __name__=='__main__':main()
