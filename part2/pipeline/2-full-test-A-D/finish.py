"""Finish this running evaluation after the complete prediction freeze exists."""
from pathlib import Path
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

started = time.monotonic()
while not (HERE/'prediction-freeze.json').exists():
    if time.monotonic()-started > 3600:
        raise TimeoutError('Prediction freeze did not arrive within one hour')
    time.sleep(10)

steps = [
    [sys.executable, '-u', str(HERE/'run.py'), 'score'],
    [sys.executable, '-u', str(HERE/'verify_results.py')],
    [sys.executable, '-u', str(ROOT/'evaluation/messytable-part-two/build.py')],
]
for command in steps:
    print('Starting', ' '.join(command[2:]), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
assert json.loads((HERE/'independent-verification.json').read_text())['passed']
(HERE/'completion.json').write_text(json.dumps({
    'status': 'complete', 'independent_verification_passed': True,
    'training': False, 'photos': 16749, 'completed_unix': time.time(),
}, indent=2)+'\n')
print('Scoring, independent verification and Part Two rebuild complete.', flush=True)
