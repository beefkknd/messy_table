"""Part Two pipeline archive: which workspace files are archived, by stage.

Rule: include every first-party script/config whose output Part Two's page uses; leave out scripts whose only
job is identity matching (Step 2 / Part Three, deferred). Mixed scripts that Part Two depends on are included
and marked in NOTES. Files are copied byte-for-byte by archive_pipeline.py, which writes pipeline/MANIFEST.json.
"""

SL = 'research/self-learning-program'
ENG = f'{SL}/engineering'
SMALL = 'research/detector-small-object-round'
SCALE = 'research/detector-scale-and-integration'
TEACH = f'{SL}/stronger-teacher'
FULL = 'evaluation/messytable-detection-full-test'

STAGES = {
    '0-data': [
        'data/messytable-full/scripts/acquire.py', 'data/messytable-full/scripts/audit_dataset.py',
        f'{SL}/data/prepare.py', f'{SL}/data/reveal_development.py', f'{SL}/data/export_original100.py',
        f'{SL}/data/yolo/supervised-frames.json',
    ],
    '1-train-150': [
        f'{ENG}/train_detector.py', f'{ENG}/partial_loss.py', f'{ENG}/common.py', f'{ENG}/run_round2.sh',
        f'{ENG}/requirements.lock.txt', f'{ENG}/verify_round2_training.py', f'{ENG}/check_loss.py',
    ],
    '2-full-test-A-D': [
        f'{FULL}/run.py', f'{FULL}/verify_results.py', f'{FULL}/finish.py',
        'evaluation/run_full_scene.py',
        f'{ENG}/predict_detector.py', f'{ENG}/score_detector.py', f'{ENG}/build_teacher.py',
        f'{ENG}/score_round2.py', f'{SL}/data/summarize_round2.py',
        f'{SL}/step0-tiles/engineering/tile_detector.py', f'{SL}/step0-tiles/engineering/common.py',
        f'{SL}/step0-tiles/data/score_tiles.py',
    ],
    '3-train-600-E-F': [
        f'{SMALL}/{name}' for name in (
            'prepare.py', 'common_round.py', 'train.py', 'tile_policy.py', 'evaluate.py', 'run_round.py',
            'run_control_followup.py', 'finish.py', 'audit_results.py',
            'protocol.json', 'protocol-amendments.json', 'train600.json')
    ],
    '4-train-2400-G-H': [
        f'{SCALE}/{name}' for name in (
            'common_new.py', 'prepare.py', 'train2400.py', 'detect.py', 'run.py', 'score_step1.py',
            'step2_inputs.py', 'finish.py', 'protocol.json', 'train2400.json')
    ],
    '5-self-labelling': [
        f'{TEACH}/PROTOCOL.json', f'{TEACH}/AMENDMENT_01.json',
        f'{TEACH}/data/freeze_cohorts.py', f'{TEACH}/data/audit_teacher.py', f'{TEACH}/data/score_study.py',
        f'{SL}/data/audit_pseudo180_diagnostics.py',
        f'{TEACH}/engineering/build_stronger_teacher.py', f'{TEACH}/engineering/continue_training.py',
        f'{TEACH}/engineering/run_followup_arm.sh', f'{TEACH}/engineering/run_prediction.py',
        f'{TEACH}/engineering/common.py', f'{TEACH}/engineering/run_scoring.py',
        f'{TEACH}/engineering/score_followup.py', f'{TEACH}/engineering/qualify_refresh.py',
    ],
    '6-report-audit': [
        'evaluation/messytable-part-two/audit_metrics.py', 'evaluation/messytable-part-two/verify_report.mjs',
    ],
}

NOTES = {
    f'{SCALE}/finish.py': 'Mixed: writes audit.json, which Part Two build.py requires to say "passed". It also checks '
                          'Step 2 (Part Three) artefacts, which are not published.',
    f'{SCALE}/step2_inputs.py': 'Mixed by name only: exports the merged detector boxes on the 900 Part One photos '
                                '(step2/boxes.json). Part Two uses them for the Section 2 example. No matching.',
    f'{SCALE}/score_step1.py': 'Normally launched by post_detect.py (Part Three launcher, not archived); runs alone.',
    f'{SL}/step0-tiles/engineering/common.py': 'This is the "common" module full-test run.py imports (step0-tiles is '
                                               'first on sys.path), not engineering/common.py.',
}

EXCLUDED_PART_THREE = [
    f'{SCALE}/{name}' for name in (
        'step2_views.py', 'step2_dense.py', 'evaluation_maps.py', 'score_step2_native.py',
        'fullframe_control_inputs.py', 'fullframe_control_native.py', 'fullframe_control_followup.py',
        'wait_and_integrate.py', 'post_detect.py', 'build_report.py', 'detect_development.py')
]
