"""Package the reviewed Part Two page from the research workspace into this repo.

The page itself is built in the workspace (evaluation/messytable-part-two/build.py). This script only
copies it and its evidence here and rewrites links for the repo layout. It changes no numbers.

Links to detailed study folders are turned into plain text: those folders hold Part Three material
(the detector-to-matcher test) or have not been reviewed as published pages. Every rewrite is listed
in PACKAGING.json together with the sha256 of the source and the output.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

PART2 = Path(__file__).resolve().parent
SITE, REPORT, RESULTS = PART2 / 'site', PART2 / 'report', PART2 / 'results'
SOURCE_REL = 'evaluation/messytable-part-two'
FULL_TEST_REL = 'evaluation/messytable-detection-full-test/results.json'
CREDIT = ('Research, experiments and report: GPT Astra and Sol. Review and number cross-checks, '
          'and repository packaging: Claude (Anthropic).')

LINK_REWRITES = [  # (old, new, expected count)
    ('href="../messytable-part-one/report.html"', 'href="../../part1/site/"', 2),
    ('href="../messytable-detection-full-test/results.json"', 'href="../results/detection-full-test/results.json"', 1),
    ('href="build.py"', 'href="../report/build.py"', 1),
]
UNLINKED = [  # (old list item / fragment, replacement)
    ('<li><a href="../../research/detector-scale-and-integration/report.html">Detector evidence: 2,400 training photos '
     'and input resolution</a> · <a href="../../research/detector-scale-and-integration/protocol.json">frozen protocol</a> · '
     '<a href="../../research/detector-scale-and-integration/audit.json">completed audit</a> · '
     '<a href="../../research/detector-scale-and-integration/README.txt">reproduction details</a></li>',
     '<li>Detector evidence: 2,400 training photos and input resolution. The study files also contain the '
     'detector-to-matcher test, so they will be published with Part Three.</li>'),
    ('<li><a href="../../research/detector-small-object-round/report.html">600-photo training and tile-filtering study</a></li>',
     '<li>600-photo training and tile-filtering study (project archive).</li>'),
    ('<li><a href="../../research/self-learning-program/stronger-teacher/report.html">Automatic-label training: control arms, '
     'label quality and stop decision</a></li>',
     '<li>Automatic-label training: control arms, label quality and stop decision (project archive).</li>'),
    (' · <a href="archive/2026-11-11/index.html">Archived previous Part Two</a>', ''),
    ('<p><a href="../../research/self-learning-program/stronger-teacher/report.html">Read the complete automatic-label '
     'experiment and its controls</a>.</p>\n',
     '<p>The complete automatic-label experiment and its controls are kept in the project archive.</p>\n'),
]


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def workspace() -> Path:
    configured = os.environ.get('MESSYTABLE_WORKSPACE')
    candidate = Path(configured).expanduser() if configured else PART2.parents[1]
    if not (candidate / SOURCE_REL / 'report.html').is_file():
        raise SystemExit(f'Part Two source not found under {candidate}. Set MESSYTABLE_WORKSPACE.')
    return candidate


def replace_exact(page: str, old: str, new: str, count: int) -> str:
    found = page.count(old)
    if found != count:
        raise SystemExit(f'Expected {count} occurrence(s), found {found}: {old[:80]}')
    return page.replace(old, new)


def transform(page: str) -> str:
    for old, new, count in LINK_REWRITES:
        page = replace_exact(page, old, new, count)
    for old, new in UNLINKED:
        page = replace_exact(page, old, new, 1)
    return replace_exact(page, '</footer>', f' {CREDIT}</footer>', 1)


def copy(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    assert sha256(source) == sha256(destination)
    return {'from': str(source), 'to': str(destination.relative_to(PART2.parent)), 'sha256': sha256(destination)}


def check_links(page_path: Path) -> list[str]:
    links = set(re.findall(r'(?:href|src)="([^"#]+)"', page_path.read_text()))
    local = [link for link in links if not link.startswith(('http://', 'https://', 'mailto:'))]
    return sorted(link for link in local if not (page_path.parent / link).resolve().exists())


def main() -> None:
    ws = workspace()
    source = ws / SOURCE_REL
    original = (source / 'report.html').read_text()
    used = sorted(set(re.findall(r'(?:href|src)="assets/([^"#]+)"', original))  # assets the page references,
                  | set(re.findall(r'content="https://[^"]*/part2/site/assets/([^"#]+)"', original)))  # incl. og:image
    if SITE.exists():
        shutil.rmtree(SITE)
    copies = [copy(source / 'assets' / name, SITE / 'assets' / name) for name in used]
    copies += [copy(source / name, SITE / name) for name in ('evidence.json', 'metric-audit.json')]
    copies += [copy(source / name, REPORT / name) for name in ('build.py', 'template.html', 'pipeline_example.py')]
    copies.append(copy(ws / FULL_TEST_REL, RESULTS / 'detection-full-test/results.json'))
    page = transform((source / 'report.html').read_text())
    (SITE / 'index.html').write_text(page)
    leftovers = [l for l in re.findall(r'(?:href|src)="([^"]+)"', page) if 'research/' in l or 'archive/' in l]
    broken = check_links(SITE / 'index.html')
    if leftovers or broken:
        raise SystemExit(f'Unpublished links remain: {leftovers}; broken links: {broken}')
    (PART2 / 'PACKAGING.json').write_text(json.dumps({
        'source_page': f'{SOURCE_REL}/report.html', 'source_sha256': sha256(source / 'report.html'),
        'output_sha256': sha256(SITE / 'index.html'),
        'link_rewrites': [{'from': o, 'to': n} for o, n, _ in LINK_REWRITES],
        'unlinked': [o for o, _ in UNLINKED], 'credit_added_to_footer': CREDIT,
        'copied': [{k: v for k, v in c.items() if k != 'from'} for c in copies],
    }, indent=2) + '\n')
    print('Packaged', SITE / 'index.html', '| files copied:', len(copies))


if __name__ == '__main__':
    main()
