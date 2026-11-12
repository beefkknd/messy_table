# Part Two · Teaching the system to find the objects

Before an object can be matched across photos, it has to be found. Part Two measures how well a YOLO26m
detector draws a usable box around each object on the full official MessyTable test (16,749 photos,
408,937 labelled object appearances), and what improved it.

**Read the report:** https://beefkknd.github.io/messy_table/part2/site/

The source is [`site/index.html`](site/index.html); GitHub shows that file as code.

## Main results (full official test, confidence > 0.25, IoU ≥ 0.50)

| Configuration | Recall | Precision | Boxed in both views | Small objects (< 32 px) |
|---|---|---|---|---|
| Released YOLO26m, no local training | 38.22% | 81.51% | 26.83% | — |
| 150 labelled photos, full frame | 84.61% | 85.26% | 76.31% | 20.96% |
| 2,400 labelled photos + tiles, 1024 px | 94.01% | 85.89% | 90.12% | 51.20% |

The gains came from human-labelled training data, a stronger training recipe, overlapping close-up
tiles and higher full-frame resolution. A self-labelling (teacher/student) experiment did not produce
the required gain and was parked; even complete human labels gave no gain under that short recipe.

## Limits

- The official test set has been examined throughout the project, so it is not a fresh holdout.
- The latest training comparison uses one seed and a fixed compute budget.
- Detection only. Feeding these boxes into the identity matcher is Part Three, not yet published.
- Tables are not trucks: road scenes, long time gaps and absent targets are untested.

## Layout

| Path | What it holds |
|---|---|
| `site/` | The published page, its charts, `evidence.json` (source hashes and accounting checks) and `metric-audit.json`. |
| `results/detection-full-test/results.json` | Full-test detector counts and subgroup results the page links to. |
| `report/` | The page builder and template, for reference. They run in the research workspace, not from this repo. |
| `package.py`, `PACKAGING.json` | How the page was copied here: link rewrites, unlinked study folders, file hashes. |

Detailed study folders (2,400-photo training, 600-photo tile training, the self-labelling study) stay in
the project archive. The 2,400-photo study also contains the detector-to-matcher test, so it will be
published with Part Three.

## Review

Every number on the page was recomputed independently from raw labels and saved predictions:
full-test detection rows, small-object and stricter-overlap tables, and the 2,400-photo training
manifest's separation from the test set (no shared setups, scenes or images).

## Credits

Research, experiments and report: **GPT Astra** and **Sol**.
Review and number cross-checks, and repository packaging: Claude (Anthropic).
