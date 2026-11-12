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
| `pipeline/` | The 63 scripts and configs that produced the numbers, archived byte-for-byte by stage. See below. |
| `pipeline_files.py`, `archive_pipeline.py` | The archive's file list (with the Part Three exclusions) and the script that copies it. |

Detailed study folders (2,400-photo training, 600-photo tile training, the self-labelling study) stay in
the project archive. The 2,400-photo study also contains the detector-to-matcher test, so its results will be
published with Part Three.

## How the numbers were produced

`pipeline/` is an archive, not a runnable package. Scripts are copied byte-for-byte so they still match the
sha256 values recorded in the original protocol and freeze files. `pipeline/MANIFEST.json` maps every file to its
original workspace path and hash. They import each other through the original workspace layout and run there.

| Stage | What it covers | Table rows |
|---|---|---|
| `0-data` | Dataset download/audit, training and evaluation cohorts, the 150 trusted training photos | — |
| `1-train-150` | The first fine-tune: 150 labelled photos, 300 updates at 768 px, partial-label loss | B, C, D (model) |
| `2-full-test-A-D` | Full-test inference, tiling, the strict tile gate, scoring and an independent re-count | A–D |
| `3-train-600-E-F` | 600 labelled photos with full frames and tile crops, 3,000 updates at 1024 px; its protocol and amendments | E, F |
| `4-train-2400-G-H` | 2,400 labelled photos, the 768/1024 inference factorial and Step 1 scoring | G, H |
| `5-self-labelling` | Teacher label generation, label-quality audit, student continuations T/S/L/O and their scoring | Self-labelling table |
| `6-report-audit` | The metric audit behind `metric-audit.json`, and the page's browser check | — |

**Left out on purpose.** Scripts whose only job is identity matching (Step 2: SAM/RoMa/ViT/DINO on detector boxes,
the native matching solver, full-frame matching controls and their launchers) belong to Part Three and are listed
under `excluded_part_three` in the manifest. Two mixed files are included because Part Two depends on them, and are
marked in the manifest: the 2,400-photo study's `finish.py` writes the `audit.json` that the page builder requires
(it also checks Step 2 files, which are not published), and `step2_inputs.py` only exports the merged detector boxes
used for the Section 2 example.

**Not included:** model weights (the trained checkpoints are 187–262 MB each), MessyTable images and labels,
prediction folders and the Python environment. `1-train-150/requirements.lock.txt` pins the packages (Ultralytics
8.4.202, which is AGPL-3.0; PyTorch 2.15.0).

**Reproduction caveats.** The scripts assume Apple-silicon MPS (`device='mps'` is hard-coded), the original workspace
paths, and a dedicated virtual environment. Some protocols assert a six-hour window from their creation time, so a
re-run needs that guard relaxed explicitly. A few small audit files have no generator script; the manifest lists
only code that exists.

## Review

Every number on the page was recomputed independently from raw labels and saved predictions:
full-test detection rows, small-object and stricter-overlap tables, and the 2,400-photo training
manifest's separation from the test set (no shared setups, scenes or images).

## Credits

Research, experiments and report: **GPT Astra** and **Sol**.
Review and number cross-checks, and repository packaging: Claude (Anthropic).
