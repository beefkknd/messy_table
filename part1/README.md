# Part One · Matching the same object across photos

A study of cross-view object re-identification on a fixed 100-scene sample of the
[MessyTable](https://github.com/caizhongang/MessyTable) test split: 900 photos, 3,600 camera-view
pairs, 78,335 true object links. It reproduces two published matchers (ASNet, ViT-P3DE Tiny) and
compares 45 configurations. The best, a 75/25 blend of RoMa + SAM foreground correspondence with
ViT-P3DE appearance, reaches 90.63% AP against 81.29% for ViT-P3DE alone.

**Read the report:** open [`site/index.html`](site/index.html). It is a static page with its data
embedded, so it needs no server. With GitHub Pages enabled it is served at
`https://beefkknd.github.io/messy_table/part1/site/`.

## Layout

| Folder | What it holds |
|---|---|
| `site/` | The built report: `index.html`, its images and `provenance.json`. Generated; do not edit by hand. |
| `report/` | Source of the report build: `build.py`, `flow_story.py` (the illustrated five-step walkthrough), `scatter_charts.py`, `template.html`, `paths.py` (every input location), `external-inputs.json` (sha256 of the large inputs). |
| `results/` | Frozen results the page links to and the small inputs the build reads. `sample100/` mirrors the original run folder; `part-one/` holds the FPR95 tie audit and the replay of the authors' pairing solver. |
| `tools/` | Two Docker replay tools archived verbatim: `audit_fpr95.py` and `flow_native_links.py`. |
| `pipeline/` | The 29 scripts that produced the frozen results, archived byte-for-byte and grouped by stage. See below. |

## Rebuild the report

The build reads small inputs from `results/` and five kinds of large inputs from outside git: the
MessyTable `test.json` labels (224 MB), six scene photos, three 20 MB score arrays and the
ground-truth vector. `report/external-inputs.json` lists each one with its sha256, and the build
stops if any is missing or different.

```bash
pip install -r part1/report/requirements.txt
MESSYTABLE_WORKSPACE=/path/to/workspace python part1/report/build.py
```

`MESSYTABLE_WORKSPACE` is the folder holding `data/messytable-full/MessyTable/` and
`evaluation/messytable-official-reproduction/sample100/`. When this repo sits inside that workspace
the variable can be omitted.

Every number in the walkthrough is computed from the frozen results, and the build asserts the
claims it states. Examples: the blend equals 75% pixel rank plus 25% appearance rank, the fingerprints
reproduce the stored appearance distances, the solver replay reproduces the official 74,159 correct and
6,147 wrong links, and the wrong mug pair clears the assignment cutoff.

## How the frozen results were produced

`pipeline/` is an archive, not a runnable package. The scripts are kept byte-for-byte so they still
match the sha256 values recorded inside the frozen results. `pipeline/MANIFEST.json` maps every file
to its original workspace path and hash. They import each other through the original workspace
layout, so they run from that workspace, not from this repo.

| Stage | Scripts |
|---|---|
| `0-shared` | `common.py`: paths, hashing, loaders for upstream definitions |
| `1-sample` | `sample.py` picks 100 scenes from 100 distinct camera setups by a fixed hash rule; `variant_common.py` defines the Step 2 plan |
| `2-encode` | `encode.py` (ViT-P3DE and ASNet features), `variant_views.py` (SAM masks, DINOv2 crops), `variant_dense.py` (RoMa v2 correspondence), `variant_perspective.py` |
| `3-score` | `distances.py` (reference distances), `variant_scores.py` (all 45 score arrays and blends) |
| `4-evaluate` | `variant_evaluate.py`, `finish_sample.py` and `runtime/`: the authors' metric code and native solver in a pinned Python 3.7 Docker image |
| `5-sweep-report` | `build_variants_report.py` writes `results/sample100/step2/report.html` |
| `borrowed/` | Helper modules the pipeline imports from earlier experiments, mirrored at their original paths |

The metric runtime is `pipeline/4-evaluate/runtime/Dockerfile`
(`python:3.7-slim-bullseye`, linux/amd64, numpy 1.21.6, scipy 1.7.3, scikit-learn 1.0.2), tagged
`messytable-published-metrics:20261111`. `tools/flow_native_links.py` was run in that image with the
workspace mounted at `/proj`:

```bash
docker run --rm --platform linux/amd64 -v "$WORKSPACE":/proj messytable-published-metrics:20261111 \
  python /proj/evaluation/messytable-part-one/flow_native_links.py
```

## Third-party material

Not included in this repo; obtain from the publishers.

| Material | Source | License |
|---|---|---|
| MessyTable images and labels | github.com/caizhongang/MessyTable | Code MIT; no redistribution statement found for the dataset |
| ASNet code and weights | github.com/caizhongang/MessyTable | Code MIT; weights unlicensed (Google Drive link in the upstream README) |
| ViT-P3DE code, weights, KMSolver binary | github.com/La-tale/ViT-P3DE (commit 14f75ec7) | Code MIT; weights unlicensed |
| RoMa v2 code and checkpoint | github.com/Parskatt/RoMaV2 (v2.0.1) | MIT |
| DINOv3 code (inside RoMa v2) | facebookresearch/dinov3 | DINOv3 License |
| DINOv2-small, SAM 2.1-S | Hugging Face facebook/dinov2-small; Ultralytics SAM 2.1 | Per publisher |

`site/assets/` contains six MessyTable photos and images derived from them, used to illustrate the
task. Their redistribution terms were not confirmed with the publisher.

## Limits

- The 100-scene sample is not the full official test (1,861 scenes); published full-test figures are context only.
- The winning configuration was selected after comparing all 45 on this same sample, so its numbers may be optimistic for new photos.
- FPR95 is reported as not applicable for four RoMa-only scorers whose 95%-recall cutoff falls inside a block of tied scores; see `results/part-one/fpr95-tie-audit.json`.
- Human boxes are supplied to every method; detection is out of scope for Part One.

## Credits

Research, experiments, pipeline and report: **GPT Astra** and **Sol**.
Review and number cross-checks, the illustrated five-step method walkthrough, and repository packaging: Claude (Anthropic).
