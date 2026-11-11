# MessyTable · cross-view object re-identification

Studies of how to tell that an object in one photo is the same physical object in another, using the
[MessyTable](https://github.com/caizhongang/MessyTable) multi-camera dataset.

| Part | Topic | Status |
|---|---|---|
| [Part One](part1/) | Matching the same object across photos, given human object boxes | Published: [report](part1/site/index.html) |
| [Part Two](part2/) | Image processing and object detection | Built; under review |
| [Part Three](part3/) | Next stage | Planned |

Each part keeps its static report in `partN/site/`, its build source in `partN/report/`, and its frozen
results in `partN/results/`. Large datasets, model weights and score arrays stay outside git and are
listed with sha256 hashes so builds can verify them.
