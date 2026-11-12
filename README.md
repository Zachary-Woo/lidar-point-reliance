# When does a LiDAR detector stop relying on the object?

> [!WARNING]
> **IN PROGRESS / TODO.** rough. re-check as things change.
>
> - [ ] strip week labeling
> - [ ] review README — consistency, accuracy (steps keep changing)
> - [ ] review other READMEs
> - [ ] review .gitignore
> - [ ] review code readability, adjust names where needed

A stratified study of evidence dependence in 3D detection confidence.

Public KITTI checkpoints are held fixed. The measurement is how much a detector's confidence depends on the LiDAR points inside a predicted box versus the surrounding scan, and whether that dependence is stronger when the object's own evidence is weak (far, sparse, or small). Nothing is trained. The detector weights are not edited.

This is a semester research project (CAP 6908, Fall 2026). The report is being written in CVPR format and is not in this repository yet.

## Contents

- [The measurement](#the-measurement)
- [Results so far](#results-so-far)
- [Repository layout](#repository-layout)
- [Reproducing the numbers](#reproducing-the-numbers)
- [Scripts and what they produce](#scripts-and-what-they-produce)
- [Status](#status)
- [Acknowledgements](#acknowledgements)

## The measurement

Each eligible detection (score at least 0.7, at least one LiDAR point inside the box) is edited on its own, one box per forward pass. The detector is run again on the edited cloud and the new prediction is matched back to the original box.

| Edit | What is deleted | What it tests |
| --- | --- | --- |
| Interior | A fraction of the points inside the predicted box | Does the score need the object's own points? |
| Placebo | The same fraction inside a same-sized box in empty space | Does deleting any same-sized patch move the score? |
| Context | The same fraction of the points outside the box | Does the score need the surrounding scene? |

Fractions are 0, 0.25, 0.5, 0.75, and 1.

**Retained confidence** is the new score divided by the original score when a returned box still overlaps the original at 3D IoU 0.5, and 0 when none does. A miss stays in the mean. Results are broken out by three properties recorded for every box:

- sensor range: 0–20 m, 20–40 m, 40 m and farther
- interior point count: 0–19, 20–99, 100 and above
- box volume: under 8 m³, 8–12 m³, 12 m³ and larger

## Results so far

All numbers are on the KITTI val split (3,769 frames) and come from the summary files in [`DIS Notes and Outputs/outputs by week/`](DIS%20Notes%20and%20Outputs/outputs%20by%20week/).

### Checkpoints reproduce the published AP

| Detector | Checkpoint | Car moderate 3D AP@R11 | Published |
| --- | --- | --- | --- |
| PointPillars (one-stage) | `pointpillar_7728.pth` | 77.2898 | 77.28 |
| PV-RCNN (two-stage) | `pv_rcnn_8369.pth` | 83.6974 | 83.61 |

Compare the R11 numbers only. AP@R40 is a different number and does not match the published column.

### Emptying the box costs more for far and sparse cars

![Mean retained confidence against removal fraction for PointPillars cars, one panel per range bin, for the interior, placebo, and context edits](DIS%20Notes%20and%20Outputs/outputs%20by%20week/week7-outputs/images/retention_by_range.png)

PointPillars, Car, 11,125 eligible boxes. Interior removal lowers retained confidence in every range bin, and the drop arrives sooner for farther cars. The placebo and context edits stay near 1.

Mean retained confidence after all interior points are removed (fraction 1):

| Bin | PointPillars *n* | PointPillars | PV-RCNN *n* | PV-RCNN |
| --- | ---: | ---: | ---: | ---: |
| 0–20 m | 4,864 | 0.193 | 5,496 | 0.228 |
| 20–40 m | 4,868 | 0.149 | 7,110 | 0.103 |
| 40 m and farther | 1,393 | 0.074 | 3,669 | 0.038 |
| 0–19 points | 370 | 0.086 | 2,371 | 0.021 |
| 20–99 points | 3,419 | 0.098 | 5,668 | 0.054 |
| 100 points and above | 7,336 | 0.190 | 8,236 | 0.215 |
| Under 8 m³ | 1,299 | 0.177 | 1,360 | 0.149 |
| 8–12 m³ | 9,082 | 0.154 | 13,460 | 0.127 |
| 12 m³ and larger | 744 | 0.178 | 1,455 | 0.151 |

- Both detectors keep less confidence at longer range and on boxes with fewer interior points.
- Box volume does not order the effect for either detector.
- Each detector has its own eligible set, so the two columns are not paired box by box.

### Neither control reproduces the drop

At fraction 1, PointPillars placebo retention is between 0.999 and 1.000 in every bin, and context retention is between 1.014 and 1.059. The placebo region holds far fewer points than the car it is compared with, so it rules out deletion of an arbitrary same-sized patch. It does not rule out a deletion matched on point count.

### Pedestrian and Cyclist

PointPillars, 672 eligible pedestrians and 394 eligible cyclists. Mean retained confidence under interior removal:

| Bin | Pedestrian 0.75 | Pedestrian 1.00 | Cyclist 0.75 | Cyclist 1.00 |
| --- | ---: | ---: | ---: | ---: |
| 0–20 m | 0.740 | 0.010 | 0.497 | 0.006 |
| 20–40 m | 0.413 | 0.007 | 0.396 | 0.002 |
| 40 m and farther | – | – | 0.252 | 0.000 |

- The range ordering appears at partial removal.
- Once the box is empty almost no confidence is left, unlike Car.
- No eligible pedestrian lies beyond 40 m, and only 15 eligible cyclists do.

### Ablations

PointPillars, Car. Each row changes one setting of the reference protocol. Mean retained confidence at fraction 0.75:

| Setting | 0–20 m | 20–40 m | 40 m+ | 0–19 pts | 20–99 pts | 100+ pts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | 0.929 | 0.770 | 0.618 | 0.519 | 0.667 | 0.908 |
| Farthest-point removal | 0.953 | 0.811 | 0.700 | 0.625 | 0.724 | 0.933 |
| Voxel-wise removal | 0.798 | 0.694 | 0.583 | 0.518 | 0.601 | 0.794 |
| Eligibility floor 0.5 | 0.917 | 0.709 | 0.525 | 0.409 | 0.606 | 0.900 |
| Eligibility floor 0.9 | 0.952 | 0.861 | 0.769 | 0.657 | 0.793 | 0.931 |
| Overlap bar 0.3 | 0.930 | 0.773 | 0.653 | 0.576 | 0.679 | 0.908 |
| Overlap bar 0.7 | 0.922 | 0.703 | 0.442 | 0.336 | 0.545 | 0.891 |
| Nearest-center association | 0.930 | 0.783 | 0.676 | 0.592 | 0.698 | 0.909 |

The settings change how much confidence is retained. They do not change which cars retain less: far below near and sparse below dense in every row. The two floor rows use different eligible sets (14,154 boxes at floor 0.5 and 5,883 at floor 0.9).

### Limits

- One dataset (KITTI). Car on two checkpoints; Pedestrian and Cyclist on PointPillars only.
- The two checkpoints differ in training and operating point as well as architecture, so the Car comparison is not an isolated architecture effect.
- The placebo is not matched on point count.
- Ablations were run one at a time, on PointPillars cars only.

## Repository layout

```text
README.md                    this file
docs/SETUP.md                full setup walkthrough for a new machine
OpenPCDet/                   OpenPCDet with one local fix and the measurement scripts
  tools/week*.py             the measurement scripts (see the table below)
  pcdet/.../rotate_iou.py    CPU replacement for the Numba CUDA IoU used by KITTI eval
DIS Notes and Outputs/
  outputs by week/           summaries, tables, and figures for each week (results of record)
```

Not in git: the KITTI data (`data/`), the checkpoints (`OpenPCDet/checkpoints/`), and the large per-forward working files (`OpenPCDet/output/`).

## Reproducing the numbers

The full walkthrough is in [docs/SETUP.md](docs/SETUP.md). In short:

1. Use Ubuntu 22.04 with an NVIDIA GPU. On Windows that means WSL.
2. Download four KITTI object archives and the two public checkpoints.
3. Build the `pcdet` conda environment: Python 3.9, torch 2.1.2+cu121, spconv-cu121.
4. Build the KITTI info files and run `test.py` once per checkpoint. Check the AP against the table above before going further.
5. Run the measurement scripts in order from `OpenPCDet/tools`.

```bash
conda activate pcdet
cd "$HOME/lidar-point-reliance/OpenPCDet/tools"
python week3_per_detection_table.py
python week4_baseline_stratification.py
python week5_intervention_harness.py
python week6_pilot.py
python week7_stratified.py
python week9_cross_detector.py
python week11_per_class.py
python week12_strategies.py
python week12_ablations.py --part draw
python week12_ablations.py --part probe
python week12_ablations.py --part strategy
python week12_ablations.py --part floor
python week12_ablations.py --part association
python week12_ablations.py --part report
```

Things to know before running:

- Every script reads saved predictions and edits clouds in memory. No script overwrites a Velodyne file or `result.pkl`.
- The long scripts (`week7`, `week9`, `week11`, `week12_ablations`) run from tens of thousands to a few hundred thousand forward passes each. They append one row per finished cell and resume from the finished rows when rerun.
- `week11_per_class.py` and the forward parts of `week12_ablations.py` exit with code 75 when they pause. Rerun the same command.
- The draw seed is 6908.
- The summary files in `outputs by week/` are the results of record. Read them there; do not launch a long run to inspect a number.

## Scripts and what they produce

| Script | What it does | Results folder |
| --- | --- | --- |
| `week3_per_detection_table.py` | One row per predicted Car box: range, interior point count, box size, best IoU to ground truth | `week3-outputs` |
| `week4_baseline_stratification.py` | AP, confidence, and calibration error by range, point count, and volume; reliability diagrams | `week4-outputs` |
| `week5_intervention_harness.py` | Correctness check of the edit, the forward pass, and the association on a five-target smoke set | `week5-outputs` |
| `week6_pilot.py` | 200-detection pilot that fixed the metric definitions | `week6-outputs` |
| `week7_stratified.py` | Primary run: every eligible PointPillars Car under the interior, placebo, and context edits | `week7-outputs` |
| `week9_cross_detector.py` | Same protocol on PV-RCNN, and the comparison with PointPillars | `week9-outputs` |
| `week11_per_class.py` | Same protocol on PointPillars Pedestrian and Cyclist | `week11-outputs` |
| `week12_strategies.py` | Removal strategies, association rules, and table math, with a self-test (numpy only) | – |
| `week12_ablations.py` | Ablations: removal strategy, confidence thresholds, association rule | `week12-outputs` |

From `week7` on, each results folder has a `*_summary.txt` with the headline numbers and a `*_table.csv` with every stratum. The locked settings are in that folder's definitions or sidecar file.

## Status

| Stage | State |
| --- | --- |
| Baseline stratification | Done |
| Primary run and controls (PointPillars, Car) | Done |
| Cross-architecture comparison (PV-RCNN) | Done |
| Pedestrian and Cyclist | Done |
| Ablations | Done |
| Qualitative gallery | Not started |
| Final tables and reproducibility check from committed code | Not started |
| Final report | In progress |

## Acknowledgements

- Detection code and pretrained checkpoints: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), Apache 2.0. Its license is at [`OpenPCDet/LICENSE`](OpenPCDet/LICENSE).
- Data: the [KITTI 3D object detection benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). KITTI is not redistributed here.
- The measurement is written against the occlusion result in OccAM's Laser (Schinagl et al., CVPR 2022).
