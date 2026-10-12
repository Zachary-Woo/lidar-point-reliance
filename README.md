# When does a LiDAR detector stop relying on the object?

A stratified study of evidence dependence in 3D detection confidence.

One public KITTI checkpoint is held fixed. The measurement is how much its confidence depends on the LiDAR points inside a predicted box versus the surrounding scan, and whether that dependence is stronger when the object's own evidence is weak (far, sparse, or small). Nothing is trained. The detector weights are not edited.

The primary run is PointPillars (`pointpillar_7728.pth`) on the KITTI val split, Car only. Official Car moderate 3D AP@R11 on the saved predictions is **77.2898** (published 77.28). Compare that R11 number only. Do not compare AP@R40 to 77.28.

## Layout

The runnable tree is the Ubuntu clone, not this Windows folder.

| Piece | Path |
| --- | --- |
| Code, env, outputs | `~/OpenPCDet` on Ubuntu 22.04 under WSL |
| This Windows tree | `OpenPCDet/` here, linked into that clone |
| KITTI clouds, images, calib, labels | Windows `data/data_object_*`, joined into `OpenPCDet/data/kitti/training` and `testing` |
| Checkpoint | `~/OpenPCDet/checkpoints/pointpillar_7728.pth` |
| Val infos | `~/OpenPCDet/data/kitti/kitti_infos_val.pkl` |
| Saved val predictions | `~/OpenPCDet/output/kitti_models/pointpillar/default/eval/epoch_7728/val/default/result.pkl` |

Open a Ubuntu 22.04 terminal and `conda activate pcdet`. The known stack is Python 3.9, torch 2.1.2+cu121, spconv-cu121, and the CUDA 12.1 toolkit bits. Closing the terminal does not remove the env. Do not reinstall Miniconda or OpenPCDet unless the distro itself is gone.

KITTI infos are created once:

```bash
conda activate pcdet
cd ~/OpenPCDet
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

The val predictions are already stored. Regenerate them only if `result.pkl` is missing:

```bash
cd ~/OpenPCDet/tools
python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --batch_size 4 --ckpt ~/OpenPCDet/checkpoints/pointpillar_7728.pth
```

## Local change inside OpenPCDet

Stock KITTI eval computes rotated-box overlap with Numba CUDA. On this WSL machine that path segfaults even when torch CUDA works. The replacement is

`OpenPCDet/pcdet/datasets/kitti/kitti_object_eval_python/rotate_iou.py`

The public function is still `rotate_iou_gpu_eval`. It now runs the same IoU on CPU Numba. `device_id` is accepted and ignored so existing callers stay valid. Identical boxes are treated as a full overlap, because the polygon clipper returns a bad area when two boxes share edges.

`OpenPCDet/pcdet/datasets/kitti/kitti_object_eval_python/eval.py` is unchanged. It imports `rotate_iou_gpu_eval` and uses it for the BEV overlap inside 3D matching. The Windows copy of `rotate_iou.py` is the source of truth. If a fresh Ubuntu clone still segfaults during eval, copy this file over the upstream one.

A `NumbaPerformanceWarning` about `parallel=True` is not a failure.

## Measurement scripts

The study's own code lives in `OpenPCDet/tools/`. Those filenames still use a temporary `weekN_` prefix. The prefix is only the name. Defaults point at `~/OpenPCDet` and write under `~/OpenPCDet/output/`. Each script also copies its summary into `DIS Notes and Outputs/outputs by week/`.

Edits to a cloud are in memory. The scripts do not overwrite `training/velodyne` or `result.pkl`.

Run them from `~/OpenPCDet/tools` after `conda activate pcdet`. Bare commands use the locked paths.

**Per-detection table.** `week3_per_detection_table.py` walks every KITTI val frame in the saved predictions and writes one row per Car box: sensor range `sqrt(x^2+y^2)`, interior point count on the full velodyne cloud, box size, best 3D IoU to a Car ground-truth box, and a match flag at IoU 0.5. Raw IoU is stored so later scoring can use the official 0.7 overlap without rebuilding the table.

```bash
python week3_per_detection_table.py
```

Output: `~/OpenPCDet/output/week3/per_detection_car.csv`. The locked table is 3769 frames, 33133 Car rows, 13458 matched at 3D IoU >= 0.5.

**Baseline stratification.** `week4_baseline_stratification.py` reads that table and the same `result.pkl`. Average precision is official KITTI 3D AP, with ground-truth cars outside a bin ignored. Confidence and expected calibration error are computed on the predictions. Bins are sensor range (0–20 m, 20–40 m, 40 m and farther), interior point count (0–19, 20–99, 100+), and box volume in cubic meters (under 8, 8–12, 12+).

```bash
python week4_baseline_stratification.py
```

Output: `~/OpenPCDet/output/week4/`. The unstratified moderate AP@R11 in `week4_summary.txt` must still be 77.2898. Reliability diagrams are `reliability_range.png`, `reliability_num_points.png`, and `reliability_size.png`.

**Intervention harness.** `week5_intervention_harness.py` is the correctness check. It deletes a fraction of the points inside one predicted box, runs PointPillars on that cloud, and associates the new Car box to the original prediction at 3D IoU 0.5. Fractions are 0, 0.25, 0.5, 0.75, and 1. The draw seed is 6908. The script forwards a five-target smoke set, not the full val split.

```bash
python week5_intervention_harness.py
```

Output: `~/OpenPCDet/output/week5/week5_check_log.txt`. A good run has zero failed checks, fraction 0 reproduces the stored box, and the velodyne files are unchanged.

**Pilot.** `week6_pilot.py` draws 200 high-confidence Car detections (score at least 0.7 and at least one interior point), stratified by range, and runs the same interior deletion. Retained confidence is the new score divided by the original score when the 0.5 IoU bar is met, and 0 when it is not. Score survival also requires the returned score to be at least 0.5.

```bash
python week6_pilot.py
```

Output: `~/OpenPCDet/output/week6/week6_summary.txt`. The recorded decision on those definitions is GO.

**Full stratified run.** `week7_stratified.py` applies the locked metrics to every eligible Car (11125 rows) under three edits: interior deletion, deletion inside a same-size empty box (placebo), and deletion of points outside the box (context). This is the long job, on the order of 10^5 forwards. A restart skips a cell that is already in the results CSV. `--draw-only` writes the target list and stops. `--max-targets` is a preflight and refuses the official output directory.

```bash
python week7_stratified.py
```

Output: `~/OpenPCDet/output/week7/`. Read `week7_summary.txt` for the marginal rates and `week7_table.csv` for every stratum, including the joint cells. Those files are the results of record. Do not launch this script to inspect them.
