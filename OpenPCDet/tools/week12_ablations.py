"""Week-12 ablations on the PointPillars Car run.

The week-7 Car run is the reference. Each ablation changes one setting and
holds every other week-7 setting fixed:

  A. removal strategy: random (week-7 rows), farthest-point, voxel-wise
  B. confidence threshold: score-survival threshold and eligibility floor
  C. association: the IoU bar, and two alternative rules

This driver does not train, does not edit Velodyne files, and does not
overwrite result.pkl or any week-7 file. Run the parts in this order. A
forward part exits 75 after a chunk of forwards; rerun the same command
and finished cells are kept.

    python week12_ablations.py --part draw          # definitions, census, no forwards
    python week12_ablations.py --part probe         # unmodified batch-of-4 probe
    python week12_ablations.py --part strategy      # A: fps and voxel forwards
    python week12_ablations.py --part floor         # B: new targets with score 0.5 to < 0.7
    python week12_ablations.py --part association   # C: random rerun, every Car box saved
    python week12_ablations.py --part report        # audits, tables, summary, copy

Preflight on a few targets (refuses the official output directory):

    python week12_ablations.py --part strategy --max-targets 8 --output-dir ../output/week12_preflight
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
os.chdir(TOOLS_DIR)
sys.path.insert(0, str(TOOLS_DIR))

import _init_path  # noqa: F401

import torch
import yaml

import week7_stratified as week7
import week12_strategies as ws
from pcdet.datasets.kitti.kitti_object_eval_python.eval import d3_box_overlap
from pcdet.utils import box_utils
from week3_per_detection_table import (
    CLASS_NAME,
    camera_boxes_from_anno,
    count_points_in_boxes,
    load_velodyne,
    read_csv_rows,
)
from week5_intervention_harness import (
    ASSOC_IOU,
    F0_IOU_MIN,
    F0_SCORE_TOL,
    FRACTIONS,
    SEED,
    apply_delete,
    associate_to_original,
    box_from_row,
    choose_delete_indices,
    cloud_keeps_outside,
    confirm_locked_artifacts,
    confirm_week3_counts,
    file_stamp,
    num_to_remove,
    velodyne_bin_paths,
)
from week6_pilot import (
    EVAL_BATCH_SIZE,
    MIN_POINTS,
    PilotStop,
    RANGE_LABELS,
    SCORE_FLOOR,
    SURVIVAL_SCORE,
    annotate_row,
    check_f0_forward,
    forward_in_eval_batch,
    load_runtime,
)

# ---------------------------------------------------------------------------
# Locked week-12 values. Written into the definitions file before any forward.
# ---------------------------------------------------------------------------

SURVIVAL_THRESHOLDS = (0.3, 0.5, 0.7)
ELIGIBILITY_FLOORS = (0.5, 0.7, 0.9)
LOW_FLOOR = 0.5
ASSOCIATION_BARS = (0.3, 0.5, 0.7)
ALT_SCORE_BAR = 0.5
CENTER_RADIUS_M = 2.0
REPRO_TOL = 0.01
FPS_REFERENCE_MAX_POINTS = 300
AUDIT_STRIDE = 25
RESTART_AFTER_FORWARDS = 20000

EXPECTED_WEEK7_TARGETS = 11125
FRACTION_LABELS = week7.FRACTION_LABELS
RESULT_FRACTION_LABELS = ("0.25", "0.50", "0.75")
ASSOC_FRACTION_LABELS = ("0.25", "0.50", "0.75", "1.00")
CONDITIONS = week7.CONDITIONS
LABELS = {"range": RANGE_LABELS, "points": week7.POINT_LABELS, "size": week7.SIZE_LABELS}
EMPTY_OBSTACLES = np.zeros((0, 7), dtype=np.float32)

STRATEGY_COLUMNS = week7.RESULT_COLUMNS + [
    "strategy",
    "pillars_emptied",
    "partial_pillar_points",
    "fps_start",
]
FLOOR_COLUMNS = list(week7.RESULT_COLUMNS)
ASSOC_COLUMNS = week7.RESULT_COLUMNS + ["car_boxes"]

PARTS = ("draw", "probe", "strategy", "floor", "association", "report")
FORWARD_PARTS = ("strategy", "floor", "association")

OPENPCDET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    OPENPCDET_ROOT
    / "output/kitti_models/pointpillar/default/eval/epoch_7728/val/default/result.pkl"
)
DEFAULT_INFOS = OPENPCDET_ROOT / "data/kitti/kitti_infos_val.pkl"
DEFAULT_WEEK3 = OPENPCDET_ROOT / "output/week3/per_detection_car.csv"
DEFAULT_WALKED = OPENPCDET_ROOT / "output/week3/walked_frames.txt"
DEFAULT_WEEK4 = OPENPCDET_ROOT / "output/week4/week4_summary.txt"
DEFAULT_WEEK7 = OPENPCDET_ROOT / "output/week7"
DEFAULT_VELODYNE = OPENPCDET_ROOT / "data/kitti/training/velodyne"
DEFAULT_CKPT = OPENPCDET_ROOT / "checkpoints/pointpillar_7728.pth"
DEFAULT_CFG = OPENPCDET_ROOT / "tools/cfgs/kitti_models/pointpillar.yaml"
DEFAULT_DATA_ROOT = OPENPCDET_ROOT / "data/kitti"
DEFAULT_OUTPUT = OPENPCDET_ROOT / "output/week12"
DEFAULT_COPY = (
    OPENPCDET_ROOT.parent
    / "DIS Notes and Outputs"
    / "outputs by week"
    / "week12-outputs"
)

NAMES = {
    "definitions": "week12_definitions.txt",
    "preflight": "week12_preflight.txt",
    "stamps": "week12_input_stamps.json",
    "strategy": "week12_strategy_results.csv",
    "floor": "week12_floor05_results.csv",
    "association": "week12_association_results.csv",
    "strategy_table": "week12_removal_strategy_table.csv",
    "threshold_table": "week12_threshold_table.csv",
    "association_table": "week12_association_table.csv",
    "summary": "week12_summary.txt",
    "sample": "week12_sample.csv",
}
COPY_KEYS = (
    "definitions",
    "summary",
    "strategy_table",
    "threshold_table",
    "association_table",
    "sample",
)


class Week12Stop(RuntimeError):
    """A locked check failed. Finished rows already on disk are kept."""


class RestartChunk(Exception):
    """Enough new forwards for this process. Every CSV is flushed."""


# ---------------------------------------------------------------------------
# Arguments, paths, and small file helpers
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Week 12 ablations")
    parser.add_argument("--part", choices=PARTS, required=True)
    parser.add_argument("--result-pkl", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--infos-pkl", type=Path, default=DEFAULT_INFOS)
    parser.add_argument("--week3-csv", type=Path, default=DEFAULT_WEEK3)
    parser.add_argument("--walked-txt", type=Path, default=DEFAULT_WALKED)
    parser.add_argument("--week4-summary", type=Path, default=DEFAULT_WEEK4)
    parser.add_argument("--week7-dir", type=Path, default=DEFAULT_WEEK7)
    parser.add_argument("--velodyne-dir", type=Path, default=DEFAULT_VELODYNE)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--cfg-file", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--copy-dir", type=Path, default=DEFAULT_COPY)
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Preflight only. Truncates both target lists. Refuses the official output directory.",
    )
    return parser.parse_args()


def out_path(args, key):
    return Path(args.output_dir) / NAMES[key]


def refuse_preflight_on_official_dir(args):
    if args.max_targets is None:
        return
    if int(args.max_targets) <= 0:
        raise RuntimeError("max-targets must be positive")
    if Path(args.output_dir).resolve() == Path(DEFAULT_OUTPUT).resolve():
        raise RuntimeError("max-targets cannot write the official week-12 output directory")


def read_rows(path, columns):
    """Finished rows of one week-12 CSV. Missing or empty file returns []."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    week7.repair_trailing_partial(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError("%s header does not match the locked columns" % path.name)
        rows = []
        for raw in reader:
            if None in raw or any(raw[column] is None for column in columns):
                raise ValueError("%s has a row with the wrong width" % path.name)
            rows.append(raw)
    return rows


def results_started(args):
    """True when any week-12 forward CSV already holds a data row."""
    for key, columns in (
        ("strategy", STRATEGY_COLUMNS),
        ("floor", FLOOR_COLUMNS),
        ("association", ASSOC_COLUMNS),
    ):
        if read_rows(out_path(args, key), columns):
            return True
    return False


def copy_outputs(args):
    """Copy the concise files to the notes folder. Missing files are skipped."""
    if args.max_targets is not None or args.copy_dir is None:
        return
    copy_dir = Path(args.copy_dir)
    if not copy_dir.parent.exists():
        print("skip copy; parent missing: %s" % copy_dir.parent, flush=True)
        return
    copy_dir.mkdir(parents=True, exist_ok=True)
    for key in COPY_KEYS:
        source = out_path(args, key)
        if source.exists():
            shutil.copy2(str(source), str(copy_dir / NAMES[key]))


# ---------------------------------------------------------------------------
# Self-checks, grid, and targets
# ---------------------------------------------------------------------------


def self_check():
    """The numpy helpers must agree with the week-5 and week-7 code they replace."""
    ws.self_test()
    if ws.SEED != SEED:
        raise RuntimeError("week12_strategies SEED %d != week-5 SEED %d" % (ws.SEED, SEED))
    for frame_id, pred_idx in (("000002", 0), ("007169", 6)):
        for fraction in FRACTIONS:
            if ws.random_seed(frame_id, pred_idx, fraction) != week7.condition_draw_seed(
                "interior", frame_id, pred_idx, fraction
            ):
                raise RuntimeError("random seed no longer matches the week-7 interior seed")
    for n in range(0, 40):
        for fraction in FRACTIONS:
            if ws.num_to_remove(n, fraction) != num_to_remove(n, fraction):
                raise RuntimeError("num_to_remove drifted from week 5")
    if tuple("%.2f" % f for f in ws.RESULT_FRACTIONS) != RESULT_FRACTION_LABELS:
        raise RuntimeError("result fractions drifted")


def load_grid(cfg_file):
    """Pillar grid from the detector config. Not hard-coded."""
    with Path(cfg_file).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    data_cfg = cfg["DATA_CONFIG"]
    pc_range = [float(value) for value in data_cfg["POINT_CLOUD_RANGE"]]
    voxel_size = None
    for processor in data_cfg.get("DATA_PROCESSOR", []):
        if processor.get("NAME") == "transform_points_to_voxels":
            voxel_size = [float(value) for value in processor["VOXEL_SIZE"]]
    if len(pc_range) != 6 or voxel_size is None or len(voxel_size) != 3:
        raise Week12Stop("could not read POINT_CLOUD_RANGE and VOXEL_SIZE from %s" % cfg_file)
    return {"pc_range": pc_range, "voxel_size": voxel_size}


def week7_targets(rows3):
    """The 11,125 week-7 targets. Locked counts are checked inside week 7."""
    targets, _stats = week7.eligible_targets(rows3)
    if len(targets) != EXPECTED_WEEK7_TARGETS:
        raise Week12Stop("week-7 target count %d" % len(targets))
    return targets


def low_floor_targets(rows3):
    """New targets for floor 0.5: score from 0.5 up to below 0.7, at least one point."""
    chosen = [
        annotate_row(row)
        for row in rows3
        if LOW_FLOOR <= float(row["score"]) < SCORE_FLOOR
        and int(row["num_points_in_box"]) >= MIN_POINTS
    ]
    chosen.sort(key=lambda row: (row["frame_id"], int(row["pred_idx"])))
    keys = [(row["frame_id"], int(row["pred_idx"])) for row in chosen]
    if len(set(keys)) != len(keys):
        raise Week12Stop("duplicate low-floor identity")
    return chosen


def census_lines(targets, title):
    lines = ["%s: %d" % (title, len(targets))]
    for axis, field in (("range", "range_bin"), ("points", "num_points_bin"), ("size", "size_bin")):
        counts = ", ".join(
            "%s %d" % (label, sum(1 for t in targets if t[field] == label)) for label in LABELS[axis]
        )
        lines.append("%s by %s: %s" % (title, axis, counts))
    lines.append("%s gt_matched: %d" % (title, sum(1 for t in targets if t["gt_matched"])))
    return lines


def expected_forwards(n_week7, n_low):
    return {
        "strategy": n_week7 * len(ws.NEW_STRATEGIES) * len(RESULT_FRACTION_LABELS),
        "floor": n_low * len(FRACTION_LABELS),
        "association": n_week7 * len(ASSOC_FRACTION_LABELS),
    }


def definitions_text(args, grid, targets, low_targets):
    forwards = expected_forwards(len(targets), len(low_targets))
    lines = [
        "Week 12 definitions",
        "detector: PointPillars pointpillar_7728.pth",
        "config: tools/cfgs/kitti_models/pointpillar.yaml",
        "dataset: KITTI validation split, Car",
        "reference run: week 7 (%s)" % (Path(args.week7_dir) / "week7_results.csv").name,
        "reference eligibility: score >= %.2f and interior points >= %d" % (SCORE_FLOOR, MIN_POINTS),
        "reference fractions: %s" % ", ".join(FRACTION_LABELS),
        "seed: %d" % SEED,
        "num_removed: Python round(fraction * region_points), half even; identical across strategies",
        "eval batch size: %d. Only the target frame is edited." % EVAL_BATCH_SIZE,
        "reference association: returned Car box with the highest 3D IoU, survives at IoU >= %.2f" % ASSOC_IOU,
        "reference score survival: association and score_after >= %.2f" % SURVIVAL_SCORE,
        "retained: score_after / score_before on association, otherwise 0; no clipping",
        "bins: week-4 range, point-count, and volume edges, unchanged",
        "one setting changes per ablation; no two ablations are crossed",
        "",
        "A. removal strategy (interior condition only; placebo and context stay random)",
        "strategies: random, fps, voxel",
        "random: week-7 interior rows, not recomputed; token sha256(\"%d|interior|<frame_id>|<pred_idx>|<fraction>\")" % SEED,
        "fps: keep n - num_removed interior points by farthest-point sampling in x, y, z; delete the rest",
        "fps start: rng(seed).integers(n) over the sorted interior; seed from sha256(\"%d|fps|interior|<frame_id>|<pred_idx>|<fraction>\")" % SEED,
        "fps ties: lower sorted interior index; a chosen point is never chosen twice",
        "voxel grid: POINT_CLOUD_RANGE %s, VOXEL_SIZE %s (read from the config)"
        % (json.dumps(grid["pc_range"]), json.dumps(grid["voxel_size"])),
        "voxel cell: floor((xyz - range_min) / voxel_size)",
        "voxel order: rng(seed).permutation of the sorted occupied pillars; seed from sha256(\"%d|voxel|interior|<frame_id>|<pred_idx>|<fraction>\")" % SEED,
        "voxel walk: delete whole pillars while they fit; the first pillar that would overshoot gives up a random subset of exactly the remainder (same rng); stop",
        "result fractions: %s" % ", ".join(RESULT_FRACTION_LABELS),
        "fractions 0.00 and 1.00: the same edit under every strategy; checked at the edit level, table uses the week-7 rows",
        "",
        "B. confidence threshold (one threshold at a time)",
        "score-survival thresholds: %s (rescored from week-7 rows, all conditions)" % ", ".join("%.2f" % v for v in SURVIVAL_THRESHOLDS),
        "eligibility floors: %s (interior condition)" % ", ".join("%.2f" % v for v in ELIGIBILITY_FLOORS),
        "floor 0.90: subset of the week-7 rows, rescored",
        "floor 0.50: week-7 rows plus new targets with %.2f <= score < %.2f, random interior, all five fractions, week-7 token" % (LOW_FLOOR, SCORE_FLOOR),
        "floor 0.50 note: score survival at 0.50 is not comparable across floors for boxes that start just above 0.50",
    ]
    lines.extend(census_lines(low_targets, "floor 0.50 new targets"))
    lines.extend(
        [
            "",
            "C. association",
            "association bars: %s under the highest-IoU rule (rescored from week-7 best_iou and best_score)" % ", ".join("%.2f" % v for v in ASSOCIATION_BARS),
            "rule highest_iou: week-7 rule at IoU %.2f, recomputed from the saved boxes" % ASSOC_IOU,
            "rule highest_score_above_bar: highest-score returned Car box with 3D IoU >= %.2f" % ALT_SCORE_BAR,
            "rule nearest_center: returned Car box with the nearest BEV center within %.1f m, no IoU requirement" % CENTER_RADIUS_M,
            "rule rerun: week-7 random interior edits at %s, every returned Car box saved as [score, iou3d, bev_center_distance_m]" % ", ".join(ASSOC_FRACTION_LABELS),
            "rule tables: the iou_survival column counts boxes the rule associated",
            "reproducibility: every rerun row matches its week-7 best_iou and best_score within %.2f, otherwise stop" % REPRO_TOL,
            "",
            "week-7 targets: %d" % len(targets),
            "expected forwards: strategy %d, floor %d, association %d"
            % (forwards["strategy"], forwards["floor"], forwards["association"]),
            "max_targets: %s" % ("none" if args.max_targets is None else str(int(args.max_targets))),
            "",
        ]
    )
    return "\n".join(lines)


def ensure_definitions(args, grid, targets, low_targets):
    """Write the definitions. Once a result exists, they may not change."""
    path = out_path(args, "definitions")
    text = definitions_text(args, grid, targets, low_targets)
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if old != text and results_started(args):
            raise Week12Stop("definitions changed after results exist; refusing to continue")
    week7.atomic_write(path, text)
    return text


# ---------------------------------------------------------------------------
# Input stamps and the unmodified probe
# ---------------------------------------------------------------------------


def current_stamps(args):
    return {
        "result_pkl": list(file_stamp(args.result_pkl)),
        "checkpoint": list(file_stamp(args.ckpt)),
        "week7_results": list(file_stamp(Path(args.week7_dir) / "week7_results.csv")),
        "velodyne": {path.name: list(file_stamp(path)) for path in velodyne_bin_paths(args.velodyne_dir)},
    }


def check_stamps(args, record_if_missing):
    """Return (note, failures). Records the baseline on the first forward part."""
    path = out_path(args, "stamps")
    print("stamping velodyne, checkpoint, result.pkl, and the week-7 CSV", flush=True)
    now = current_stamps(args)
    if not path.exists():
        if not record_if_missing:
            return "no input baseline recorded (no forward part has run)", []
        week7.atomic_write(path, json.dumps(now) + "\n")
        return "input baseline recorded (%d Velodyne files)" % len(now["velodyne"]), []
    saved = json.loads(path.read_text(encoding="utf-8"))
    failures = [
        "%s changed since the week-12 baseline" % key
        for key in ("result_pkl", "checkpoint", "week7_results", "velodyne")
        if saved.get(key) != now.get(key)
    ]
    if failures:
        return "input stamp mismatch", failures
    return (
        "checkpoint, result.pkl, week-7 CSV, and %d Velodyne files unchanged" % len(now["velodyne"]),
        [],
    )


def probe_choices(targets, low_targets):
    chosen = []
    for group in (targets, low_targets):
        for label in RANGE_LABELS:
            for target in group:
                if target["range_bin"] == label:
                    chosen.append(target)
                    break
    return chosen


def preflight_is_current(path, chosen):
    path = Path(path)
    if not path.exists():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    if "decision: correctness passed" not in lines or "eval batch size: %d" % EVAL_BATCH_SIZE not in lines:
        return False
    probed = {tuple(line.split()[1:3]) for line in lines if line.startswith("probed:")}
    wanted = {(t["frame_id"], str(int(t["pred_idx"]))) for t in chosen}
    return wanted <= probed


def run_probe(args, targets, low_targets):
    """Unmodified forwards at batch size 4: IoU >= 0.99 and |score delta| < 0.01."""
    path = out_path(args, "preflight")
    chosen = probe_choices(targets, low_targets)
    if preflight_is_current(path, chosen):
        print("unmodified probe already passed", flush=True)
        return
    dataset, model, info_by_id, index_by_frame = load_runtime(args)
    lines = ["Week 12 unmodified probe", "targets: %d" % len(chosen), "eval batch size: %d" % EVAL_BATCH_SIZE]
    failures = []
    for target in chosen:
        frame_id = target["frame_id"]
        try:
            points = load_velodyne(args.velodyne_dir, frame_id)
            week7.target_interior(points, target)
            pred_dict = forward_in_eval_batch(dataset, model, info_by_id, index_by_frame, frame_id, points)
            calib = dataset.get_calib(info_by_id[frame_id]["point_cloud"]["lidar_idx"])
            best_score, best_iou, survived = associate_to_original(pred_dict, box_from_row(target), calib)
            check_f0_forward(target, best_score, best_iou, survived)
            lines.append(
                "probed: %s %d %s score %.4f iou %.4f"
                % (frame_id, int(target["pred_idx"]), target["range_bin"].replace(" ", "_"), float(target["score"]), float(best_iou))
            )
        except (Week12Stop, week7.Week7Stop, PilotStop) as exc:
            failures.append(str(exc))
            lines.append("failed: %s %d" % (frame_id, int(target["pred_idx"])))
    lines.append("correctness failures: %d" % len(failures))
    lines.extend("failure: %s" % item for item in failures)
    lines.append("decision: INVALID" if failures else "decision: correctness passed")
    week7.atomic_write(path, "\n".join(lines) + "\n")
    del model
    torch.cuda.empty_cache()
    if failures:
        raise Week12Stop("unmodified probe failed: %s" % failures[0])
    print("unmodified probe passed on %d targets" % len(chosen), flush=True)


# ---------------------------------------------------------------------------
# Edits
# ---------------------------------------------------------------------------


def strategy_delete(points, target, strategy, fraction, inside, grid):
    """Delete indices for one strategy cell. Returns (delete_idx, extras)."""
    frame_id = target["frame_id"]
    pred_idx = int(target["pred_idx"])
    n_removed = num_to_remove(inside.size, fraction)
    extras = {"pillars_emptied": "", "partial_pillar_points": "", "fps_start": ""}
    if strategy == "random":
        delete_idx = choose_delete_indices(
            inside, n_removed, week7.condition_draw_seed("interior", frame_id, pred_idx, fraction)
        )
    elif strategy == "fps":
        delete_idx, start = ws.fps_delete(
            inside, points[inside, 0:3], n_removed, ws.strategy_seed("fps", frame_id, pred_idx, fraction)
        )
        extras["fps_start"] = "" if start is None else str(int(start))
    elif strategy == "voxel":
        delete_idx, emptied, partial = ws.voxel_delete(
            inside,
            points[inside, 0:3],
            n_removed,
            ws.strategy_seed("voxel", frame_id, pred_idx, fraction),
            grid["pc_range"],
            grid["voxel_size"],
        )
        extras["pillars_emptied"] = str(int(emptied))
        extras["partial_pillar_points"] = str(int(partial))
    else:
        raise Week12Stop("unknown strategy %s" % strategy)
    return np.asarray(delete_idx, dtype=np.int64), n_removed, extras


def check_strategy_edit(points, target, strategy, fraction, inside, delete_idx, n_removed, modified, grid):
    """Stop when a strategy delete breaks the shared accounting."""
    prefix = "%s pred %d %s f=%.2f" % (target["frame_id"], int(target["pred_idx"]), strategy, float(fraction))
    if int(delete_idx.size) != int(n_removed):
        raise Week12Stop("%s deleted %d, expected %d" % (prefix, delete_idx.size, n_removed))
    if delete_idx.size != np.unique(delete_idx).size:
        raise Week12Stop("%s deleted a point twice" % prefix)
    if not cloud_keeps_outside(points, modified, delete_idx, inside):
        raise Week12Stop("%s changed a point outside the box or kept the wrong rows" % prefix)
    if float(fraction) == 0.0 and delete_idx.size != 0:
        raise Week12Stop("%s f=0 deleted points" % prefix)
    if float(fraction) == 1.0:
        after = int(count_points_in_boxes(modified[:, 0:3], box_from_row(target).reshape(1, 7))[0])
        if after != 0:
            raise Week12Stop("%s f=1 left interior points" % prefix)
    if strategy == "voxel":
        problems = ws.check_voxel_delete(
            inside, points[inside, 0:3], delete_idx, n_removed, grid["pc_range"], grid["voxel_size"]
        )
        if problems:
            raise Week12Stop("%s voxel shape: %s" % (prefix, "; ".join(problems)))


def car_box_list(pred_dict, original_box_lidar, calib):
    """Every returned Car box as [score, iou3d to the original, BEV center distance m].

    Same overlap call and output order as associate_to_original.
    """
    names = np.asarray(pred_dict["name"])
    if names.size == 0:
        return []
    mask = names == CLASS_NAME
    if not np.any(mask):
        return []
    new_cam = camera_boxes_from_anno(
        np.asarray(pred_dict["location"])[mask],
        np.asarray(pred_dict["dimensions"])[mask],
        np.asarray(pred_dict["rotation_y"])[mask],
    )
    original = np.asarray(original_box_lidar, dtype=np.float32).reshape(1, 7)
    original_cam = np.asarray(box_utils.boxes3d_lidar_to_kitti_camera(original, calib), dtype=np.float64).reshape(1, 7)
    overlaps = d3_box_overlap(
        np.ascontiguousarray(new_cam, dtype=np.float64),
        np.ascontiguousarray(original_cam, dtype=np.float64),
    )
    overlaps = np.asarray(overlaps, dtype=np.float64).reshape(-1)
    scores = np.asarray(pred_dict["score"], dtype=np.float64)[mask]
    lidar = np.asarray(pred_dict["boxes_lidar"], dtype=np.float64)[mask]
    dists = np.hypot(lidar[:, 0] - float(original[0, 0]), lidar[:, 1] - float(original[0, 1]))
    return [[float(s), float(i), float(d)] for s, i, d in zip(scores, overlaps, dists)]


def rounded_boxes(boxes):
    return [[round(b[0], 6), round(b[1], 6), round(b[2], 4)] for b in boxes]


# ---------------------------------------------------------------------------
# Forward parts
# ---------------------------------------------------------------------------


class Runner:
    """Shared runtime for one forward part: model, clouds, calib, progress."""

    def __init__(self, args):
        self.args = args
        self.dataset, self.model, self.info_by_id, self.index_by_frame = load_runtime(args)
        self.clouds = {}
        self.calib = {}
        self.forwarded = 0
        self.written = 0
        self.started = time.perf_counter()

    def cloud(self, frame_id):
        return week7.cached_cloud(self.clouds, self.args.velodyne_dir, frame_id)

    def calib_for(self, frame_id):
        if frame_id not in self.calib:
            self.calib.clear()
            self.calib[frame_id] = self.dataset.get_calib(self.info_by_id[frame_id]["point_cloud"]["lidar_idx"])
        return self.calib[frame_id]

    def forward(self, frame_id, points):
        pred_dict = forward_in_eval_batch(
            self.dataset, self.model, self.info_by_id, self.index_by_frame, frame_id, points
        )
        self.forwarded += 1
        if self.forwarded % 500 == 0:
            torch.cuda.empty_cache()
        return pred_dict

    def wrote(self, n_targets):
        self.written += 1
        week7.log_progress(self.started, self.forwarded, self.written, n_targets)

    def maybe_restart(self):
        if self.forwarded >= RESTART_AFTER_FORWARDS:
            raise RestartChunk("forwarded %d new cells" % self.forwarded)


def strategy_key(row):
    return (row["frame_id"], int(row["pred_idx"]), row["strategy"], row["fraction"])


def keyed(rows, key_fn, label):
    keys = set()
    for row in rows:
        key = key_fn(row)
        if key in keys:
            raise Week12Stop("duplicate %s row %s" % (label, (key,)))
        keys.add(key)
    return keys


def run_strategy(args, targets, grid):
    path = out_path(args, "strategy")
    done = keyed(read_rows(path, STRATEGY_COLUMNS), strategy_key, "strategy")
    print("strategy finished cells %d" % len(done), flush=True)
    runner = Runner(args)
    handle, writer = week7.open_writer(path, STRATEGY_COLUMNS, already_has_rows=path.exists() and path.stat().st_size > 0)
    try:
        for target in targets:
            frame_id = target["frame_id"]
            pred_idx = int(target["pred_idx"])
            pending = [
                (strategy, fraction, label)
                for strategy in ws.NEW_STRATEGIES
                for fraction, label in zip(ws.RESULT_FRACTIONS, RESULT_FRACTION_LABELS)
                if (frame_id, pred_idx, strategy, label) not in done
            ]
            if not pending:
                continue
            points = runner.cloud(frame_id)
            inside = week7.target_interior(points, target)
            calib = runner.calib_for(frame_id)
            for strategy, fraction, label in pending:
                delete_idx, n_removed, extras = strategy_delete(points, target, strategy, fraction, inside, grid)
                modified = apply_delete(points, delete_idx)
                check_strategy_edit(points, target, strategy, fraction, inside, delete_idx, n_removed, modified, grid)
                pred_dict = runner.forward(frame_id, modified)
                best_score, best_iou, survived = associate_to_original(pred_dict, box_from_row(target), calib)
                row = week7.metric_row(
                    target, "interior", label, int(inside.size), n_removed, "n/a", best_score, best_iou, survived
                )
                row["strategy"] = strategy
                row.update(extras)
                week7.append_row(handle, writer, row)
                done.add((frame_id, pred_idx, strategy, label))
                runner.wrote(len(targets))
            runner.maybe_restart()
    finally:
        handle.flush()
        handle.close()
    return runner.forwarded


def floor_key(row):
    return (row["frame_id"], int(row["pred_idx"]), row["fraction"])


def run_floor(args, low_targets):
    path = out_path(args, "floor")
    done = keyed(read_rows(path, FLOOR_COLUMNS), floor_key, "floor")
    print("floor finished cells %d" % len(done), flush=True)
    runner = Runner(args)
    handle, writer = week7.open_writer(path, FLOOR_COLUMNS, already_has_rows=path.exists() and path.stat().st_size > 0)
    try:
        for target in low_targets:
            frame_id = target["frame_id"]
            pred_idx = int(target["pred_idx"])
            pending = [
                (fraction, label)
                for fraction, label in zip(FRACTIONS, FRACTION_LABELS)
                if (frame_id, pred_idx, label) not in done
            ]
            if not pending:
                continue
            points = runner.cloud(frame_id)
            inside = week7.target_interior(points, target)
            calib = runner.calib_for(frame_id)
            for fraction, label in pending:
                if float(fraction) == 0.0:
                    modified, region_points, n_removed = points, int(inside.size), 0
                else:
                    modified, region_points, n_removed = week7.edit_condition(
                        points, target, "interior", fraction, inside, EMPTY_OBSTACLES
                    )
                pred_dict = runner.forward(frame_id, modified)
                best_score, best_iou, survived = associate_to_original(pred_dict, box_from_row(target), calib)
                if float(fraction) == 0.0:
                    check_f0_forward(target, best_score, best_iou, survived)
                row = week7.metric_row(
                    target, "interior", label, region_points, n_removed, "n/a", best_score, best_iou, survived
                )
                week7.append_row(handle, writer, row)
                done.add((frame_id, pred_idx, label))
                runner.wrote(len(low_targets))
            runner.maybe_restart()
    finally:
        handle.flush()
        handle.close()
    return runner.forwarded


def run_association(args, targets):
    path = out_path(args, "association")
    done = keyed(read_rows(path, ASSOC_COLUMNS), floor_key, "association")
    print("association finished cells %d" % len(done), flush=True)
    runner = Runner(args)
    handle, writer = week7.open_writer(path, ASSOC_COLUMNS, already_has_rows=path.exists() and path.stat().st_size > 0)
    fractions = [(float(label), label) for label in ASSOC_FRACTION_LABELS]
    try:
        for target in targets:
            frame_id = target["frame_id"]
            pred_idx = int(target["pred_idx"])
            pending = [(f, label) for f, label in fractions if (frame_id, pred_idx, label) not in done]
            if not pending:
                continue
            points = runner.cloud(frame_id)
            inside = week7.target_interior(points, target)
            calib = runner.calib_for(frame_id)
            box = box_from_row(target)
            for fraction, label in pending:
                modified, region_points, n_removed = week7.edit_condition(
                    points, target, "interior", fraction, inside, EMPTY_OBSTACLES
                )
                pred_dict = runner.forward(frame_id, modified)
                best_score, best_iou, survived = associate_to_original(pred_dict, box, calib)
                boxes = car_box_list(pred_dict, box, calib)
                rule_score, rule_iou, rule_survived = ws.rule_highest_iou(boxes, ASSOC_IOU)
                same = (rule_survived == survived) and abs(rule_iou - float(best_iou)) < 1.0e-9
                same = same and ((rule_score is None) == (best_score is None))
                if same and rule_score is not None:
                    same = abs(rule_score - float(best_score)) < 1.0e-9
                if not same:
                    raise Week12Stop(
                        "%s pred %d f=%s saved boxes disagree with associate_to_original" % (frame_id, pred_idx, label)
                    )
                row = week7.metric_row(
                    target, "interior", label, region_points, n_removed, "n/a", best_score, best_iou, survived
                )
                row["car_boxes"] = json.dumps(rounded_boxes(boxes), separators=(",", ":"))
                week7.append_row(handle, writer, row)
                done.add((frame_id, pred_idx, label))
                runner.wrote(len(targets))
            runner.maybe_restart()
    finally:
        handle.flush()
        handle.close()
    return runner.forwarded


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def load_week7_rows(args, targets):
    """Week-7 rows, indexed. The week-7 run must be complete and passed."""
    week7_dir = Path(args.week7_dir)
    summary = (week7_dir / "week7_summary.txt").read_text(encoding="utf-8")
    if "decision: correctness passed" not in summary.splitlines():
        raise Week12Stop("week-7 summary is not 'decision: correctness passed'")
    rows = week7.read_result_rows(week7_dir / "week7_results.csv")
    by_key = {}
    for row in rows:
        key = (row["frame_id"], int(row["pred_idx"]), row["condition"], row["fraction"])
        if key in by_key:
            raise Week12Stop("duplicate week-7 row %s" % (key,))
        by_key[key] = row
    wanted = {(t["frame_id"], int(t["pred_idx"])) for t in targets}
    kept = {key: row for key, row in by_key.items() if (key[0], key[1]) in wanted}
    expected = len(targets) * len(CONDITIONS) * len(FRACTION_LABELS)
    if len(kept) != expected:
        raise Week12Stop("week-7 rows for these targets: %d, expected %d" % (len(kept), expected))
    return kept


def audit_metric_rows(rows, targets, expected_keys, key_fn, label):
    """Shared field checks on week-12 rows built by week7.metric_row."""
    failures = []
    by_id = {(t["frame_id"], int(t["pred_idx"])): t for t in targets}
    seen = set()
    for row in rows:
        key = key_fn(row)
        where = "%s %s" % (label, (key,))
        if key in seen:
            failures.append("duplicate %s" % where)
        seen.add(key)
        if key not in expected_keys:
            failures.append("unexpected %s" % where)
            continue
        target = by_id[(row["frame_id"], int(row["pred_idx"]))]
        if abs(float(row["score_before"]) - float(target["score"])) > 1.0e-6:
            failures.append("score_before drifted at %s" % where)
        if int(row["num_points_before"]) != int(target["num_points_in_box"]):
            failures.append("num_points_before drifted at %s" % where)
        if int(row["region_points"]) != int(row["num_points_before"]):
            failures.append("region_points != interior at %s" % where)
        fraction = float(row["fraction"])
        if int(row["num_removed"]) != num_to_remove(int(row["region_points"]), fraction):
            failures.append("num_removed does not match round at %s" % where)
        if row["survived_iou05"] != "true":
            if row["retained"] != "0.000000" or row["score_after"] != "":
                failures.append("miss is not stored as retained 0 at %s" % where)
        else:
            ratio = float(row["score_after"]) / float(row["score_before"])
            if abs(float(row["retained"]) - ratio) > week7.RETAINED_TOL:
                failures.append("retained != score_after / score_before at %s" % where)
            if (row["survived_score05"] == "true") != (float(row["score_after"]) >= SURVIVAL_SCORE):
                failures.append("survived_score05 disagrees with score_after at %s" % where)
        if fraction == 0.0:
            if row["survived_iou05"] != "true" or float(row["best_iou"]) < F0_IOU_MIN:
                failures.append("f=0 iou below %.2f at %s" % (F0_IOU_MIN, where))
            elif abs(float(row["score_after"]) - float(row["score_before"])) >= F0_SCORE_TOL:
                failures.append("f=0 score moved at %s" % where)
    missing = expected_keys - seen
    if missing:
        failures.append("%s missing %d cells, first %s" % (label, len(missing), (sorted(missing)[0],)))
    return failures


def strategy_expected(targets):
    return {
        (t["frame_id"], int(t["pred_idx"]), s, label)
        for t in targets
        for s in ws.NEW_STRATEGIES
        for label in RESULT_FRACTION_LABELS
    }


def fraction_expected(targets, labels):
    return {(t["frame_id"], int(t["pred_idx"]), label) for t in targets for label in labels}


def audit_strategy_edits(args, targets, grid, rows_by_key):
    """Recompute deletes on every AUDIT_STRIDE-th target, all five fractions."""
    failures = []
    clouds = {}
    checked = 0
    fps_ref = 0
    for target in targets[::AUDIT_STRIDE]:
        frame_id = target["frame_id"]
        pred_idx = int(target["pred_idx"])
        try:
            points = week7.cached_cloud(clouds, args.velodyne_dir, frame_id)
            inside = week7.target_interior(points, target)
            for fraction, label in zip(FRACTIONS, FRACTION_LABELS):
                random_idx, _n, _x = strategy_delete(points, target, "random", fraction, inside, grid)
                for strategy in ws.NEW_STRATEGIES:
                    delete_idx, n_removed, extras = strategy_delete(points, target, strategy, fraction, inside, grid)
                    modified = apply_delete(points, delete_idx)
                    check_strategy_edit(points, target, strategy, fraction, inside, delete_idx, n_removed, modified, grid)
                    if label in ("0.00", "1.00") and not np.array_equal(delete_idx, random_idx):
                        failures.append("%s pred %d %s f=%s differs from random" % (frame_id, pred_idx, strategy, label))
                    if strategy == "fps" and 0 < n_removed < inside.size <= FPS_REFERENCE_MAX_POINTS:
                        n = int(inside.size)
                        start = ws.fps_start(n, ws.strategy_seed("fps", frame_id, pred_idx, fraction))
                        keep = ws.fps_reference(points[inside, 0:3], n - n_removed, start)
                        mask = np.ones((n,), dtype=bool)
                        mask[keep] = False
                        if not np.array_equal(inside[mask], delete_idx):
                            failures.append("%s pred %d fps f=%s differs from the reference" % (frame_id, pred_idx, label))
                        fps_ref += 1
                    row = rows_by_key.get((frame_id, pred_idx, strategy, label))
                    if label in RESULT_FRACTION_LABELS:
                        if row is None:
                            failures.append("%s pred %d %s f=%s has no row" % (frame_id, pred_idx, strategy, label))
                            continue
                        if int(row["num_removed"]) != int(n_removed):
                            failures.append("%s pred %d %s f=%s num_removed drifted" % (frame_id, pred_idx, strategy, label))
                        for field in ("pillars_emptied", "partial_pillar_points", "fps_start"):
                            if row[field] != extras[field]:
                                failures.append("%s pred %d %s f=%s %s drifted" % (frame_id, pred_idx, strategy, label, field))
                    checked += 1
        except (Week12Stop, week7.Week7Stop, PilotStop, ValueError, RuntimeError) as exc:
            failures.append(str(exc))
    return failures, checked, fps_ref


def reproducibility(assoc_rows, week7_rows):
    """Compare every rerun row with its week-7 interior row."""
    failures = []
    worst_iou = 0.0
    worst_score = 0.0
    flips = 0
    for row in assoc_rows:
        key = (row["frame_id"], int(row["pred_idx"]), "interior", row["fraction"])
        old = week7_rows.get(key)
        where = "%s pred %s f=%s" % (row["frame_id"], row["pred_idx"], row["fraction"])
        if old is None:
            failures.append("no week-7 row for %s" % where)
            continue
        if row["num_removed"] != old["num_removed"]:
            failures.append("num_removed differs from week 7 at %s" % where)
        d_iou = abs(float(row["best_iou"]) - float(old["best_iou"]))
        worst_iou = max(worst_iou, d_iou)
        if d_iou > REPRO_TOL:
            failures.append("best_iou differs by %.4f at %s" % (d_iou, where))
        if (row["best_score"] == "") != (old["best_score"] == ""):
            failures.append("one run returned no Car at %s" % where)
        elif row["best_score"] != "":
            d_score = abs(float(row["best_score"]) - float(old["best_score"]))
            worst_score = max(worst_score, d_score)
            if d_score > REPRO_TOL:
                failures.append("best_score differs by %.4f at %s" % (d_score, where))
        if row["survived_iou05"] != old["survived_iou05"]:
            flips += 1
    return failures, {"rows": len(assoc_rows), "max_iou_delta": worst_iou, "max_score_delta": worst_score, "flag_flips": flips}


# ---------------------------------------------------------------------------
# Records for the tables
# ---------------------------------------------------------------------------


def stored_record(row, score_threshold=SURVIVAL_SCORE, condition=None):
    """A week-7-style row as a scored record. Retained is the stored value."""
    survived = row["survived_iou05"] == "true"
    score_ok = survived and float(row["score_after"]) >= float(score_threshold)
    return ws.record(
        row,
        row["fraction"],
        int(row["num_removed"]),
        float(row["retained"]),
        score_ok,
        survived,
        condition=condition or row["condition"],
    )


def bar_record(row, bar):
    """Rescore a week-7 row at a different IoU bar under the highest-IoU rule."""
    has_box = row["best_score"] != ""
    survived = has_box and float(row["best_iou"]) >= float(bar)
    score = float(row["best_score"]) if has_box else None
    retained = ws.retained_from(float(row["score_before"]), score, survived)
    score_ok = survived and score >= SURVIVAL_SCORE
    return ws.record(row, row["fraction"], int(row["num_removed"]), retained, score_ok, survived, condition=row["condition"])


def rule_record(row, rule):
    boxes = json.loads(row["car_boxes"])
    if rule == "highest_iou":
        score, _metric, survived = ws.rule_highest_iou(boxes, ASSOC_IOU)
    elif rule == "highest_score_above_bar":
        score, _metric, survived = ws.rule_highest_score(boxes, ALT_SCORE_BAR)
    elif rule == "nearest_center":
        score, _metric, survived = ws.rule_nearest_center(boxes, CENTER_RADIUS_M)
    else:
        raise ValueError(rule)
    retained = ws.retained_from(float(row["score_before"]), score, survived)
    score_ok = survived and score >= SURVIVAL_SCORE
    return ws.record(row, row["fraction"], int(row["num_removed"]), retained, score_ok, survived)


def applied(rows):
    """Drop unavailable placebo rows. Week 7 had none, but the rule stays."""
    return [row for row in rows if row["retained"] != ""]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ws.TABLE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def table_lines(rows, fractions):
    """Compact summary lines: one per setting, condition, and stratum."""
    lines = []
    grouped = {}
    order = []
    for row in rows:
        key = (row["ablation"], row["setting"], row["condition"], row["stratum_type"], row["stratum"])
        if key not in grouped:
            grouped[key] = {}
            order.append(key)
        grouped[key][row["fraction"]] = row
    for key in order:
        cells = grouped[key]
        n = next(iter(cells.values()))["n"]
        parts = []
        for label in fractions:
            cell = cells.get(label)
            if cell is None or cell["n"] == "0":
                continue
            parts.append(
                "f=%s r=%s score=%s iou=%s"
                % (label, cell["mean_retained"][:5], cell["score_survival_hits"], cell["iou_survival_hits"])
            )
        body = "  ".join(parts) if parts else "empty"
        lines.append("%s %s | %s | %s %s n=%s  %s" % (key[0], key[1], key[2], key[3], key[4], n, body))
    return lines


def write_sample(args, strategy_rows):
    path = out_path(args, "sample")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STRATEGY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in strategy_rows[:60]:
            writer.writerow(row)


def run_report(args, targets, low_targets, grid):
    week7_rows = load_week7_rows(args, targets)
    week7_list = list(week7_rows.values())
    lines = [
        "Week 12 summary",
        "detector: PointPillars, Car, KITTI val",
        "week-7 targets: %d" % len(targets),
        "floor 0.50 new targets: %d" % len(low_targets),
        "max_targets: %s" % ("none" if args.max_targets is None else str(int(args.max_targets))),
    ]
    file_note, file_failures = check_stamps(args, record_if_missing=False)
    lines.append("inputs: %s" % file_note)
    lines.extend("input failure: %s" % item for item in file_failures)
    preflight = out_path(args, "preflight")
    probe_ok = preflight.exists() and "decision: correctness passed" in preflight.read_text(encoding="utf-8").splitlines()
    lines.append("unmodified probe: %s" % ("passed" if probe_ok else "not passed"))
    global_fail = list(file_failures)
    if not probe_ok:
        global_fail.append("unmodified probe has not passed")

    # Bar 0.5 rescore must reproduce the stored week-7 retention exactly.
    bar_check = []
    for row in applied(week7_list):
        rec = bar_record(row, ASSOC_IOU)
        if abs(rec["retained"] - float(row["retained"])) > week7.RETAINED_TOL:
            bar_check.append("bar 0.50 rescore != stored retained at %s pred %s" % (row["frame_id"], row["pred_idx"]))
            break
    global_fail.extend(bar_check)

    # A. strategy
    strategy_rows = read_rows(out_path(args, "strategy"), STRATEGY_COLUMNS)
    strategy_by_key = {strategy_key(row): row for row in strategy_rows}
    expected = strategy_expected(targets)
    lines.append("")
    lines.append("A. removal strategy: %d / %d cells" % (len(strategy_rows), len(expected)))
    strategy_table = []
    if len(strategy_rows) == 0:
        lines.append("  has not been run")
    else:
        fails = audit_metric_rows(strategy_rows, targets, expected, strategy_key, "strategy")
        edit_fails, checked, fps_ref = audit_strategy_edits(args, targets, grid, strategy_by_key)
        fails.extend(edit_fails)
        fails.extend(global_fail)
        lines.append("  edit audit: %d cells recomputed, %d checked against the reference FPS" % (checked, fps_ref))
        lines.append("  failures: %d" % len(fails))
        lines.extend("  %s" % item for item in fails[:50])
        if fails:
            lines.append("  decision: INVALID")
        else:
            lines.append("  decision: correctness passed")
            interior7 = [row for row in week7_list if row["condition"] == "interior"]
            ends = [row for row in interior7 if row["fraction"] in ("0.00", "1.00")]
            records = {"random": [stored_record(row) for row in interior7]}
            for strategy in ws.NEW_STRATEGIES:
                mine = [stored_record(row) for row in strategy_rows if row["strategy"] == strategy]
                records[strategy] = mine + [stored_record(row) for row in ends]
            for strategy in ws.STRATEGIES:
                strategy_table.extend(
                    ws.table_rows("removal_strategy", strategy, records[strategy], LABELS, FRACTION_LABELS)
                )
            write_table(out_path(args, "strategy_table"), strategy_table)
            lines.extend("  " + item for item in table_lines(strategy_table, FRACTION_LABELS))

    # B. thresholds
    lines.append("")
    threshold_table = []
    survival_rows = applied(week7_list)
    for threshold in SURVIVAL_THRESHOLDS:
        records = [stored_record(row, score_threshold=threshold) for row in survival_rows]
        threshold_table.extend(
            ws.table_rows("survival_threshold", "%.2f" % threshold, records, LABELS, FRACTION_LABELS, conditions=CONDITIONS)
        )
    interior7 = [row for row in week7_list if row["condition"] == "interior"]
    floor_rows = read_rows(out_path(args, "floor"), FLOOR_COLUMNS)
    floor_expected = fraction_expected(low_targets, FRACTION_LABELS)
    lines.append("B. confidence threshold: floor 0.50 new cells %d / %d" % (len(floor_rows), len(floor_expected)))
    floor_ok = False
    if floor_rows:
        fails = audit_metric_rows(floor_rows, low_targets, floor_expected, floor_key, "floor") + global_fail
        lines.append("  floor 0.50 failures: %d" % len(fails))
        lines.extend("  %s" % item for item in fails[:50])
        floor_ok = not fails
        lines.append("  floor 0.50 decision: %s" % ("correctness passed" if floor_ok else "INVALID"))
    else:
        lines.append("  floor 0.50 has not been run; floors 0.70 and 0.90 and the survival thresholds need no forwards")
    for floor in ELIGIBILITY_FLOORS:
        if floor == LOW_FLOOR:
            if not floor_ok:
                continue
            chosen = interior7 + floor_rows
        else:
            chosen = [row for row in interior7 if float(row["score_before"]) >= floor]
        records = [stored_record(row) for row in chosen]
        threshold_table.extend(ws.table_rows("eligibility_floor", "%.2f" % floor, records, LABELS, FRACTION_LABELS))
    if not global_fail:
        write_table(out_path(args, "threshold_table"), threshold_table)
        lines.extend("  " + item for item in table_lines(threshold_table, FRACTION_LABELS))
    else:
        lines.append("  threshold table not written: %s" % "; ".join(global_fail))

    # C. association
    lines.append("")
    association_table = []
    for bar in ASSOCIATION_BARS:
        records = [bar_record(row, bar) for row in applied(week7_list)]
        association_table.extend(
            ws.table_rows("association_bar", "%.2f" % bar, records, LABELS, FRACTION_LABELS, conditions=CONDITIONS)
        )
    assoc_rows = read_rows(out_path(args, "association"), ASSOC_COLUMNS)
    assoc_expected = fraction_expected(targets, ASSOC_FRACTION_LABELS)
    lines.append("C. association: rule rerun cells %d / %d" % (len(assoc_rows), len(assoc_expected)))
    if assoc_rows:
        fails = audit_metric_rows(assoc_rows, targets, assoc_expected, floor_key, "association") + global_fail
        repro_fails, repro = reproducibility(assoc_rows, week7_rows)
        fails.extend(repro_fails)
        lines.append(
            "  reproducibility vs week 7: rows %d, max |d iou| %.4f, max |d score| %.4f, survival flag flips %d"
            % (repro["rows"], repro["max_iou_delta"], repro["max_score_delta"], repro["flag_flips"])
        )
        lines.append("  rule failures: %d" % len(fails))
        lines.extend("  %s" % item for item in fails[:50])
        if fails:
            lines.append("  rule decision: INVALID")
        else:
            lines.append("  rule decision: correctness passed")
            for rule in ws.ASSOCIATION_RULES:
                records = [rule_record(row, rule) for row in assoc_rows]
                association_table.extend(
                    ws.table_rows("association_rule", rule, records, LABELS, ASSOC_FRACTION_LABELS)
                )
    else:
        lines.append("  rule rerun has not been run; the bar settings need no forwards")
    if not global_fail:
        write_table(out_path(args, "association_table"), association_table)
        lines.extend("  " + item for item in table_lines(association_table, FRACTION_LABELS))
    else:
        lines.append("  association table not written: %s" % "; ".join(global_fail))

    if strategy_rows:
        write_sample(args, strategy_rows)
    lines.append("")
    text = "\n".join(lines)
    week7.atomic_write(out_path(args, "summary"), text)
    copy_outputs(args)
    print(text, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    refuse_preflight_on_official_dir(args)
    self_check()
    confirm_locked_artifacts(args)
    week7_csv = Path(args.week7_dir) / "week7_results.csv"
    if not week7_csv.exists():
        raise FileNotFoundError("week-7 results missing: %s" % week7_csv)
    rows3 = read_csv_rows(args.week3_csv)
    confirm_week3_counts(rows3, args.walked_txt)
    targets = week7_targets(rows3)
    low_targets = low_floor_targets(rows3)
    if args.max_targets is not None:
        targets = targets[: int(args.max_targets)]
        low_targets = low_targets[: int(args.max_targets)]
    grid = load_grid(args.cfg_file)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    text = ensure_definitions(args, grid, targets, low_targets)

    if args.part == "draw":
        print(text, flush=True)
        copy_outputs(args)
        return
    if args.part == "report":
        run_report(args, targets, low_targets, grid)
        return

    note, failures = check_stamps(args, record_if_missing=True)
    print(note, flush=True)
    if failures:
        raise Week12Stop("; ".join(failures))
    try:
        run_probe(args, targets, low_targets)
        if args.part == "probe":
            return
        if args.part == "strategy":
            forwarded = run_strategy(args, targets, grid)
        elif args.part == "floor":
            forwarded = run_floor(args, low_targets)
        else:
            forwarded = run_association(args, targets)
        print("%s forwards this process %d" % (args.part, forwarded), flush=True)
        print("%s complete. Run --part report once every part is done." % args.part, flush=True)
    except RestartChunk as exc:
        print("chunk stop: %s. Rerun the same command." % exc, flush=True)
        raise SystemExit(75)
    except (Week12Stop, week7.Week7Stop, PilotStop) as exc:
        print("stopped: %s" % exc, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
