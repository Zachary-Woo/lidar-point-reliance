"""Week-7 full stratified run on the locked week-6 metrics.

Every eligible Car is forwarded under three in-memory edits: interior
removal, a matched-volume empty-region placebo, and removal of points
outside the box. This driver does not train, does not edit velodyne bins,
and does not overwrite result.pkl.
"""

import argparse
import csv
import hashlib
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

# pcdet reads a git commit at import time from the current directory.
os.chdir(Path(__file__).resolve().parent)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _init_path  # noqa: F401

from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu

from week3_per_detection_table import (
    count_points_in_boxes,
    load_pickle,
    load_velodyne,
    non_dontcare_gt,
    read_csv_rows,
)
from week4_baseline_stratification import display_bin_label
from week5_intervention_harness import (
    ASSOC_IOU,
    F0_IOU_MIN,
    F0_SCORE_TOL,
    FRACTIONS,
    SEED,
    apply_delete,
    associate_to_original,
    bool_text,
    box_from_row,
    choose_delete_indices,
    cloud_keeps_outside,
    confirm_locked_artifacts,
    confirm_week3_counts,
    file_stamp,
    index_infos,
    interior_index,
    num_to_remove,
    score_text,
    stamp_tree,
    velodyne_bin_paths,
)
from week6_pilot import (
    EVAL_BATCH_SIZE,
    EXPECTED_ELIGIBLE,
    EXPECTED_ELIGIBLE_MATCHED,
    EXPECTED_NEAR_AT_LEAST_100,
    EXPECTED_NEAR_UNDER_20,
    EXPECTED_RANGE_COUNTS,
    MIN_POINTS,
    PilotStop,
    RANGE_LABELS,
    SCORE_FLOOR,
    SURVIVAL_SCORE,
    annotate_row,
    check_f0_forward,
    confirm_week5_smoke,
    forward_in_eval_batch,
    load_runtime,
    require_equal,
)

CONDITIONS = ("interior", "placebo", "context")
PLACEBO_MAX_M = 40.0
RETAINED_TOL = 2.0e-6
# A long process has crashed once. Exit cleanly so the shell can resume.
RESTART_AFTER_FORWARDS = 20000
FRACTION_LABELS = tuple("%.2f" % fraction for fraction in FRACTIONS)
POINT_LABELS = tuple(display_bin_label("num_points", index) for index in range(3))
SIZE_LABELS = tuple(display_bin_label("size_m3", index) for index in range(3))
EXPECTED_FORWARDS_PER_TARGET = 1 + 4 * 3

RESULT_COLUMNS = [
    "frame_id",
    "pred_idx",
    "range_bin",
    "num_points_bin",
    "size_bin",
    "condition",
    "fraction",
    "score_before",
    "num_points_before",
    "region_points",
    "num_removed",
    "placebo_status",
    "best_iou",
    "best_score",
    "survived_iou05",
    "score_after",
    "retained",
    "survived_score05",
]

TABLE_COLUMNS = [
    "condition",
    "stratum_type",
    "stratum",
    "fraction",
    "n",
    "mean_region_points",
    "mean_num_removed",
    "mean_retained",
    "score_survival_hits",
    "score_survival",
    "iou_survival_hits",
    "iou_survival",
]

IDENTITY_COLUMNS = [
    "frame_id",
    "pred_idx",
    "range_bin",
    "num_points_bin",
    "size_bin",
    "score",
    "num_points_in_box",
    "placebo_status",
    "placebo_points",
    "placebo_x",
    "placebo_y",
    "placebo_z",
    "placebo_l",
    "placebo_w",
    "placebo_h",
    "placebo_yaw",
]

OPENPCDET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    OPENPCDET_ROOT
    / "output/kitti_models/pointpillar/default/eval/epoch_7728/val/default/result.pkl"
)
DEFAULT_INFOS = OPENPCDET_ROOT / "data/kitti/kitti_infos_val.pkl"
DEFAULT_WEEK3 = OPENPCDET_ROOT / "output/week3/per_detection_car.csv"
DEFAULT_WALKED = OPENPCDET_ROOT / "output/week3/walked_frames.txt"
DEFAULT_WEEK4 = OPENPCDET_ROOT / "output/week4/week4_summary.txt"
DEFAULT_WEEK5 = OPENPCDET_ROOT / "output/week5"
DEFAULT_VELODYNE = OPENPCDET_ROOT / "data/kitti/training/velodyne"
DEFAULT_CKPT = OPENPCDET_ROOT / "checkpoints/pointpillar_7728.pth"
DEFAULT_CFG = OPENPCDET_ROOT / "tools/cfgs/kitti_models/pointpillar.yaml"
DEFAULT_DATA_ROOT = OPENPCDET_ROOT / "data/kitti"
DEFAULT_OUTPUT = OPENPCDET_ROOT / "output/week7"
DEFAULT_COPY = (
    OPENPCDET_ROOT.parent
    / "DIS Notes and Outputs"
    / "outputs by week"
    / "week7-outputs"
)

COPY_NAMES = (
    "week7_results.csv",
    "week7_sidecar.txt",
    "week7_summary.txt",
    "week7_table.csv",
)


class Week7Stop(RuntimeError):
    """A locked check failed. Finished rows already on disk are kept."""


class RestartChunk(Exception):
    """Enough new forwards for this process. The CSV is flushed."""


def parse_args():
    """Parse CLI paths. Defaults are inside this repository."""
    parser = argparse.ArgumentParser(description="Week 7 stratified retention run")
    parser.add_argument("--result-pkl", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--infos-pkl", type=Path, default=DEFAULT_INFOS)
    parser.add_argument("--week3-csv", type=Path, default=DEFAULT_WEEK3)
    parser.add_argument("--walked-txt", type=Path, default=DEFAULT_WALKED)
    parser.add_argument("--week4-summary", type=Path, default=DEFAULT_WEEK4)
    parser.add_argument("--week5-dir", type=Path, default=DEFAULT_WEEK5)
    parser.add_argument("--velodyne-dir", type=Path, default=DEFAULT_VELODYNE)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--cfg-file", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--copy-dir", type=Path, default=DEFAULT_COPY)
    parser.add_argument(
        "--draw-only",
        action="store_true",
        help="Write the sidecar and stop before any forward",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Preflight only. Refuses the official output directory.",
    )
    return parser.parse_args()


def atomic_write(path, text):
    """Replace path in one step so a crash cannot leave a partial sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def condition_draw_seed(condition, frame_id, pred_idx, fraction):
    """32-bit seed for one condition cell. Independent of visit order."""
    token = "%d|%s|%s|%d|%.2f" % (SEED, condition, frame_id, int(pred_idx), float(fraction))
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def self_check_counts_and_shifts():
    """Fail fast if half-even rounding or the placebo shift order drifted."""
    if num_to_remove(1, 0.5) != 0 or num_to_remove(3, 0.5) != 2:
        raise RuntimeError("Python round half-even is not the locked num_removed rule")
    box = np.array([0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0], dtype=np.float32)
    shifts = list(iter_placebo_shifts(box))
    if len(shifts) != 80:
        raise RuntimeError("placebo shift count is %d, expected 80" % len(shifts))
    if abs(float(shifts[0][1]) - 2.0) > 1.0e-5 or abs(float(shifts[0][0])) > 1.0e-5:
        raise RuntimeError("first placebo shift is not one width along lateral")
    if abs(float(shifts[2][0]) - 2.0) > 1.0e-5 or abs(float(shifts[2][1])) > 1.0e-5:
        raise RuntimeError("third placebo shift is not one width along longitudinal")
    obstacles = box.reshape(1, 7)
    if not boxes_overlap_3d(box, obstacles):
        raise RuntimeError("a box must overlap itself")


def iter_placebo_shifts(box):
    """Yield shifted copies of box in the locked order, out to 40 m.

    Step length is the box width. At each step, lateral comes before
    longitudinal, and the positive axis comes before the negative axis.
    """
    step = float(box[4])
    if step <= 0.0:
        return
    yaw = float(box[6])
    axes = (
        (-math.sin(yaw), math.cos(yaw)),
        (math.cos(yaw), math.sin(yaw)),
    )
    max_k = int(math.floor((PLACEBO_MAX_M + 1.0e-9) / step))
    for k in range(1, max_k + 1):
        distance = k * step
        for unit in axes:
            for sign in (1.0, -1.0):
                shifted = np.array(box, dtype=np.float32, copy=True)
                shifted[0] = float(box[0]) + sign * distance * unit[0]
                shifted[1] = float(box[1]) + sign * distance * unit[1]
                yield shifted


def boxes_overlap_3d(candidate, obstacles):
    """True when candidate has positive 3D overlap with any obstacle box."""
    if obstacles.shape[0] == 0:
        return False
    bev = boxes_bev_iou_cpu(
        np.ascontiguousarray(candidate.reshape(1, 7), dtype=np.float32),
        np.ascontiguousarray(obstacles, dtype=np.float32),
    )
    bev = np.asarray(bev, dtype=np.float64).reshape(-1)
    cand_z = float(candidate[2])
    cand_h = float(candidate[5])
    height = np.minimum(cand_z + cand_h / 2.0, obstacles[:, 2] + obstacles[:, 5] / 2.0)
    height = height - np.maximum(cand_z - cand_h / 2.0, obstacles[:, 2] - obstacles[:, 5] / 2.0)
    return bool(np.any((bev > 0.0) & (height > 0.0)))


def region_hits_obstacle(points_xyz, region_idx, obstacles):
    """True when any point of the region also lies in an obstacle box."""
    if region_idx.size == 0 or obstacles.shape[0] == 0:
        return False
    counts = count_points_in_boxes(points_xyz[region_idx], obstacles)
    return bool(int(np.sum(counts)) > 0)


def frame_obstacle_boxes(frame_rows, gt_boxes):
    """Stack the frame's predicted Car boxes and its GT Car boxes."""
    boxes = [box_from_row(row) for row in frame_rows]
    if gt_boxes is not None and len(gt_boxes):
        gt_arr = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 7)
        for gt_box in gt_arr:
            boxes.append(gt_box)
    if not boxes:
        return np.zeros((0, 7), dtype=np.float32)
    return np.stack(boxes, axis=0).astype(np.float32)


def find_placebo(points_xyz, box, obstacles):
    """Return (shifted box, point count) or (None, None) when none qualifies."""
    for shifted in iter_placebo_shifts(box):
        if boxes_overlap_3d(shifted, obstacles):
            continue
        region_idx = interior_index(points_xyz, shifted)
        if region_hits_obstacle(points_xyz, region_idx, obstacles):
            continue
        return shifted, int(region_idx.size)
    return None, None


def cached_cloud(cache, velodyne_dir, frame_id):
    """Keep only the current frame. The cached array is an owned copy."""
    if frame_id not in cache:
        cache.clear()
        loaded = load_velodyne(velodyne_dir, frame_id)
        cache[frame_id] = np.array(loaded, dtype=np.float32, copy=True, order="C")
    return cache[frame_id]


def eligible_targets(rows):
    """Every score-and-point eligible row, ordered by frame_id then pred_idx.

    Locked week-6 counts are checked here. Placebo boxes are not searched yet.
    """
    eligible = [
        annotate_row(row)
        for row in rows
        if float(row["score"]) >= SCORE_FLOOR and int(row["num_points_in_box"]) >= MIN_POINTS
    ]
    eligible.sort(key=lambda row: (row["frame_id"], int(row["pred_idx"])))
    seen = set()
    for row in eligible:
        key = (row["frame_id"], int(row["pred_idx"]))
        if key in seen:
            raise ValueError("duplicate eligible identity %s pred %s" % key)
        seen.add(key)
    require_equal("eligible rows", len(eligible), EXPECTED_ELIGIBLE)
    range_counts = []
    for label in RANGE_LABELS:
        range_counts.append(sum(1 for row in eligible if row["range_bin"] == label))
    for index, label in enumerate(RANGE_LABELS):
        require_equal("eligible %s" % label, range_counts[index], EXPECTED_RANGE_COUNTS[index])
    matched = sum(1 for row in eligible if row["gt_matched"])
    require_equal("eligible gt_matched", matched, EXPECTED_ELIGIBLE_MATCHED)
    near = [row for row in eligible if row["range_bin"] == RANGE_LABELS[0]]
    near_dense = sum(1 for row in near if int(row["num_points_in_box"]) >= 100)
    near_sparse = sum(1 for row in near if int(row["num_points_in_box"]) < 20)
    require_equal("near eligible with >=100 points", near_dense, EXPECTED_NEAR_AT_LEAST_100)
    require_equal("near eligible with <20 points", near_sparse, EXPECTED_NEAR_UNDER_20)
    stats = {
        "eligible": len(eligible),
        "eligible_matched": matched,
        "range_counts": tuple(range_counts),
    }
    return eligible, stats


def attach_placebos(targets, rows, infos, velodyne_dir):
    """Search one empty box per target. Mutates each target with the placement."""
    by_frame = {}
    for row in rows:
        by_frame.setdefault(row["frame_id"], []).append(row)
    info_by_id = index_infos(infos)
    clouds = {}
    placed = 0
    started = time.perf_counter()
    for index, target in enumerate(targets, start=1):
        frame_id = target["frame_id"]
        points = cached_cloud(clouds, velodyne_dir, frame_id)
        gt_boxes = non_dontcare_gt(info_by_id[frame_id]["annos"])["boxes_lidar"]
        obstacles = frame_obstacle_boxes(by_frame[frame_id], gt_boxes)
        shifted, region_points = find_placebo(points[:, 0:3], box_from_row(target), obstacles)
        if shifted is None:
            target["placebo_status"] = "unavailable"
            target["placebo_box"] = None
            target["placebo_points"] = None
        else:
            target["placebo_status"] = "applied"
            target["placebo_box"] = shifted
            target["placebo_points"] = int(region_points)
            placed += 1
        if index % 500 == 0:
            print(
                "placed %d / %d  applied %d  %.0f s"
                % (index, len(targets), placed, time.perf_counter() - started),
                flush=True,
            )
    return placed, len(targets) - placed


def joint_census(targets):
    """Count targets in each of the 27 locked joint cells, including zeros."""
    counts = {}
    for range_label in RANGE_LABELS:
        for point_label in POINT_LABELS:
            for size_label in SIZE_LABELS:
                counts[(range_label, point_label, size_label)] = 0
    for target in targets:
        key = (target["range_bin"], target["num_points_bin"], target["size_bin"])
        if key not in counts:
            raise ValueError("target fell outside the locked bins: %s" % (key,))
        counts[key] += 1
    return counts


def joint_label(range_label, point_label, size_label):
    """One stratum name for a joint cell."""
    return "%s | %s | %s" % (range_label, point_label, size_label)


def format_box_field(box, index):
    """Six-decimal sidecar field. Empty when this target has no placebo box."""
    if box is None:
        return ""
    return "%.6f" % float(box[index])


def sidecar_text(targets, stats, max_targets):
    """Census, placebo rule, and one identity row per target."""
    lines = [
        "Week 7 sidecar",
        "seed: %d" % SEED,
        "fractions: %s" % ", ".join(FRACTION_LABELS),
        "conditions: %s" % ", ".join(CONDITIONS),
        "draw token: sha256(\"%d|<condition>|<frame_id>|<pred_idx>|<fraction>\")" % SEED,
        "num_removed: Python round(fraction * region_points), half even",
        "eval batch: %d. Only the target frame is edited." % EVAL_BATCH_SIZE,
        "association bar: %.2f" % ASSOC_IOU,
        "survival score threshold: %.2f" % SURVIVAL_SCORE,
        "eligible filter: score >= %.2f and num_points_in_box >= %d" % (SCORE_FLOOR, MIN_POINTS),
        "eligible rows: %d" % stats["eligible"],
        "eligible by range: %s"
        % ", ".join(
            "%s %d" % (label, stats["range_counts"][index])
            for index, label in enumerate(RANGE_LABELS)
        ),
        "eligible gt_matched: %d" % stats["eligible_matched"],
        "order: frame_id, then pred_idx",
        "placebo: same length, width, height, yaw, and z",
        "placebo step: box width, lateral then longitudinal, positive then negative",
        "placebo lateral: (-sin yaw, cos yaw)",
        "placebo longitudinal: (cos yaw, sin yaw)",
        "placebo max distance m: %.1f" % PLACEBO_MAX_M,
        "placebo applied: %d" % stats["placebo_applied"],
        "placebo unavailable: %d" % stats["placebo_unavailable"],
        "max_targets: %s" % ("none" if max_targets is None else str(int(max_targets))),
        "joint counts:",
    ]
    for range_label in RANGE_LABELS:
        for point_label in POINT_LABELS:
            for size_label in SIZE_LABELS:
                key = (range_label, point_label, size_label)
                lines.append(
                    "  %s n=%d" % (joint_label(range_label, point_label, size_label), stats["joint_counts"][key])
                )
    lines.append("identity_table:")
    lines.append(",".join(IDENTITY_COLUMNS))
    for target in targets:
        box = target["placebo_box"]
        points_text = "" if target["placebo_points"] is None else str(int(target["placebo_points"]))
        fields = [
            target["frame_id"],
            str(int(target["pred_idx"])),
            target["range_bin"],
            target["num_points_bin"],
            target["size_bin"],
            "%.6f" % float(target["score"]),
            str(int(target["num_points_in_box"])),
            target["placebo_status"],
            points_text,
            format_box_field(box, 0),
            format_box_field(box, 1),
            format_box_field(box, 2),
            format_box_field(box, 3),
            format_box_field(box, 4),
            format_box_field(box, 5),
            format_box_field(box, 6),
        ]
        if len(fields) != len(IDENTITY_COLUMNS):
            raise RuntimeError("identity row width does not match the header")
        lines.append(",".join(fields))
    lines.append("")
    return "\n".join(lines)


def identity_record(target):
    """Comparable identity tuple, including the formatted placebo box."""
    box = target["placebo_box"]
    points_text = "" if target["placebo_points"] is None else str(int(target["placebo_points"]))
    return (
        target["frame_id"],
        str(int(target["pred_idx"])),
        target["range_bin"],
        target["num_points_bin"],
        target["size_bin"],
        "%.6f" % float(target["score"]),
        str(int(target["num_points_in_box"])),
        target["placebo_status"],
        points_text,
        format_box_field(box, 0),
        format_box_field(box, 1),
        format_box_field(box, 2),
        format_box_field(box, 3),
        format_box_field(box, 4),
        format_box_field(box, 5),
        format_box_field(box, 6),
    )


def parse_sidecar_identities(path):
    """Return identity tuples in sidecar order."""
    text = Path(path).read_text(encoding="utf-8")
    marker = "identity_table:\n"
    if marker not in text:
        raise ValueError("sidecar has no identity_table: %s" % path)
    table = text.split(marker, 1)[1]
    reader = csv.DictReader(table.splitlines())
    if reader.fieldnames != IDENTITY_COLUMNS:
        raise ValueError("sidecar identity header does not match the locked columns")
    identities = []
    for raw in reader:
        identities.append(tuple(raw[column] for column in IDENTITY_COLUMNS))
    return identities


def repair_trailing_partial(path):
    """Drop a last line that was not finished with a newline."""
    path = Path(path)
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    head, separator, _tail = data.rpartition(b"\n")
    if separator:
        path.write_bytes(head + b"\n")
    else:
        path.write_bytes(b"")


def read_result_rows(path):
    """Load finished result rows. An empty or missing file returns no rows."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    repair_trailing_partial(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RESULT_COLUMNS:
            raise ValueError("results CSV header does not match the locked columns")
        rows = []
        for raw in reader:
            if None in raw or any(raw[column] is None for column in RESULT_COLUMNS):
                raise ValueError("results CSV has a row with the wrong width")
            rows.append(raw)
    return rows


def finished_keys(rows):
    """Set of (frame_id, pred_idx, condition, fraction) already stored."""
    keys = set()
    for row in rows:
        key = (row["frame_id"], int(row["pred_idx"]), row["condition"], row["fraction"])
        if key in keys:
            raise ValueError(
                "duplicate result %s pred %s %s %s" % (key[0], key[1], key[2], key[3])
            )
        keys.add(key)
    return keys


def open_writer(path, fieldnames, already_has_rows):
    """Append rows. Write the header only when the file is still empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    if not already_has_rows:
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())
    return handle, writer


def append_row(handle, writer, row):
    """Write one finished cell and flush it before the next forward."""
    writer.writerow(row)
    handle.flush()
    os.fsync(handle.fileno())


def outside_indices(num_points, inside):
    """Indices of points that are not in the target interior."""
    all_idx = np.arange(int(num_points), dtype=np.int64)
    return np.setdiff1d(all_idx, np.asarray(inside, dtype=np.int64), assume_unique=True)


def target_interior(points, target):
    """Interior indices. Stop when the count disagrees with the week-3 CSV."""
    inside = interior_index(points[:, 0:3], box_from_row(target))
    csv_points = int(target["num_points_in_box"])
    if int(inside.size) != csv_points:
        raise Week7Stop(
            "%s pred %d interior %d != week-3 %d"
            % (target["frame_id"], int(target["pred_idx"]), int(inside.size), csv_points)
        )
    return inside


def edit_condition(points, target, condition, fraction, inside, obstacles):
    """Delete one fraction for one condition. The input cloud is not modified.

    Returns the edited cloud, the region size, and the removed count.
    """
    prefix = "%s pred %d %s f=%.2f" % (
        target["frame_id"],
        int(target["pred_idx"]),
        condition,
        float(fraction),
    )
    xyz = points[:, 0:3]
    if condition == "interior":
        region_idx = inside
    elif condition == "context":
        region_idx = outside_indices(points.shape[0], inside)
    elif condition == "placebo":
        if target["placebo_status"] != "applied":
            raise Week7Stop("%s placebo edit requested without a placement" % prefix)
        region_idx = interior_index(xyz, target["placebo_box"])
        if int(region_idx.size) != int(target["placebo_points"]):
            raise Week7Stop(
                "%s placebo points %d != placement %d"
                % (prefix, int(region_idx.size), int(target["placebo_points"]))
            )
        if region_hits_obstacle(xyz, region_idx, obstacles):
            raise Week7Stop("%s placebo region intersects an object" % prefix)
    else:
        raise Week7Stop("%s unknown condition" % prefix)

    region_points = int(region_idx.size)
    n_removed = num_to_remove(region_points, fraction)
    delete_idx = choose_delete_indices(
        region_idx,
        n_removed,
        condition_draw_seed(condition, target["frame_id"], target["pred_idx"], fraction),
    )
    modified = apply_delete(points, delete_idx)
    check_edit(points, target, condition, fraction, inside, region_idx, delete_idx, modified, obstacles)
    return modified, region_points, n_removed


def check_edit(points, target, condition, fraction, inside, region_idx, delete_idx, modified, obstacles):
    """Stop when a delete does not match the condition's accounting."""
    prefix = "%s pred %d %s f=%.2f" % (
        target["frame_id"],
        int(target["pred_idx"]),
        condition,
        float(fraction),
    )
    if delete_idx.size and not bool(np.all(np.isin(delete_idx, region_idx))):
        raise Week7Stop("%s deleted a point outside its region" % prefix)
    if not cloud_keeps_outside(points, modified, delete_idx, region_idx):
        raise Week7Stop("%s kept rows do not match the delete" % prefix)
    if float(fraction) == 0.0:
        if delete_idx.size != 0 or not np.array_equal(modified, points):
            raise Week7Stop("%s f=0 cloud was modified" % prefix)
    if condition == "interior" and float(fraction) == 1.0:
        if int(inside.size) <= 0:
            raise Week7Stop("%s f=1 target is empty" % prefix)
        after = int(count_points_in_boxes(modified[:, 0:3], box_from_row(target).reshape(1, 7))[0])
        if after != 0:
            raise Week7Stop("%s f=1 left interior points" % prefix)
        if not cloud_keeps_outside(points, modified, delete_idx, inside):
            raise Week7Stop("%s f=1 changed points outside the box" % prefix)
    if condition == "context" and float(fraction) == 1.0:
        if not np.array_equal(modified, points[inside]):
            raise Week7Stop("%s f=1 did not leave exactly the interior points" % prefix)
    if condition == "placebo":
        if region_hits_obstacle(points[:, 0:3], delete_idx, obstacles):
            raise Week7Stop("%s deleted a point that lies in an object box" % prefix)
        if float(fraction) == 1.0 and int(region_idx.size) > 0:
            after = int(count_points_in_boxes(modified[:, 0:3], target["placebo_box"].reshape(1, 7))[0])
            if after != 0:
                raise Week7Stop("%s f=1 left points in the placebo box" % prefix)


def blank_metric_row(target, condition, fraction_label, status):
    """One CSV row with no forward. Used for an unavailable placebo."""
    return {
        "frame_id": target["frame_id"],
        "pred_idx": str(int(target["pred_idx"])),
        "range_bin": target["range_bin"],
        "num_points_bin": target["num_points_bin"],
        "size_bin": target["size_bin"],
        "condition": condition,
        "fraction": fraction_label,
        "score_before": "%.6f" % float(target["score"]),
        "num_points_before": str(int(target["num_points_in_box"])),
        "region_points": "",
        "num_removed": "",
        "placebo_status": status,
        "best_iou": "",
        "best_score": "",
        "survived_iou05": "",
        "score_after": "",
        "retained": "",
        "survived_score05": "",
    }


def metric_row(target, condition, fraction_label, region_points, n_removed, status, best_score, best_iou, survived):
    """One applied CSV row. A miss stores retained 0 and an empty score_after."""
    if survived and best_score is None:
        raise Week7Stop(
            "%s pred %d %s f=%s survived with no score"
            % (target["frame_id"], int(target["pred_idx"]), condition, fraction_label)
        )
    score_before = float(target["score"])
    if score_before <= 0.0:
        raise Week7Stop(
            "%s pred %d score_before is not positive" % (target["frame_id"], int(target["pred_idx"]))
        )
    if survived:
        score_after_text = score_text(best_score)
        stored_score = float(score_after_text)
        retained = stored_score / score_before
        survived_score = stored_score >= SURVIVAL_SCORE
    else:
        score_after_text = ""
        retained = 0.0
        survived_score = False
    return {
        "frame_id": target["frame_id"],
        "pred_idx": str(int(target["pred_idx"])),
        "range_bin": target["range_bin"],
        "num_points_bin": target["num_points_bin"],
        "size_bin": target["size_bin"],
        "condition": condition,
        "fraction": fraction_label,
        "score_before": "%.6f" % score_before,
        "num_points_before": str(int(target["num_points_in_box"])),
        "region_points": str(int(region_points)),
        "num_removed": str(int(n_removed)),
        "placebo_status": status,
        "best_iou": "%.6f" % float(best_iou),
        "best_score": score_text(best_score),
        "survived_iou05": bool_text(survived),
        "score_after": score_after_text,
        "retained": "%.6f" % retained,
        "survived_score05": bool_text(survived_score),
    }


def status_for(condition, target):
    """placebo_status column for a condition that is about to be written."""
    if condition == "placebo":
        return target["placebo_status"]
    return "n/a"


def region_size(points, target, condition, inside):
    """How many points this condition is allowed to delete from."""
    if condition == "interior":
        return int(inside.size)
    if condition == "context":
        return int(points.shape[0] - inside.size)
    if target["placebo_status"] != "applied":
        raise Week7Stop(
            "%s pred %d placebo region requested without a placement"
            % (target["frame_id"], int(target["pred_idx"]))
        )
    return int(target["placebo_points"])


def target_has_pending(target, done):
    """True when any condition and fraction for this target is not in the CSV."""
    frame_id = target["frame_id"]
    pred_idx = int(target["pred_idx"])
    for condition in CONDITIONS:
        for fraction_label in FRACTION_LABELS:
            if (frame_id, pred_idx, condition, fraction_label) not in done:
                return True
    return False


def run_forwards(args, targets, results_path, done):
    """Forward unfinished cells. Fraction 0 is shared by the applied conditions."""
    dataset, model, info_by_id, index_by_frame = load_runtime(args)
    by_frame = {}
    for target in targets:
        by_frame.setdefault(target["frame_id"], []).append(target)
    # Obstacle boxes use every week-3 Car in the frame, not only eligible ones.
    all_rows = read_csv_rows(args.week3_csv)
    preds_by_frame = {}
    for row in all_rows:
        preds_by_frame.setdefault(row["frame_id"], []).append(row)
    clouds = {}
    calib_by_frame = {}
    obstacle_by_frame = {}
    header_present = results_path.exists() and results_path.stat().st_size > 0
    handle, writer = open_writer(results_path, RESULT_COLUMNS, already_has_rows=header_present)
    started = time.perf_counter()
    written = 0
    forwarded = 0
    try:
        for target in targets:
            if not target_has_pending(target, done):
                continue
            frame_id = target["frame_id"]
            points = cached_cloud(clouds, args.velodyne_dir, frame_id)
            inside = target_interior(points, target)
            if frame_id not in obstacle_by_frame:
                gt_boxes = non_dontcare_gt(info_by_id[frame_id]["annos"])["boxes_lidar"]
                obstacle_by_frame[frame_id] = frame_obstacle_boxes(preds_by_frame[frame_id], gt_boxes)
            obstacles = obstacle_by_frame[frame_id]
            if frame_id not in calib_by_frame:
                calib_by_frame[frame_id] = dataset.get_calib(info_by_id[frame_id]["point_cloud"]["lidar_idx"])
            calib = calib_by_frame[frame_id]
            for fraction_label in FRACTION_LABELS:
                key = (frame_id, int(target["pred_idx"]), "placebo", fraction_label)
                if target["placebo_status"] == "unavailable" and key not in done:
                    append_row(
                        handle,
                        writer,
                        blank_metric_row(target, "placebo", fraction_label, "unavailable"),
                    )
                    done.add(key)
                    written += 1
            f0_conditions = []
            for condition in CONDITIONS:
                key = (frame_id, int(target["pred_idx"]), condition, FRACTION_LABELS[0])
                if key in done:
                    continue
                if condition == "placebo" and target["placebo_status"] != "applied":
                    continue
                f0_conditions.append(condition)
            if f0_conditions:
                pred_dict = forward_in_eval_batch(
                    dataset, model, info_by_id, index_by_frame, frame_id, points
                )
                forwarded += 1
                best_score, best_iou, survived = associate_to_original(
                    pred_dict, box_from_row(target), calib
                )
                check_f0_forward(target, best_score, best_iou, survived)
                for condition in f0_conditions:
                    region_points = region_size(points, target, condition, inside)
                    if num_to_remove(region_points, 0.0) != 0:
                        raise Week7Stop(
                            "%s pred %d %s f=0 removed points"
                            % (frame_id, int(target["pred_idx"]), condition)
                        )
                    row = metric_row(
                        target,
                        condition,
                        FRACTION_LABELS[0],
                        region_points,
                        0,
                        status_for(condition, target),
                        best_score,
                        best_iou,
                        survived,
                    )
                    append_row(handle, writer, row)
                    done.add((frame_id, int(target["pred_idx"]), condition, FRACTION_LABELS[0]))
                    written += 1
                log_progress(started, forwarded, written, len(targets))
            for fraction, fraction_label in zip(FRACTIONS, FRACTION_LABELS):
                if float(fraction) == 0.0:
                    continue
                for condition in CONDITIONS:
                    key = (frame_id, int(target["pred_idx"]), condition, fraction_label)
                    if key in done:
                        continue
                    if condition == "placebo" and target["placebo_status"] != "applied":
                        continue
                    modified, region_points, n_removed = edit_condition(
                        points, target, condition, fraction, inside, obstacles
                    )
                    pred_dict = forward_in_eval_batch(
                        dataset, model, info_by_id, index_by_frame, frame_id, modified
                    )
                    forwarded += 1
                    best_score, best_iou, survived = associate_to_original(
                        pred_dict, box_from_row(target), calib
                    )
                    row = metric_row(
                        target,
                        condition,
                        fraction_label,
                        region_points,
                        n_removed,
                        status_for(condition, target),
                        best_score,
                        best_iou,
                        survived,
                    )
                    append_row(handle, writer, row)
                    done.add(key)
                    written += 1
                    if forwarded % 500 == 0:
                        torch.cuda.empty_cache()
                    log_progress(started, forwarded, written, len(targets))
            if forwarded >= RESTART_AFTER_FORWARDS:
                raise RestartChunk("forwarded %d new cells" % forwarded)
    finally:
        handle.flush()
        handle.close()
    return forwarded


def log_progress(started, forwarded, written, n_targets):
    """Print a pace line every 50 forwards."""
    if forwarded == 0 or forwarded % 50 != 0:
        return
    elapsed = time.perf_counter() - started
    rate = forwarded / elapsed
    print(
        "forwards %d  rows_written %d  targets %d  %.2f s/forward  elapsed %.0f s"
        % (forwarded, written, n_targets, 1.0 / rate, elapsed),
        flush=True,
    )


def applied_rows(rows):
    """Rows that received a forward. Unavailable placebo rows are excluded."""
    kept = []
    for row in rows:
        if row["condition"] == "placebo" and row["placebo_status"] == "unavailable":
            continue
        if row["placebo_status"] not in ("applied", "n/a"):
            raise ValueError(
                "bad placebo_status %s at %s" % (row["placebo_status"], row["frame_id"])
            )
        if row["condition"] != "placebo" and row["placebo_status"] != "n/a":
            raise ValueError("non-placebo row has placebo_status %s" % row["placebo_status"])
        if row["condition"] == "placebo" and row["placebo_status"] != "applied":
            raise ValueError("placebo mean row is not applied")
        kept.append(row)
    return kept


def audit_csv_rules(rows, targets):
    """Check stored fields. Does not reload clouds."""
    failures = []
    expected = {}
    for target in targets:
        expected[(target["frame_id"], int(target["pred_idx"]))] = target
    if len(rows) != len(targets) * len(CONDITIONS) * len(FRACTION_LABELS):
        failures.append(
            "row count %d, expected %d"
            % (len(rows), len(targets) * len(CONDITIONS) * len(FRACTION_LABELS))
        )
    seen = set()
    for row in rows:
        key = (row["frame_id"], int(row["pred_idx"]), row["condition"], row["fraction"])
        if key in seen:
            failures.append("duplicate %s" % (key,))
        seen.add(key)
        target = expected.get((row["frame_id"], int(row["pred_idx"])))
        if target is None:
            failures.append("row not in the sidecar draw: %s pred %s" % (row["frame_id"], row["pred_idx"]))
            continue
        if row["condition"] not in CONDITIONS:
            failures.append("unknown condition %s" % row["condition"])
            continue
        if row["fraction"] not in FRACTION_LABELS:
            failures.append("unknown fraction %s" % row["fraction"])
            continue
        if abs(float(row["score_before"]) - float(target["score"])) > 1.0e-6:
            failures.append("score_before drifted at %s pred %s" % (row["frame_id"], row["pred_idx"]))
        if int(row["num_points_before"]) != int(target["num_points_in_box"]):
            failures.append("num_points_before drifted at %s pred %s" % (row["frame_id"], row["pred_idx"]))
        if row["condition"] == "placebo" and target["placebo_status"] == "unavailable":
            if row["placebo_status"] != "unavailable" or row["retained"] != "":
                failures.append(
                    "unavailable placebo row is not blank at %s pred %s f=%s"
                    % (row["frame_id"], row["pred_idx"], row["fraction"])
                )
            continue
        if row["retained"] == "" or row["region_points"] == "" or row["num_removed"] == "":
            failures.append(
                "applied row has an empty metric at %s pred %s %s f=%s"
                % (row["frame_id"], row["pred_idx"], row["condition"], row["fraction"])
            )
            continue
        fraction = float(row["fraction"])
        region_points = int(row["region_points"])
        if int(row["num_removed"]) != num_to_remove(region_points, fraction):
            failures.append(
                "num_removed does not match round at %s pred %s %s f=%s"
                % (row["frame_id"], row["pred_idx"], row["condition"], row["fraction"])
            )
        if row["condition"] == "interior" and region_points != int(row["num_points_before"]):
            failures.append(
                "interior region_points != num_points_before at %s pred %s"
                % (row["frame_id"], row["pred_idx"])
            )
        survived = row["survived_iou05"] == "true"
        if not survived:
            if row["retained"] != "0.000000" or row["score_after"] != "":
                failures.append(
                    "miss is not stored as retained 0 at %s pred %s %s f=%s"
                    % (row["frame_id"], row["pred_idx"], row["condition"], row["fraction"])
                )
        else:
            if row["score_after"] == "":
                failures.append("survived row missing score_after")
            else:
                ratio = float(row["score_after"]) / float(row["score_before"])
                if abs(float(row["retained"]) - ratio) > RETAINED_TOL:
                    failures.append(
                        "retained does not match score_after / score_before at %s"
                        % row["frame_id"]
                    )
            score_flag = float(row["score_after"]) >= SURVIVAL_SCORE
            if (row["survived_score05"] == "true") != score_flag:
                failures.append("survived_score05 disagrees with score_after")
        if fraction == 0.0:
            if int(row["num_removed"]) != 0:
                failures.append("f=0 removed points at %s" % row["frame_id"])
            if row["survived_iou05"] != "true" or float(row["best_iou"]) < F0_IOU_MIN:
                failures.append("f=0 iou below %.2f at %s" % (F0_IOU_MIN, row["frame_id"]))
            delta = abs(float(row["score_after"]) - float(row["score_before"]))
            if delta >= F0_SCORE_TOL:
                failures.append("f=0 score delta %.4f at %s" % (delta, row["frame_id"]))
    for target in targets:
        for condition in CONDITIONS:
            for fraction_label in FRACTION_LABELS:
                key = (target["frame_id"], int(target["pred_idx"]), condition, fraction_label)
                if key not in seen:
                    failures.append("missing cell %s" % (key,))
    return failures


def audit_f1_edits(targets, velodyne_dir, info_by_id, preds_by_frame):
    """Recompute fraction-1 deletes from the original clouds."""
    failures = []
    clouds = {}
    grouped = {}
    for target in targets:
        grouped.setdefault(target["frame_id"], []).append(target)
    for frame_id, frame_targets in grouped.items():
        points = cached_cloud(clouds, velodyne_dir, frame_id)
        gt_boxes = non_dontcare_gt(info_by_id[frame_id]["annos"])["boxes_lidar"]
        obstacles = frame_obstacle_boxes(preds_by_frame[frame_id], gt_boxes)
        for target in frame_targets:
            try:
                inside = target_interior(points, target)
                for condition in CONDITIONS:
                    if condition == "placebo" and target["placebo_status"] != "applied":
                        continue
                    edit_condition(points, target, condition, 1.0, inside, obstacles)
            except (Week7Stop, PilotStop) as exc:
                failures.append(str(exc))
    return failures


def file_change_note(before_bins, after_bins, before_result, after_result):
    """Return (note, failures). Failures is empty when the inputs are unchanged."""
    bins_ok = before_bins == after_bins
    result_ok = before_result == after_result
    if bins_ok and result_ok:
        return (
            "training/velodyne size and mtime unchanged (%d files); result.pkl unchanged"
            % len(before_bins),
            [],
        )
    failures = []
    if not bins_ok:
        failures.append("training/velodyne size or mtime changed")
    if not result_ok:
        failures.append("result.pkl size or mtime changed")
    return "input files changed", failures


def slice_rows(rows, condition, fraction_label, range_bin=None, point_bin=None, size_bin=None):
    """Applied rows for one condition, fraction, and optional bin filters."""
    chosen = []
    for row in rows:
        if row["condition"] != condition or row["fraction"] != fraction_label:
            continue
        if range_bin is not None and row["range_bin"] != range_bin:
            continue
        if point_bin is not None and row["num_points_bin"] != point_bin:
            continue
        if size_bin is not None and row["size_bin"] != size_bin:
            continue
        chosen.append(row)
    return chosen


def mean_number(rows, field):
    """Mean of one numeric column. Empty input is refused."""
    if not rows:
        raise ValueError("cannot average an empty slice")
    return sum(float(row[field]) for row in rows) / float(len(rows))


def flag_rate(rows, field):
    """Fraction of rows whose flag column is the string true."""
    if not rows:
        raise ValueError("cannot rate an empty slice")
    hits = sum(1 for row in rows if row[field] == "true")
    return hits, hits / float(len(rows))


def table_row(condition, stratum_type, stratum, fraction_label, group):
    """One readable table row. n = 0 leaves the rates empty."""
    if not group:
        return {
            "condition": condition,
            "stratum_type": stratum_type,
            "stratum": stratum,
            "fraction": fraction_label,
            "n": "0",
            "mean_region_points": "",
            "mean_num_removed": "",
            "mean_retained": "",
            "score_survival_hits": "",
            "score_survival": "",
            "iou_survival_hits": "",
            "iou_survival": "",
        }
    score_hits, score_rate = flag_rate(group, "survived_score05")
    iou_hits, iou_rate = flag_rate(group, "survived_iou05")
    return {
        "condition": condition,
        "stratum_type": stratum_type,
        "stratum": stratum,
        "fraction": fraction_label,
        "n": str(len(group)),
        "mean_region_points": "%.6f" % mean_number(group, "region_points"),
        "mean_num_removed": "%.6f" % mean_number(group, "num_removed"),
        "mean_retained": "%.6f" % mean_number(group, "retained"),
        "score_survival_hits": "%d/%d" % (score_hits, len(group)),
        "score_survival": "%.6f" % score_rate,
        "iou_survival_hits": "%d/%d" % (iou_hits, len(group)),
        "iou_survival": "%.6f" % iou_rate,
    }


def strata():
    """Marginal and joint stratum specs: (type, label, filters)."""
    items = []
    for label in RANGE_LABELS:
        items.append(("range", label, {"range_bin": label}))
    for label in POINT_LABELS:
        items.append(("points", label, {"point_bin": label}))
    for label in SIZE_LABELS:
        items.append(("size", label, {"size_bin": label}))
    for range_label in RANGE_LABELS:
        for point_label in POINT_LABELS:
            for size_label in SIZE_LABELS:
                items.append(
                    (
                        "joint",
                        joint_label(range_label, point_label, size_label),
                        {
                            "range_bin": range_label,
                            "point_bin": point_label,
                            "size_bin": size_label,
                        },
                    )
                )
    return items


def build_table_rows(applied):
    """Every condition, stratum, and fraction, including empty joint cells."""
    built = []
    for condition in CONDITIONS:
        for stratum_type, stratum, filters in strata():
            for fraction_label in FRACTION_LABELS:
                group = slice_rows(applied, condition, fraction_label, **filters)
                built.append(table_row(condition, stratum_type, stratum, fraction_label, group))
    return built


def write_table(path, table_rows):
    """Write the readable results table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)


def marginal_lines(applied):
    """Summary blocks for the three marginals. Joint rates stay in the table."""
    lines = []
    marginals = (
        ("range", RANGE_LABELS, "range_bin"),
        ("points", POINT_LABELS, "point_bin"),
        ("size", SIZE_LABELS, "size_bin"),
    )
    for condition in CONDITIONS:
        for stratum_type, labels, filter_name in marginals:
            for label in labels:
                filters = {filter_name: label}
                base = slice_rows(applied, condition, FRACTION_LABELS[0], **filters)
                lines.append(
                    "%s | %s | %s  n=%d" % (condition, stratum_type, label, len(base))
                )
                if not base:
                    lines.append("  empty")
                    continue
                for fraction_label in FRACTION_LABELS:
                    group = slice_rows(applied, condition, fraction_label, **filters)
                    if len(group) != len(base):
                        raise ValueError(
                            "%s %s %s fraction %s has %d rows, expected %d"
                            % (condition, stratum_type, label, fraction_label, len(group), len(base))
                        )
                    score_hits, score_rate = flag_rate(group, "survived_score05")
                    iou_hits, iou_rate = flag_rate(group, "survived_iou05")
                    lines.append(
                        "  f=%s  mean_removed=%.3f  mean_retained=%.6f  "
                        "score_survival=%.6f (%d/%d)  iou_survival=%.6f (%d/%d)"
                        % (
                            fraction_label,
                            mean_number(group, "num_removed"),
                            mean_number(group, "retained"),
                            score_rate,
                            score_hits,
                            len(group),
                            iou_rate,
                            iou_hits,
                            len(group),
                        )
                    )
    return lines


def summary_text(stats, failures, applied, file_note, max_targets):
    """Correctness first. Rates are omitted when a check failed."""
    lines = [
        "Week 7 summary",
        "eligible: %d (score >= %.2f, num_points_in_box >= %d)"
        % (stats["eligible"], SCORE_FLOOR, MIN_POINTS),
        "eligible by range: %s"
        % ", ".join(
            "%s %d" % (label, stats["range_counts"][index])
            for index, label in enumerate(RANGE_LABELS)
        ),
        "eligible gt_matched: %d" % stats["eligible_matched"],
        "targets: %d" % stats["eligible"] if max_targets is None else "targets: %d (preflight)" % max_targets,
        "placebo applied: %d" % stats["placebo_applied"],
        "placebo unavailable: %d" % stats["placebo_unavailable"],
        "seed: %d" % SEED,
        "fractions: %s" % " ".join(FRACTION_LABELS),
        "conditions: %s" % " ".join(CONDITIONS),
        "association bar: %.2f" % ASSOC_IOU,
        "score survival threshold: %.2f" % SURVIVAL_SCORE,
        "eval batch size: %d" % EVAL_BATCH_SIZE,
        "definitions: locked by week 6",
        "correctness:",
        "  failures: %d" % len(failures),
        "  %s" % file_note,
    ]
    for failure in failures:
        lines.append("  %s" % failure)
    if failures:
        lines.append("decision: INVALID")
        lines.append("")
        return "\n".join(lines)
    lines.append("decision: correctness passed")
    lines.extend(marginal_lines(applied))
    lines.append("joint rates: week7_table.csv")
    lines.append("")
    return "\n".join(lines)


def write_stop_summary(output_dir, message):
    """Record an early stop without rate lines."""
    text = "\n".join(
        [
            "Week 7 summary",
            "decision: INVALID",
            "stopped: %s" % message,
            "",
        ]
    )
    atomic_write(Path(output_dir) / "week7_summary.txt", text)


def copy_outputs(output_dir, copy_dir):
    """Copy finished week-7 files into the notes folder. Missing files are skipped."""
    if copy_dir is None:
        return
    output_dir = Path(output_dir)
    copy_dir = Path(copy_dir)
    if not copy_dir.parent.exists():
        print("skip copy; parent missing: %s" % copy_dir.parent, flush=True)
        return
    copy_dir.mkdir(parents=True, exist_ok=True)
    for name in COPY_NAMES:
        source = output_dir / name
        if source.exists():
            shutil.copy2(str(source), str(copy_dir / name))


def refuse_preflight_on_official_dir(args):
    """A short run must not overwrite the official sidecar or CSV."""
    if args.max_targets is None:
        return
    if int(args.max_targets) <= 0:
        raise RuntimeError("max-targets must be positive")
    if Path(args.output_dir).resolve() == Path(DEFAULT_OUTPUT).resolve():
        raise RuntimeError("max-targets cannot write the official week-7 output directory")


def main():
    """Place the full eligible set, forward unfinished cells, then write the table."""
    args = parse_args()
    refuse_preflight_on_official_dir(args)
    self_check_counts_and_shifts()
    confirm_locked_artifacts(args)
    confirm_week5_smoke(args.week5_dir)
    rows = read_csv_rows(args.week3_csv)
    confirm_week3_counts(rows, args.walked_txt)
    infos = load_pickle(args.infos_pkl)
    targets, stats = eligible_targets(rows)
    if args.max_targets is not None:
        targets = targets[: int(args.max_targets)]
    print("placing placebo boxes for %d targets" % len(targets), flush=True)
    applied, unavailable = attach_placebos(targets, rows, infos, args.velodyne_dir)
    stats["joint_counts"] = joint_census(targets)
    stats["placebo_applied"] = applied
    stats["placebo_unavailable"] = unavailable
    print(
        "targets %d  placebo applied %d  unavailable %d"
        % (len(targets), applied, unavailable),
        flush=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = output_dir / "week7_sidecar.txt"
    results_path = output_dir / "week7_results.csv"
    fresh = [identity_record(target) for target in targets]
    if sidecar_path.exists():
        saved = parse_sidecar_identities(sidecar_path)
        if saved != fresh:
            raise ValueError("existing sidecar does not match the deterministic draw")
    atomic_write(sidecar_path, sidecar_text(targets, stats, args.max_targets))
    print("sidecar %s identities %d" % (sidecar_path, len(targets)), flush=True)
    if args.max_targets is None:
        copy_outputs(output_dir, args.copy_dir)
    if args.draw_only:
        print("draw-only stop", flush=True)
        return

    existing = read_result_rows(results_path)
    done = finished_keys(existing)
    print("finished cells %d" % len(done), flush=True)
    print("stamping velodyne and result.pkl", flush=True)
    before_bins = stamp_tree(velodyne_bin_paths(args.velodyne_dir))
    before_result = file_stamp(args.result_pkl)
    print("stamp done, %d velodyne files" % len(before_bins), flush=True)
    try:
        forwarded = run_forwards(args, targets, results_path, done)
        print("forwards this process %d" % forwarded, flush=True)
    except RestartChunk as exc:
        print("chunk stop: %s" % exc, flush=True)
        raise SystemExit(75)
    except (Week7Stop, PilotStop) as exc:
        write_stop_summary(output_dir, str(exc))
        if args.max_targets is None:
            copy_outputs(output_dir, args.copy_dir)
        print("stopped: %s" % exc, flush=True)
        raise SystemExit(1)

    finished = read_result_rows(results_path)
    info_by_id = index_infos(infos)
    preds_by_frame = {}
    for row in rows:
        preds_by_frame.setdefault(row["frame_id"], []).append(row)
    print("auditing fraction 1 deletes", flush=True)
    failures = []
    failures.extend(audit_csv_rules(finished, targets))
    failures.extend(audit_f1_edits(targets, args.velodyne_dir, info_by_id, preds_by_frame))
    after_bins = stamp_tree(velodyne_bin_paths(args.velodyne_dir))
    after_result = file_stamp(args.result_pkl)
    file_note, file_failures = file_change_note(before_bins, after_bins, before_result, after_result)
    failures.extend(file_failures)
    applied = applied_rows(finished) if not failures else []
    text = summary_text(stats, failures, applied, file_note, args.max_targets)
    atomic_write(output_dir / "week7_summary.txt", text)
    if not failures:
        write_table(output_dir / "week7_table.csv", build_table_rows(applied))
    if args.max_targets is None:
        copy_outputs(output_dir, args.copy_dir)
    print(text, flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
