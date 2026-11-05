"""Week-11 PointPillars run for Pedestrian and Cyclist.

The Car table and the week-9 PV-RCNN comparison stay as they are. This
driver builds one metadata table per class from the saved PointPillars
result.pkl, locks class-specific volume cuts, then repeats the week-7
edits. It does not train, does not edit Velodyne files, and does not
overwrite result.pkl.

Rerun the same command when the process exits 75. Finished cells are kept.

    python week11_per_class.py --draw-only
    python week11_per_class.py
    python week11_per_class.py --class-name Pedestrian
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
os.chdir(TOOLS_DIR)
sys.path.insert(0, str(TOOLS_DIR))

import _init_path  # noqa: F401

import torch

import week7_stratified as week7
from pcdet.datasets.kitti.kitti_object_eval_python.eval import (
    d3_box_overlap,
    get_official_eval_result,
)
from pcdet.utils import box_utils
from week3_per_detection_table import (
    DETECTOR_SCORE_THRESHOLD,
    MATCH_IOU_THRESH,
    camera_boxes_from_anno,
    count_points_in_boxes,
    horizontal_range_m,
    index_by_frame_id,
    load_pickle,
    load_val_ids,
    load_velodyne,
    match_preds_to_gt,
    read_csv_rows,
    write_csv,
)
from week4_baseline_stratification import assign_bins, size_m3
from week5_intervention_harness import (
    ASSOC_IOU,
    EXPECTED_FRAMES,
    F0_IOU_MIN,
    F0_SCORE_TOL,
    FRACTIONS,
    SEED,
    apply_delete,
    box_from_row,
    choose_delete_indices,
    confirm_locked_artifacts,
    confirm_week3_counts,
    file_stamp,
    interior_index,
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
    forward_in_eval_batch,
    load_runtime,
)

CLASSES = ("Pedestrian", "Cyclist")
PUBLISHED_MODERATE_R11 = {"Pedestrian": 52.29, "Cyclist": 62.68}
PUBLISHED_AP_TOLERANCE = 0.20
OFFICIAL_OVERLAP = 0.50
VOLUME_STEPS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
RESTART_AFTER_FORWARDS = 20000
FRACTION_LABELS = week7.FRACTION_LABELS
POINT_LABELS = week7.POINT_LABELS
CONDITIONS = week7.CONDITIONS

OPENPCDET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    OPENPCDET_ROOT
    / "output/kitti_models/pointpillar/default/eval/epoch_7728/val/default/result.pkl"
)
DEFAULT_INFOS = OPENPCDET_ROOT / "data/kitti/kitti_infos_val.pkl"
DEFAULT_VAL_TXT = OPENPCDET_ROOT / "data/kitti/ImageSets/val.txt"
DEFAULT_WEEK3 = OPENPCDET_ROOT / "output/week3/per_detection_car.csv"
DEFAULT_WALKED = OPENPCDET_ROOT / "output/week3/walked_frames.txt"
DEFAULT_WEEK4 = OPENPCDET_ROOT / "output/week4/week4_summary.txt"
DEFAULT_VELODYNE = OPENPCDET_ROOT / "data/kitti/training/velodyne"
DEFAULT_CKPT = OPENPCDET_ROOT / "checkpoints/pointpillar_7728.pth"
DEFAULT_CFG = OPENPCDET_ROOT / "tools/cfgs/kitti_models/pointpillar.yaml"
DEFAULT_DATA_ROOT = OPENPCDET_ROOT / "data/kitti"
DEFAULT_OUTPUT = OPENPCDET_ROOT / "output/week11"
DEFAULT_COPY = (
    OPENPCDET_ROOT.parent
    / "DIS Notes and Outputs"
    / "outputs by week"
    / "week11-outputs"
)
DEFINITIONS_NAME = "week11_definitions.txt"
STAMP_NAME = "week11_input_stamps.json"
AP_COLUMNS = (
    "r11_easy",
    "r11_moderate",
    "r11_hard",
    "r40_easy",
    "r40_moderate",
    "r40_hard",
)
COPY_NAMES = (
    DEFINITIONS_NAME,
    "week11_pedestrian_summary.txt",
    "week11_cyclist_summary.txt",
    "week11_pedestrian_table.csv",
    "week11_cyclist_table.csv",
    "week11_pedestrian_sample.csv",
    "week11_cyclist_sample.csv",
)


class Week11Stop(RuntimeError):
    """A locked check failed. Finished rows already on disk are kept."""


class RestartChunk(Exception):
    """Enough new forwards for this process. The CSV is flushed."""


def parse_args():
    """Parse CLI paths. Defaults point at the locked PointPillars artifacts."""
    parser = argparse.ArgumentParser(description="Week 11 Pedestrian and Cyclist run")
    parser.add_argument(
        "--class-name",
        choices=CLASSES + ("all",),
        default="all",
        help="One class, or all for Pedestrian then Cyclist.",
    )
    parser.add_argument("--result-pkl", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--infos-pkl", type=Path, default=DEFAULT_INFOS)
    parser.add_argument("--val-txt", type=Path, default=DEFAULT_VAL_TXT)
    parser.add_argument("--week3-csv", type=Path, default=DEFAULT_WEEK3)
    parser.add_argument("--walked-txt", type=Path, default=DEFAULT_WALKED)
    parser.add_argument("--week4-summary", type=Path, default=DEFAULT_WEEK4)
    parser.add_argument("--velodyne-dir", type=Path, default=DEFAULT_VELODYNE)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--cfg-file", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--copy-dir", type=Path, default=DEFAULT_COPY)
    parser.add_argument(
        "--draw-only",
        action="store_true",
        help="Write metadata, volume cuts, and the sidecar, then stop before any forward.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Prefix of the eligible list. Refuses the official output directory.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Check the volume rule and the draw seed, then exit.",
    )
    return parser.parse_args()


def slug(class_name):
    """Filename token for one KITTI class."""
    return class_name.lower()


def class_paths(output_dir, class_name):
    """Output paths for one class. The full CSVs stay in this directory."""
    root = Path(output_dir)
    token = slug(class_name)
    return {
        "metadata": root / ("week11_%s_per_detection.csv" % token),
        "walked": root / ("week11_%s_walked.txt" % token),
        "sample": root / ("week11_%s_sample.csv" % token),
        "eval": root / ("week11_%s_eval.txt" % token),
        "ap": root / ("week11_%s_ap.csv" % token),
        "sidecar": root / ("week11_%s_sidecar.txt" % token),
        "results": root / ("week11_%s_results.csv" % token),
        "preflight": root / ("week11_%s_preflight.txt" % token),
        "summary": root / ("week11_%s_summary.txt" % token),
        "table": root / ("week11_%s_table.csv" % token),
    }


def format_m3(value):
    """Two-decimal cubic-meter text. Cuts are multiples of 0.01."""
    return "%.2f" % float(value)


def snap_cut(value, step):
    """Round a cut onto the step, then onto two decimal places."""
    units = round(float(value) / float(step))
    return float("%.2f" % (units * float(step)))


def size_labels(low, high):
    """Three display labels for one class's volume cuts."""
    return (
        "under %s m3" % format_m3(low),
        "%s to under %s m3" % (format_m3(low), format_m3(high)),
        "%s m3 or larger" % format_m3(high),
    )


def assign_size_label(volume, low, high):
    """Bin a volume. The low edge belongs to the middle bin, matching week 4."""
    labels = size_labels(low, high)
    if float(volume) < float(low):
        return labels[0]
    if float(volume) < float(high):
        return labels[1]
    return labels[2]


def volume_bin_counts(volumes, low, high):
    """Count values in [under low), [low, high), [high, inf)."""
    values = np.asarray(list(volumes), dtype=np.float64)
    return (
        int(np.sum(values < float(low))),
        int(np.sum((values >= float(low)) & (values < float(high)))),
        int(np.sum(values >= float(high))),
    )


def choose_volume_cuts(volumes):
    """Pick the coarsest round cuts that leave three nonempty eligible bins.

    The one-third and two-thirds ranks of the eligible volumes are the
    starting points. Steps are tried from 1 m3 down to 0.01 m3. The first
    step whose rounded cuts all contain a box is kept.
    """
    values = np.sort(np.asarray(list(volumes), dtype=np.float64))
    count = int(values.size)
    if count < 3:
        raise Week11Stop(
            "eligible boxes are %d; three nonempty volume bins need at least 3" % count
        )
    q1 = float(values[count // 3])
    q2 = float(values[(2 * count) // 3])
    for step in VOLUME_STEPS:
        low = snap_cut(math.floor((q1 / step) + 1.0e-9) * step, step)
        high = snap_cut(math.ceil((q2 / step) - 1.0e-9) * step, step)
        if high <= low:
            high = snap_cut(low + step, step)
        counts = volume_bin_counts(values, low, high)
        if min(counts) > 0:
            return low, high, step, counts
    raise Week11Stop(
        "no round volume cut in %s splits these eligible volumes into three nonempty bins"
        % (VOLUME_STEPS,)
    )


def condition_draw_seed(class_name, condition, frame_id, pred_idx, fraction):
    """32-bit seed. The class name keeps Pedestrian and Cyclist from sharing a subset."""
    token = "%d|%s|%s|%s|%d|%.2f" % (
        SEED,
        class_name,
        condition,
        frame_id,
        int(pred_idx),
        float(fraction),
    )
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def self_check_protocol():
    """Fail before any file is written if the new locks drifted."""
    spread = choose_volume_cuts([0.2 + (i * 0.1) for i in range(30)])
    if spread[:3] != (1.0, 3.0, 1.0) or min(spread[3]) <= 0:
        raise RuntimeError("spread volume cuts drifted: %s" % (spread,))
    tight = choose_volume_cuts([0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00])
    if tight[:3] != (0.5, 1.0, 0.5) or min(tight[3]) <= 0:
        raise RuntimeError("tight volume cuts drifted: %s" % (tight,))
    try:
        choose_volume_cuts([1.0, 1.0, 1.0, 1.0])
    except Week11Stop:
        pass
    else:
        raise RuntimeError("identical volumes were given three bins")
    labels = size_labels(1.0, 3.0)
    if assign_size_label(0.99, 1.0, 3.0) != labels[0]:
        raise RuntimeError("volume below the low cut is not the small bin")
    if assign_size_label(1.0, 1.0, 3.0) != labels[1]:
        raise RuntimeError("the low cut is not in the middle bin")
    if assign_size_label(3.0, 1.0, 3.0) != labels[2]:
        raise RuntimeError("the high cut is not in the large bin")
    pedestrian = condition_draw_seed("Pedestrian", "interior", "000001", 2, 1.0)
    cyclist = condition_draw_seed("Cyclist", "interior", "000001", 2, 1.0)
    if pedestrian == cyclist:
        raise RuntimeError("class name is not part of the draw seed")
    car_token = "%d|%s|%s|%d|%.2f" % (SEED, "interior", "000001", 2, 1.0)
    car_seed = int.from_bytes(hashlib.sha256(car_token.encode("utf-8")).digest()[:4], "little")
    if pedestrian == car_seed:
        raise RuntimeError("week-11 seed collided with the week-7 car seed")
    sample = np.float32(np.pi)
    text = float32_text(sample)
    restored = np.float32(float(text))
    if restored != sample or float32_text(restored) != text:
        raise RuntimeError("placebo coordinate text does not round-trip float32")
    draw = np.random.RandomState(SEED).uniform(-80.0, 80.0, size=64).astype(np.float32)
    for value in draw:
        text = float32_text(value)
        if np.float32(float(text)) != value:
            raise RuntimeError("placebo coordinate text does not round-trip %s" % text)
    week7.self_check_counts_and_shifts()


def refuse_preflight_on_official_dir(args):
    """A short run must not overwrite the official sidecar or CSV."""
    if args.max_targets is None:
        return
    if int(args.max_targets) <= 0:
        raise Week11Stop("max-targets must be positive")
    if Path(args.output_dir).resolve() == Path(DEFAULT_OUTPUT).resolve():
        raise Week11Stop("max-targets cannot write the official week-11 output directory")


def selected_classes(args):
    """Pedestrian first, then Cyclist, unless one class was requested."""
    if args.class_name == "all":
        return CLASSES
    return (args.class_name,)


def count_named_predictions(result_by_id, val_ids, class_name):
    """Count saved predictions of one class. Zero means the pickle is the wrong file."""
    total = 0
    for frame_id in val_ids:
        names = np.asarray(result_by_id[frame_id]["name"])
        if names.size == 0:
            continue
        total += int(np.sum(names == class_name))
    return total


def class_gt(annos, class_name):
    """Ground-truth boxes of one class. DontCare and other classes are excluded."""
    names = np.asarray(annos["name"])
    gt_boxes = np.asarray(annos["gt_boxes_lidar"])
    n_boxes = int(gt_boxes.shape[0])
    names_aligned = names[:n_boxes]
    location = np.asarray(annos["location"])[:n_boxes]
    dimensions = np.asarray(annos["dimensions"])[:n_boxes]
    rotation_y = np.asarray(annos["rotation_y"])[:n_boxes]
    mask = names_aligned == class_name
    indices = np.nonzero(mask)[0]
    return {
        "gt_idx": indices.astype(np.int64),
        "cam_boxes": camera_boxes_from_anno(
            location[mask], dimensions[mask], rotation_y[mask]
        ),
    }


def process_frame(frame_id, pred_dict, info, velodyne_dir, class_name):
    """One row per saved prediction of this class. pred_idx indexes the full output."""
    names = np.asarray(pred_dict["name"])
    if names.size == 0:
        return []
    pred_idx = np.nonzero(names == class_name)[0]
    if pred_idx.size == 0:
        return []
    boxes_lidar = np.asarray(pred_dict["boxes_lidar"], dtype=np.float64)[pred_idx]
    scores = np.asarray(pred_dict["score"], dtype=np.float64)[pred_idx]
    pred_cam = camera_boxes_from_anno(
        np.asarray(pred_dict["location"])[pred_idx],
        np.asarray(pred_dict["dimensions"])[pred_idx],
        np.asarray(pred_dict["rotation_y"])[pred_idx],
    )
    gt = class_gt(info["annos"], class_name)
    points = load_velodyne(velodyne_dir, frame_id)
    point_counts = count_points_in_boxes(points[:, 0:3], boxes_lidar)
    ranges = horizontal_range_m(boxes_lidar)
    matched, gt_idx, iou_3d = match_preds_to_gt(
        pred_cam, gt["cam_boxes"], gt["gt_idx"], MATCH_IOU_THRESH
    )
    rows = []
    for local_i, full_idx in enumerate(pred_idx):
        box = boxes_lidar[local_i]
        rows.append(
            {
                "frame_id": frame_id,
                "pred_idx": int(full_idx),
                "score": float(scores[local_i]),
                "box_x": float(box[0]),
                "box_y": float(box[1]),
                "box_z": float(box[2]),
                "box_l": float(box[3]),
                "box_w": float(box[4]),
                "box_h": float(box[5]),
                "box_yaw": float(box[6]),
                "range_m": float(ranges[local_i]),
                "num_points_in_box": int(point_counts[local_i]),
                "gt_matched": bool(matched[local_i]),
                "gt_idx": int(gt_idx[local_i]) if matched[local_i] else "",
                "iou_3d": float(iou_3d[local_i]),
            }
        )
    return rows


def validate_metadata(rows, walked_ids, val_ids, expected_rows, class_name):
    """Check the written table against the pickle. Do not apply the Car near-box rule."""
    if walked_ids != list(val_ids):
        raise Week11Stop("%s metadata walk is not the val split" % class_name)
    if len(rows) != int(expected_rows):
        raise Week11Stop(
            "%s metadata has %d rows, result.pkl has %d"
            % (class_name, len(rows), int(expected_rows))
        )
    seen = set()
    min_score = None
    for row in rows:
        identity = (row["frame_id"], int(row["pred_idx"]))
        if identity in seen:
            raise Week11Stop("%s duplicate metadata identity %s" % (class_name, identity))
        seen.add(identity)
        if row["num_points_in_box"] < 0:
            raise Week11Stop("%s negative interior count at %s" % (class_name, row["frame_id"]))
        score = float(row["score"])
        min_score = score if min_score is None else min(min_score, score)
        if row["gt_matched"]:
            if not (0.0 < float(row["iou_3d"]) <= 1.0 + 1.0e-6) or row["gt_idx"] == "":
                raise Week11Stop(
                    "%s matched row is incomplete at %s pred %s"
                    % (class_name, row["frame_id"], row["pred_idx"])
                )
        elif row["gt_idx"] != "" or abs(float(row["iou_3d"])) > 1.0e-12:
            raise Week11Stop(
                "%s unmatched row stores an IoU at %s pred %s"
                % (class_name, row["frame_id"], row["pred_idx"])
            )
    if min_score is not None and min_score < float(DETECTOR_SCORE_THRESHOLD) - 1.0e-6:
        raise Week11Stop(
            "%s metadata score %.4f is below the saved-output floor %.1f"
            % (class_name, min_score, DETECTOR_SCORE_THRESHOLD)
        )
    return {
        "rows": len(rows),
        "matched": sum(1 for row in rows if row["gt_matched"]),
        "min_score": min_score,
    }


def build_or_load_metadata(args, class_name, val_ids, result_by_id, info_by_id):
    """Build the per-detection table once. A later process only reloads it."""
    paths = class_paths(args.output_dir, class_name)
    expected = count_named_predictions(result_by_id, val_ids, class_name)
    if expected == 0:
        raise Week11Stop(
            "%s has no rows in result.pkl. Do not rerun test.py from this driver."
            % class_name
        )
    if not paths["metadata"].exists() or not paths["walked"].exists():
        rows = []
        walked = []
        print(
            "building %s metadata over %d frames, expected rows %d"
            % (class_name, len(val_ids), expected),
            flush=True,
        )
        for index, frame_id in enumerate(val_ids, start=1):
            walked.append(frame_id)
            rows.extend(
                process_frame(
                    frame_id,
                    result_by_id[frame_id],
                    info_by_id[frame_id],
                    args.velodyne_dir,
                    class_name,
                )
            )
            if index % 200 == 0 or index == len(val_ids):
                print(
                    "metadata %s %d/%d frames, rows %d"
                    % (class_name, index, len(val_ids), len(rows)),
                    flush=True,
                )
        paths["metadata"].parent.mkdir(parents=True, exist_ok=True)
        write_csv(paths["metadata"], rows)
        week7.atomic_write(paths["walked"], "\n".join(walked) + "\n")
    rows = read_csv_rows(paths["metadata"])
    walked = [
        line.strip()
        for line in paths["walked"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = validate_metadata(rows, walked, val_ids, expected, class_name)
    write_csv(paths["sample"], rows[:50])
    print("%s metadata rows %d matched %d" % (class_name, report["rows"], report["matched"]), flush=True)
    return rows, report


def parse_class_ap(result_str, class_name):
    """Read the official 0.50 block. The later 0.25 block is not the zoo number."""
    r11 = re.search(
        r"%s AP@0\.50, 0\.50, 0\.50:\s*.*?3d   AP:([0-9.]+), ([0-9.]+), ([0-9.]+)"
        % class_name,
        result_str,
        flags=re.DOTALL,
    )
    r40 = re.search(
        r"%s AP_R40@0\.50, 0\.50, 0\.50:\s*.*?3d   AP:([0-9.]+), ([0-9.]+), ([0-9.]+)"
        % class_name,
        result_str,
        flags=re.DOTALL,
    )
    if r11 is None or r40 is None:
        raise Week11Stop("failed to parse %s official 3D AP from eval output" % class_name)
    return {
        "r11_easy": float(r11.group(1)),
        "r11_moderate": float(r11.group(2)),
        "r11_hard": float(r11.group(3)),
        "r40_easy": float(r40.group(1)),
        "r40_moderate": float(r40.group(2)),
        "r40_hard": float(r40.group(3)),
    }


def load_or_score_ap(args, class_name, val_ids, result_by_id, info_by_id):
    """Score the saved predictions once. Compare moderate AP@R11 with the zoo."""
    paths = class_paths(args.output_dir, class_name)
    published = PUBLISHED_MODERATE_R11[class_name]
    if paths["ap"].exists() and paths["eval"].exists():
        with paths["ap"].open("r", encoding="utf-8", newline="") as handle:
            loaded = list(csv.DictReader(handle))
        if len(loaded) != 1 or any(column not in loaded[0] for column in AP_COLUMNS):
            raise Week11Stop("%s cached AP file has the wrong shape" % class_name)
        ap = {column: float(loaded[0][column]) for column in AP_COLUMNS}
    else:
        gt_annos = [copy.deepcopy(info_by_id[frame_id]["annos"]) for frame_id in val_ids]
        dt_annos = [copy.deepcopy(result_by_id[frame_id]) for frame_id in val_ids]
        result_str, _ret = get_official_eval_result(gt_annos, dt_annos, class_name)
        ap = parse_class_ap(result_str, class_name)
        paths["ap"].parent.mkdir(parents=True, exist_ok=True)
        with paths["ap"].open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=AP_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerow({column: "%.4f" % ap[column] for column in AP_COLUMNS})
        week7.atomic_write(paths["eval"], result_str)
    difference = abs(float(ap["r11_moderate"]) - published)
    if difference > PUBLISHED_AP_TOLERANCE:
        raise Week11Stop(
            "%s moderate 3D AP@R11 %.4f differs from published %.2f by %.4f"
            % (class_name, ap["r11_moderate"], published, difference)
        )
    ap["published_moderate"] = published
    ap["difference"] = difference
    return ap


def annotate_row(row, low, high):
    """Add week-4 range and point bins, plus this class's volume bin."""
    volume = size_m3(row["box_l"], row["box_w"], row["box_h"])
    range_index, point_index, _ignored = assign_bins(
        row["range_m"], row["num_points_in_box"], volume
    )
    annotated = dict(row)
    annotated["size_m3"] = float(volume)
    annotated["range_bin"] = RANGE_LABELS[int(range_index)]
    annotated["num_points_bin"] = POINT_LABELS[int(point_index)]
    annotated["size_bin"] = assign_size_label(volume, low, high)
    return annotated


def read_definition_blocks(path):
    """Return class blocks already stored in the shared definitions file."""
    path = Path(path)
    if not path.exists():
        return {}
    blocks = {}
    current = None
    buffer = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("class: "):
            if current is not None:
                blocks[current] = "\n".join(buffer).strip() + "\n"
            current = line.split("class: ", 1)[1].strip()
            buffer = [line]
        elif current is not None:
            buffer.append(line)
    if current is not None:
        blocks[current] = "\n".join(buffer).strip() + "\n"
    return blocks


def cuts_from_block(block, class_name):
    """Read a previously written volume cut. Missing fields stop the run."""
    low = None
    high = None
    step = None
    for line in block.splitlines():
        if line.startswith("volume_low_m3:"):
            low = float(line.split(":", 1)[1].strip())
        elif line.startswith("volume_high_m3:"):
            high = float(line.split(":", 1)[1].strip())
        elif line.startswith("volume_step_m3:"):
            step = float(line.split(":", 1)[1].strip())
    if low is None or high is None or high <= low:
        raise Week11Stop("%s definitions block has no usable volume cuts" % class_name)
    return low, high, step


def resolve_volume_cuts(class_name, volumes, stored, results_started):
    """Use the stored cuts once a result row exists. Otherwise use the rule."""
    computed = choose_volume_cuts(volumes)
    if stored is None:
        return computed
    same = abs(stored[0] - computed[0]) < 1.0e-9 and abs(stored[1] - computed[1]) < 1.0e-9
    if same:
        return computed
    counts = volume_bin_counts(volumes, stored[0], stored[1])
    if results_started:
        raise Week11Stop(
            "%s volume cuts are locked at %s and %s because result rows exist"
            % (class_name, format_m3(stored[0]), format_m3(stored[1]))
        )
    if min(counts) == 0:
        print("replacing an empty stored volume cut for %s" % class_name, flush=True)
        return computed
    print("keeping stored volume cuts for %s" % class_name, flush=True)
    return stored[0], stored[1], stored[2], counts


def protocol_header():
    """Shared definitions. Class blocks carry the volume cuts."""
    return "\n".join(
        [
            "Week 11 definitions",
            "detector: PointPillars %s" % DEFAULT_CKPT.name,
            "config: tools/cfgs/kitti_models/pointpillar.yaml",
            "dataset: KITTI validation split, %d frames" % EXPECTED_FRAMES,
            "classes: Pedestrian and Cyclist, reported separately",
            "predictions: saved result.pkl, no new test.py run",
            "official 3D overlap: %.2f for Pedestrian and for Cyclist" % OFFICIAL_OVERLAP,
            "association overlap: %.2f to the original predicted box of the same class"
            % ASSOC_IOU,
            "eligibility: score >= %.2f and interior points >= %d" % (SCORE_FLOOR, MIN_POINTS),
            "ground-truth match: best 3D IoU to the same class at %.2f, not required"
            % MATCH_IOU_THRESH,
            "range bins: 0-20 m, 20-40 m, 40 m or farther",
            "interior-point bins: 0-19, 20-99, 100 or more",
            "car volume cuts are not used",
            "volume rule: coarsest step in %s m3 whose rounded one-third and two-thirds ranks leave three nonempty bins"
            % ", ".join(format_m3(step) for step in VOLUME_STEPS),
            "fractions: %s" % ", ".join(FRACTION_LABELS),
            "seed: %d" % SEED,
            'draw token: sha256("%d|<class>|<condition>|<frame_id>|<pred_idx>|<fraction>")' % SEED,
            "num_removed: Python round(fraction * region_points), half even",
            "eval batch size: %d" % EVAL_BATCH_SIZE,
            "conditions: interior, same-size empty-region placebo, context",
            "placebo obstacles: every predicted box and every non-DontCare ground-truth box",
            "retained confidence: score_after / score_before on association, otherwise 0",
            "score survival: association and score_after >= %.2f" % SURVIVAL_SCORE,
            "published moderate 3D AP@R11: Pedestrian %.2f, Cyclist %.2f"
            % (PUBLISHED_MODERATE_R11["Pedestrian"], PUBLISHED_MODERATE_R11["Cyclist"]),
            "AP@R40 is recorded and is not compared with the published R11 numbers",
            "",
        ]
    )


def class_block(class_name, ap, report, stats, cuts):
    """One class section, including the cuts that later forwards must keep."""
    low, high, step, counts = cuts
    labels = size_labels(low, high)
    volumes = stats["volumes"]
    lines = [
        "class: %s" % class_name,
        "official 3D overlap: %.2f" % OFFICIAL_OVERLAP,
        "published moderate 3D AP@R11: %.2f" % ap["published_moderate"],
        "reproduced 3D AP@R11 easy/moderate/hard: %.4f / %.4f / %.4f"
        % (ap["r11_easy"], ap["r11_moderate"], ap["r11_hard"]),
        "reproduced 3D AP@R40 easy/moderate/hard: %.4f / %.4f / %.4f"
        % (ap["r40_easy"], ap["r40_moderate"], ap["r40_hard"]),
        "moderate R11 absolute difference: %.4f" % ap["difference"],
        "prediction rows: %d" % report["rows"],
        "prediction rows matched at %.2f: %d" % (MATCH_IOU_THRESH, report["matched"]),
        "eligible: %d" % stats["eligible"],
        "eligible gt_matched: %d" % stats["eligible_matched"],
        "eligible by range: %s"
        % ", ".join(
            "%s %d" % (label, stats["range_counts"][index])
            for index, label in enumerate(RANGE_LABELS)
        ),
        "eligible by points: %s"
        % ", ".join(
            "%s %d" % (label, stats["point_counts"][index])
            for index, label in enumerate(POINT_LABELS)
        ),
        "volume_low_m3: %s" % format_m3(low),
        "volume_high_m3: %s" % format_m3(high),
        "volume_step_m3: %s" % format_m3(step) if step is not None else "volume_step_m3: ",
        "volume labels: %s" % " | ".join(labels),
        "eligible by volume: %s"
        % ", ".join("%s %d" % (label, counts[index]) for index, label in enumerate(labels)),
        "eligible volume min/median/max m3: %.4f / %.4f / %.4f"
        % (min(volumes), statistics.median(volumes), max(volumes)),
    ]
    return "\n".join(lines) + "\n"


def write_definitions(output_dir, blocks):
    """Rewrite the shared definitions file without dropping another class."""
    parts = [protocol_header()]
    for class_name in CLASSES:
        if class_name in blocks:
            parts.append(blocks[class_name].strip() + "\n")
    week7.atomic_write(Path(output_dir) / DEFINITIONS_NAME, "\n".join(parts))


def eligible_targets(rows, low, high):
    """Score and interior filters only. Counts are recorded, not matched to Car."""
    kept = []
    for row in rows:
        if float(row["score"]) < SCORE_FLOOR or int(row["num_points_in_box"]) < MIN_POINTS:
            continue
        kept.append(row)
    if not kept:
        raise Week11Stop(
            "no row passed score >= %.2f and interior points >= %d" % (SCORE_FLOOR, MIN_POINTS)
        )
    targets = [annotate_row(row, low, high) for row in kept]
    targets.sort(key=lambda row: (row["frame_id"], int(row["pred_idx"])))
    seen = set()
    for target in targets:
        identity = (target["frame_id"], int(target["pred_idx"]))
        if identity in seen:
            raise Week11Stop("duplicate eligible identity %s" % (identity,))
        seen.add(identity)
    return targets, population_stats(targets, low, high)


def population_stats(targets, low, high):
    """Counts for the targets in this run. A max-targets prefix replaces the full census."""
    labels = size_labels(low, high)
    return {
        "eligible": len(targets),
        "eligible_matched": sum(1 for target in targets if target["gt_matched"]),
        "range_counts": tuple(
            sum(1 for target in targets if target["range_bin"] == label) for label in RANGE_LABELS
        ),
        "point_counts": tuple(
            sum(1 for target in targets if target["num_points_bin"] == label)
            for label in POINT_LABELS
        ),
        "size_counts": tuple(
            sum(1 for target in targets if target["size_bin"] == label) for label in labels
        ),
        "volumes": [float(target["size_m3"]) for target in targets],
    }


def as_boxes(array):
    """Return (N, 7) float32 boxes. An empty detector output is an empty array."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 7), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise Week11Stop("expected boxes of shape (N, 7), got %s" % (arr.shape,))
    return arr


def frame_obstacles(pred_dict, annos):
    """Every predicted box, plus every ground-truth box except DontCare."""
    pred = as_boxes(pred_dict["boxes_lidar"])
    gt = as_boxes(annos["gt_boxes_lidar"])
    names = np.asarray(annos["name"])
    if int(names.shape[0]) < int(gt.shape[0]):
        raise Week11Stop("ground-truth names are shorter than gt_boxes_lidar")
    gt = gt[names[: int(gt.shape[0])] != "DontCare"]
    parts = [item for item in (pred, gt) if item.shape[0]]
    if not parts:
        return np.zeros((0, 7), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def build_obstacles(targets, result_by_id, info_by_id):
    """Obstacle boxes for every frame that contains a target."""
    obstacles = {}
    for target in targets:
        frame_id = target["frame_id"]
        if frame_id not in obstacles:
            obstacles[frame_id] = frame_obstacles(
                result_by_id[frame_id], info_by_id[frame_id]["annos"]
            )
    return obstacles


def attach_placebos(targets, obstacles, velodyne_dir):
    """Search one empty box per target. Obstacles are not limited to the target class."""
    clouds = {}
    placed = 0
    started = time.perf_counter()
    for index, target in enumerate(targets, start=1):
        frame_id = target["frame_id"]
        points = week7.cached_cloud(clouds, velodyne_dir, frame_id)
        shifted, region_points = week7.find_placebo(
            points[:, 0:3], box_from_row(target), obstacles[frame_id]
        )
        if shifted is None:
            target["placebo_status"] = "unavailable"
            target["placebo_box"] = None
            target["placebo_points"] = None
        else:
            target["placebo_status"] = "applied"
            target["placebo_box"] = shifted
            target["placebo_points"] = int(region_points)
            placed += 1
        if index % 500 == 0 or index == len(targets):
            print(
                "placed %d / %d  applied %d  %.0f s"
                % (index, len(targets), placed, time.perf_counter() - started),
                flush=True,
            )
    return placed, len(targets) - placed


def float32_text(value):
    """Nine significant digits. This round-trips every finite float32 value."""
    return format(float(np.float32(value)), ".9g")


def box_field(box, index):
    """Sidecar coordinate. Empty when this target has no placebo box."""
    if box is None:
        return ""
    return float32_text(box[index])


def identity_record(target):
    """Identity tuple, including lossless float32 placebo coordinates."""
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
        box_field(box, 0),
        box_field(box, 1),
        box_field(box, 2),
        box_field(box, 3),
        box_field(box, 4),
        box_field(box, 5),
        box_field(box, 6),
    )


def target_interior(points, target):
    """Interior indices. Stop when the live count disagrees with the metadata row."""
    inside = interior_index(points[:, 0:3], box_from_row(target))
    csv_points = int(target["num_points_in_box"])
    if int(inside.size) != csv_points:
        raise Week11Stop(
            "%s pred %d interior %d != metadata %d"
            % (target["frame_id"], int(target["pred_idx"]), int(inside.size), csv_points)
        )
    return inside


def joint_census(targets, labels):
    """Count the 27 joint cells for this class's volume labels, including zeros."""
    counts = {}
    for range_label in RANGE_LABELS:
        for point_label in POINT_LABELS:
            for size_label in labels:
                counts[(range_label, point_label, size_label)] = 0
    for target in targets:
        key = (target["range_bin"], target["num_points_bin"], target["size_bin"])
        if key not in counts:
            raise Week11Stop("target fell outside the bins: %s" % (key,))
        counts[key] += 1
    return counts


def sidecar_text(class_name, targets, stats, cuts, max_targets):
    """Census and placebo placements. A restart has to reproduce this text."""
    low, high, step, _counts = cuts
    labels = size_labels(low, high)
    lines = [
        "Week 11 sidecar",
        "class: %s" % class_name,
        "seed: %d" % SEED,
        'draw token: sha256("%d|<class>|<condition>|<frame_id>|<pred_idx>|<fraction>")' % SEED,
        "fractions: %s" % ", ".join(FRACTION_LABELS),
        "conditions: %s" % ", ".join(CONDITIONS),
        "num_removed: Python round(fraction * region_points), half even",
        "eval batch: %d. Only the target frame is edited." % EVAL_BATCH_SIZE,
        "association class: %s" % class_name,
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
        "volume_low_m3: %s" % format_m3(low),
        "volume_high_m3: %s" % format_m3(high),
        "volume_step_m3: %s" % ("" if step is None else format_m3(step)),
        "order: frame_id, then pred_idx",
        "placebo: same length, width, height, yaw, and z",
        "placebo obstacles: all predicted boxes and all non-DontCare ground-truth boxes",
        "placebo applied: %d" % stats["placebo_applied"],
        "placebo unavailable: %d" % stats["placebo_unavailable"],
        "max_targets: %s" % ("none" if max_targets is None else str(int(max_targets))),
        "joint counts:",
    ]
    for range_label in RANGE_LABELS:
        for point_label in POINT_LABELS:
            for size_label in labels:
                key = (range_label, point_label, size_label)
                lines.append(
                    "  %s n=%d"
                    % (week7.joint_label(range_label, point_label, size_label), stats["joint_counts"][key])
                )
    lines.append("identity_table:")
    lines.append(",".join(week7.IDENTITY_COLUMNS))
    text = "\n".join(lines) + "\n"
    identity_lines = []
    for target in targets:
        identity_lines.append(",".join(identity_record(target)))
    return text + "\n".join(identity_lines) + "\n"


def confirm_sidecar(path, class_name, targets, low, high):
    """Refuse a CSV whose sidecar was drawn with different bins or placements."""
    text = Path(path).read_text(encoding="utf-8")
    required = (
        "class: %s" % class_name,
        "volume_low_m3: %s" % format_m3(low),
        "volume_high_m3: %s" % format_m3(high),
    )
    missing = [line for line in required if line not in text.splitlines()]
    if missing:
        raise Week11Stop("existing sidecar does not match this class draw: %s" % missing)
    try:
        saved = week7.parse_sidecar_identities(path)
    except ValueError as exc:
        raise Week11Stop("existing sidecar cannot be read: %s" % exc) from exc
    fresh = [identity_record(target) for target in targets]
    if saved != fresh:
        raise Week11Stop("existing sidecar does not match the deterministic draw")


def restore_placebos(path, class_name, targets, low, high):
    """Load the locked placements. A restart does not search for them again."""
    text = Path(path).read_text(encoding="utf-8")
    required = (
        "class: %s" % class_name,
        "volume_low_m3: %s" % format_m3(low),
        "volume_high_m3: %s" % format_m3(high),
    )
    missing = [line for line in required if line not in text.splitlines()]
    if missing:
        raise Week11Stop("existing sidecar does not match this class draw: %s" % missing)
    try:
        saved = week7.parse_sidecar_identities(path)
    except ValueError as exc:
        raise Week11Stop("existing sidecar cannot be read: %s" % exc) from exc
    if len(saved) != len(targets):
        raise Week11Stop("sidecar has %d rows and this draw has %d" % (len(saved), len(targets)))
    box_fields = (
        "placebo_x",
        "placebo_y",
        "placebo_z",
        "placebo_l",
        "placebo_w",
        "placebo_h",
        "placebo_yaw",
    )
    for saved_row, target in zip(saved, targets):
        record = dict(zip(week7.IDENTITY_COLUMNS, saved_row))
        if record["frame_id"] != target["frame_id"] or int(record["pred_idx"]) != int(target["pred_idx"]):
            raise Week11Stop(
                "sidecar identity %s pred %s does not match the eligible list"
                % (record["frame_id"], record["pred_idx"])
            )
        if record["range_bin"] != target["range_bin"] or record["num_points_bin"] != target["num_points_bin"]:
            raise Week11Stop("sidecar bins changed for %s pred %s" % (record["frame_id"], record["pred_idx"]))
        if record["size_bin"] != target["size_bin"]:
            raise Week11Stop("sidecar volume bin changed for %s pred %s" % (record["frame_id"], record["pred_idx"]))
        if record["score"] != "%.6f" % float(target["score"]):
            raise Week11Stop("sidecar score changed for %s pred %s" % (record["frame_id"], record["pred_idx"]))
        if int(record["num_points_in_box"]) != int(target["num_points_in_box"]):
            raise Week11Stop(
                "sidecar interior count changed for %s pred %s" % (record["frame_id"], record["pred_idx"])
            )
        status = record["placebo_status"]
        if status == "applied":
            target["placebo_status"] = "applied"
            target["placebo_points"] = int(record["placebo_points"])
            target["placebo_box"] = np.asarray(
                [np.float32(record[field]) for field in box_fields], dtype=np.float32
            )
        elif status == "unavailable":
            if record["placebo_points"] != "" or any(record[field] != "" for field in box_fields):
                raise Week11Stop(
                    "unavailable sidecar row stores a box at %s pred %s"
                    % (record["frame_id"], record["pred_idx"])
                )
            target["placebo_status"] = "unavailable"
            target["placebo_points"] = None
            target["placebo_box"] = None
        else:
            raise Week11Stop("unknown placebo status %s" % status)
    confirm_sidecar(path, class_name, targets, low, high)


def results_started(path):
    """True when any result row is on disk. Volume cuts are then locked."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    return len(week7.read_result_rows(path)) > 0


def edit_condition(points, target, class_name, condition, fraction, inside, obstacles):
    """Week-7 delete, with the class name inside the draw seed."""
    prefix = "%s %s pred %d %s f=%.2f" % (
        class_name,
        target["frame_id"],
        int(target["pred_idx"]),
        condition,
        float(fraction),
    )
    xyz = points[:, 0:3]
    if condition == "interior":
        region_idx = inside
    elif condition == "context":
        region_idx = week7.outside_indices(points.shape[0], inside)
    elif condition == "placebo":
        if target["placebo_status"] != "applied":
            raise Week11Stop("%s placebo edit requested without a placement" % prefix)
        region_idx = interior_index(xyz, target["placebo_box"])
        if int(region_idx.size) != int(target["placebo_points"]):
            raise Week11Stop(
                "%s placebo points %d != placement %d"
                % (prefix, int(region_idx.size), int(target["placebo_points"]))
            )
        if week7.region_hits_obstacle(xyz, region_idx, obstacles):
            raise Week11Stop("%s placebo region intersects an object" % prefix)
    else:
        raise Week11Stop("%s unknown condition" % prefix)
    region_points = int(region_idx.size)
    n_removed = num_to_remove(region_points, fraction)
    delete_idx = choose_delete_indices(
        region_idx,
        n_removed,
        condition_draw_seed(
            class_name, condition, target["frame_id"], target["pred_idx"], fraction
        ),
    )
    modified = apply_delete(points, delete_idx)
    week7.check_edit(
        points, target, condition, fraction, inside, region_idx, delete_idx, modified, obstacles
    )
    return modified, region_points, n_removed


def associate_to_class(pred_dict, original_box_lidar, calib, class_name):
    """Best returned box of this class. A box of another class does not count."""
    names = np.asarray(pred_dict["name"])
    if names.size == 0 or not np.any(names == class_name):
        return None, 0.0, False
    mask = names == class_name
    new_cam = camera_boxes_from_anno(
        np.asarray(pred_dict["location"])[mask],
        np.asarray(pred_dict["dimensions"])[mask],
        np.asarray(pred_dict["rotation_y"])[mask],
    )
    original_cam = box_utils.boxes3d_lidar_to_kitti_camera(
        np.asarray(original_box_lidar, dtype=np.float32).reshape(1, 7),
        calib,
    )
    original_cam = np.asarray(original_cam, dtype=np.float64).reshape(1, 7)
    overlaps = d3_box_overlap(
        np.ascontiguousarray(new_cam, dtype=np.float64),
        np.ascontiguousarray(original_cam, dtype=np.float64),
    )
    overlaps = np.asarray(overlaps, dtype=np.float64).reshape(-1)
    best = int(np.argmax(overlaps))
    iou = float(overlaps[best])
    scores = np.asarray(pred_dict["score"], dtype=np.float64)[mask]
    return float(scores[best]), iou, bool(iou >= ASSOC_IOU)


def require_unmodified(class_name, target, best_score, best_iou, survived):
    """Fraction 0 has to come back as the saved box, within the week-7 tolerance."""
    prefix = "%s %s pred %d f=0.00" % (
        class_name,
        target["frame_id"],
        int(target["pred_idx"]),
    )
    if not survived or best_score is None:
        raise Week11Stop("%s forward produced no associated %s" % (prefix, class_name))
    if float(best_iou) < F0_IOU_MIN:
        raise Week11Stop("%s iou %.4f is below %.2f" % (prefix, float(best_iou), F0_IOU_MIN))
    delta = abs(float(best_score) - float(target["score"]))
    if delta >= F0_SCORE_TOL:
        raise Week11Stop(
            "%s |score_after %.4f - score_before %.4f| = %.4f"
            % (prefix, float(best_score), float(target["score"]), delta)
        )


def run_forwards(args, class_name, targets, results_path, done, obstacle_by_frame):
    """Forward unfinished cells. Fraction 0 is one forward shared by the applied conditions."""
    dataset, model, info_by_id, index_by_frame = load_runtime(args)
    clouds = {}
    calib_by_frame = {}
    header_present = results_path.exists() and results_path.stat().st_size > 0
    handle, writer = week7.open_writer(
        results_path, week7.RESULT_COLUMNS, already_has_rows=header_present
    )
    started = time.perf_counter()
    written = 0
    forwarded = 0
    try:
        for target in targets:
            if not week7.target_has_pending(target, done):
                continue
            frame_id = target["frame_id"]
            points = week7.cached_cloud(clouds, args.velodyne_dir, frame_id)
            inside = target_interior(points, target)
            if frame_id not in obstacle_by_frame:
                raise Week11Stop("missing obstacle boxes for %s" % frame_id)
            obstacles = obstacle_by_frame[frame_id]
            if frame_id not in calib_by_frame:
                calib_by_frame[frame_id] = dataset.get_calib(
                    info_by_id[frame_id]["point_cloud"]["lidar_idx"]
                )
            calib = calib_by_frame[frame_id]
            for fraction_label in FRACTION_LABELS:
                key = (frame_id, int(target["pred_idx"]), "placebo", fraction_label)
                if target["placebo_status"] == "unavailable" and key not in done:
                    week7.append_row(
                        handle,
                        writer,
                        week7.blank_metric_row(target, "placebo", fraction_label, "unavailable"),
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
                best_score, best_iou, survived = associate_to_class(
                    pred_dict, box_from_row(target), calib, class_name
                )
                require_unmodified(class_name, target, best_score, best_iou, survived)
                for condition in f0_conditions:
                    region_points = week7.region_size(points, target, condition, inside)
                    if num_to_remove(region_points, 0.0) != 0:
                        raise Week11Stop(
                            "%s %s pred %d %s f=0 removed points"
                            % (class_name, frame_id, int(target["pred_idx"]), condition)
                        )
                    row = week7.metric_row(
                        target,
                        condition,
                        FRACTION_LABELS[0],
                        region_points,
                        0,
                        week7.status_for(condition, target),
                        best_score,
                        best_iou,
                        survived,
                    )
                    week7.append_row(handle, writer, row)
                    done.add((frame_id, int(target["pred_idx"]), condition, FRACTION_LABELS[0]))
                    written += 1
                week7.log_progress(started, forwarded, written, len(targets))
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
                        points, target, class_name, condition, fraction, inside, obstacles
                    )
                    pred_dict = forward_in_eval_batch(
                        dataset, model, info_by_id, index_by_frame, frame_id, modified
                    )
                    forwarded += 1
                    best_score, best_iou, survived = associate_to_class(
                        pred_dict, box_from_row(target), calib, class_name
                    )
                    row = week7.metric_row(
                        target,
                        condition,
                        fraction_label,
                        region_points,
                        n_removed,
                        week7.status_for(condition, target),
                        best_score,
                        best_iou,
                        survived,
                    )
                    week7.append_row(handle, writer, row)
                    done.add(key)
                    written += 1
                    if forwarded % 500 == 0:
                        torch.cuda.empty_cache()
                    week7.log_progress(started, forwarded, written, len(targets))
            if forwarded >= RESTART_AFTER_FORWARDS:
                raise RestartChunk("%s forwarded %d new cells" % (class_name, forwarded))
    finally:
        handle.flush()
        handle.close()
    return forwarded


def probe_choices(targets):
    """One eligible target from each range bin that contains one."""
    chosen = []
    for label in RANGE_LABELS:
        for target in targets:
            if target["range_bin"] == label:
                chosen.append(target)
                break
    if not chosen:
        raise Week11Stop("no eligible target for the unmodified probe")
    return chosen


def preflight_is_current(path, targets):
    """True when this eligible set already has a passing batch-size-4 probe."""
    path = Path(path)
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    required = (
        "eval batch size: %d" % EVAL_BATCH_SIZE,
        "correctness failures: 0",
        "decision: correctness passed",
    )
    if any(line not in text.splitlines() for line in required):
        return False
    eligible = {(target["frame_id"], int(target["pred_idx"])) for target in targets}
    found = False
    for line in text.splitlines():
        if not line.startswith("probed:"):
            continue
        found = True
        parts = line.split()
        if len(parts) < 3:
            return False
        identity = (parts[1], int(parts[2]))
        if identity not in eligible:
            return False
    return found


def probe_unmodified(args, class_name, targets):
    """Live unmodified forwards before the edited run. Batch size stays 4."""
    paths = class_paths(args.output_dir, class_name)
    chosen = probe_choices(targets)
    if preflight_is_current(paths["preflight"], targets):
        print("%s unmodified probe already passed" % class_name, flush=True)
        return
    dataset, model, info_by_id, index_by_frame = load_runtime(args)
    lines = [
        "Week 11 unmodified probe",
        "class: %s" % class_name,
        "targets: %d" % len(chosen),
        "eval batch size: %d" % EVAL_BATCH_SIZE,
    ]
    failures = []
    for target in chosen:
        frame_id = target["frame_id"]
        points = load_velodyne(args.velodyne_dir, frame_id)
        try:
            target_interior(points, target)
            pred_dict = forward_in_eval_batch(
                dataset, model, info_by_id, index_by_frame, frame_id, points
            )
            calib = dataset.get_calib(info_by_id[frame_id]["point_cloud"]["lidar_idx"])
            best_score, best_iou, survived = associate_to_class(
                pred_dict, box_from_row(target), calib, class_name
            )
            require_unmodified(class_name, target, best_score, best_iou, survived)
            lines.append(
                "probed: %s %d %s iou %.4f"
                % (frame_id, int(target["pred_idx"]), target["range_bin"], float(best_iou))
            )
        except (Week11Stop, week7.Week7Stop, PilotStop) as exc:
            failures.append(str(exc))
            lines.append(
                "probed: %s %d %s failed" % (frame_id, int(target["pred_idx"]), target["range_bin"])
            )
    lines.append("correctness failures: %d" % len(failures))
    if failures:
        lines.extend("failure: %s" % item for item in failures)
        lines.append("decision: INVALID")
    else:
        lines.append("decision: correctness passed")
    week7.atomic_write(paths["preflight"], "\n".join(lines) + "\n")
    del model
    torch.cuda.empty_cache()
    if failures:
        raise Week11Stop("%s unmodified probe failed: %s" % (class_name, failures[0]))
    print("%s unmodified probe passed on %d targets" % (class_name, len(chosen)), flush=True)


def current_stamps(args):
    """Size and mtime for the files the run is not allowed to change."""
    return {
        "result_pkl": list(file_stamp(args.result_pkl)),
        "checkpoint": list(file_stamp(args.ckpt)),
        "velodyne": {
            path.name: list(file_stamp(path)) for path in velodyne_bin_paths(args.velodyne_dir)
        },
    }


def initialize_or_check_stamps(args):
    """Record the input stamps once, then compare them on every restart."""
    stamp_path = Path(args.output_dir) / STAMP_NAME
    print("stamping velodyne, checkpoint, and result.pkl", flush=True)
    now = current_stamps(args)
    if not stamp_path.exists():
        week7.atomic_write(stamp_path, json.dumps(now) + "\n")
        return "input baseline recorded (%d Velodyne files)" % len(now["velodyne"]), []
    saved = json.loads(stamp_path.read_text(encoding="utf-8"))
    failures = []
    for key in ("result_pkl", "checkpoint", "velodyne"):
        if saved.get(key) != now.get(key):
            failures.append("%s changed since the week-11 baseline" % key)
    if failures:
        return "input stamp mismatch", failures
    return (
        "checkpoint, result.pkl, and %d Velodyne files unchanged" % len(now["velodyne"]),
        [],
    )


def audit_f1_edits(class_name, targets, velodyne_dir, obstacle_by_frame):
    """Recompute fraction-1 deletes from the original clouds."""
    failures = []
    clouds = {}
    for target in targets:
        frame_id = target["frame_id"]
        points = week7.cached_cloud(clouds, velodyne_dir, frame_id)
        if frame_id not in obstacle_by_frame:
            failures.append("missing obstacles for %s" % frame_id)
            continue
        obstacles = obstacle_by_frame[frame_id]
        try:
            inside = target_interior(points, target)
            for condition in CONDITIONS:
                if condition == "placebo" and target["placebo_status"] != "applied":
                    continue
                edit_condition(points, target, class_name, condition, 1.0, inside, obstacles)
        except (Week11Stop, week7.Week7Stop, PilotStop) as exc:
            failures.append(str(exc))
    return failures


def strata(labels):
    """Marginal and joint strata for this class's three volume labels."""
    items = []
    for label in RANGE_LABELS:
        items.append(("range", label, {"range_bin": label}))
    for label in POINT_LABELS:
        items.append(("points", label, {"point_bin": label}))
    for label in labels:
        items.append(("size", label, {"size_bin": label}))
    for range_label in RANGE_LABELS:
        for point_label in POINT_LABELS:
            for size_label in labels:
                items.append(
                    (
                        "joint",
                        week7.joint_label(range_label, point_label, size_label),
                        {
                            "range_bin": range_label,
                            "point_bin": point_label,
                            "size_bin": size_label,
                        },
                    )
                )
    return items


def build_table_rows(applied, labels):
    """Every condition, stratum, and fraction, including empty cells."""
    built = []
    for condition in CONDITIONS:
        for stratum_type, stratum, filters in strata(labels):
            for fraction_label in FRACTION_LABELS:
                group = week7.slice_rows(applied, condition, fraction_label, **filters)
                built.append(
                    week7.table_row(condition, stratum_type, stratum, fraction_label, group)
                )
    return built


def marginal_lines(applied, labels):
    """Summary lines for the three marginals. Joint rates stay in the table."""
    lines = []
    marginals = (
        ("range", RANGE_LABELS, "range_bin"),
        ("points", POINT_LABELS, "point_bin"),
        ("size", labels, "size_bin"),
    )
    for condition in CONDITIONS:
        for stratum_type, axis_labels, filter_name in marginals:
            for label in axis_labels:
                filters = {filter_name: label}
                base = week7.slice_rows(applied, condition, FRACTION_LABELS[0], **filters)
                lines.append("%s | %s | %s  n=%d" % (condition, stratum_type, label, len(base)))
                if not base:
                    lines.append("  empty")
                    continue
                for fraction_label in FRACTION_LABELS:
                    group = week7.slice_rows(applied, condition, fraction_label, **filters)
                    if len(group) != len(base):
                        raise Week11Stop(
                            "%s %s %s fraction %s changed size"
                            % (condition, stratum_type, label, fraction_label)
                        )
                    score_hits, score_rate = week7.flag_rate(group, "survived_score05")
                    iou_hits, iou_rate = week7.flag_rate(group, "survived_iou05")
                    lines.append(
                        "  f=%s  mean_removed=%.3f  mean_retained=%.6f  "
                        "score_survival=%d/%d  iou_survival=%d/%d"
                        % (
                            fraction_label,
                            week7.mean_number(group, "num_removed"),
                            week7.mean_number(group, "retained"),
                            score_hits,
                            len(group),
                            iou_hits,
                            len(group),
                        )
                    )
    return lines


def summary_text(class_name, ap, stats, cuts, failures, applied, file_note, max_targets):
    """Correctness first. Rates are omitted when a check failed."""
    low, high, _step, counts = cuts
    labels = size_labels(low, high)
    lines = [
        "Week 11 summary",
        "class: %s" % class_name,
        "detector: PointPillars",
        "3D AP@R11 easy/moderate/hard: %.4f / %.4f / %.4f"
        % (ap["r11_easy"], ap["r11_moderate"], ap["r11_hard"]),
        "published moderate 3D AP@R11: %.2f  absolute difference %.4f"
        % (ap["published_moderate"], ap["difference"]),
        "3D AP@R40 easy/moderate/hard: %.4f / %.4f / %.4f"
        % (ap["r40_easy"], ap["r40_moderate"], ap["r40_hard"]),
        "eligible: %d (score >= %.2f, interior points >= %d)"
        % (stats["eligible"], SCORE_FLOOR, MIN_POINTS),
        "eligible gt_matched: %d" % stats["eligible_matched"],
        "eligible by range: %s"
        % ", ".join(
            "%s %d" % (label, stats["range_counts"][index])
            for index, label in enumerate(RANGE_LABELS)
        ),
        "eligible by points: %s"
        % ", ".join(
            "%s %d" % (label, stats["point_counts"][index])
            for index, label in enumerate(POINT_LABELS)
        ),
        "volume cuts m3: %s, %s" % (format_m3(low), format_m3(high)),
        "eligible by volume: %s"
        % ", ".join("%s %d" % (label, counts[index]) for index, label in enumerate(labels)),
        "placebo applied: %d" % stats["placebo_applied"],
        "placebo unavailable: %d" % stats["placebo_unavailable"],
        "max_targets: %s" % ("none" if max_targets is None else str(int(max_targets))),
        "files: %s" % file_note,
        "correctness failures: %d" % len(failures),
    ]
    if failures:
        lines.append("decision: INVALID")
        lines.extend("failure: %s" % item for item in failures[:30])
        lines.append("")
        return "\n".join(lines)
    lines.append("decision: correctness passed")
    lines.extend(marginal_lines(applied, labels))
    lines.append("")
    return "\n".join(lines)


def class_is_finished(paths, n_targets):
    """True when this class already has a passing summary and a complete CSV."""
    if not paths["results"].exists() or not paths["summary"].exists() or not paths["table"].exists():
        return False
    rows = week7.read_result_rows(paths["results"])
    expected = n_targets * len(CONDITIONS) * len(FRACTION_LABELS)
    if len(rows) != expected:
        return False
    text = paths["summary"].read_text(encoding="utf-8")
    return "decision: correctness passed" in text.splitlines()


def copy_outputs(output_dir, copy_dir):
    """Copy the concise files. Full per-detection and per-forward CSVs stay put."""
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


def write_stop_summary(paths, class_name, message):
    """Record an early stop without rate lines."""
    text = "\n".join(
        [
            "Week 11 summary",
            "class: %s" % class_name,
            "decision: INVALID",
            "stopped: %s" % message,
            "",
        ]
    )
    week7.atomic_write(paths["summary"], text)


def prepare_class(args, class_name, val_ids, result_by_id, info_by_id):
    """Metadata, AP, volume cuts, and the eligible list. No forwards."""
    paths = class_paths(args.output_dir, class_name)
    rows, report = build_or_load_metadata(args, class_name, val_ids, result_by_id, info_by_id)
    ap = load_or_score_ap(args, class_name, val_ids, result_by_id, info_by_id)
    volumes = [
        size_m3(row["box_l"], row["box_w"], row["box_h"])
        for row in rows
        if float(row["score"]) >= SCORE_FLOOR and int(row["num_points_in_box"]) >= MIN_POINTS
    ]
    blocks = read_definition_blocks(Path(args.output_dir) / DEFINITIONS_NAME)
    stored = cuts_from_block(blocks[class_name], class_name) if class_name in blocks else None
    cuts = resolve_volume_cuts(class_name, volumes, stored, results_started(paths["results"]))
    targets, stats = eligible_targets(rows, cuts[0], cuts[1])
    labels = size_labels(cuts[0], cuts[1])
    if tuple(stats["size_counts"]) != tuple(cuts[3]):
        raise Week11Stop(
            "%s volume counts %s do not match the cut counts %s"
            % (class_name, stats["size_counts"], cuts[3])
        )
    if min(stats["size_counts"]) <= 0:
        raise Week11Stop("%s volume bins are %s; refusing an empty bin" % (class_name, labels))
    blocks[class_name] = class_block(class_name, ap, report, stats, cuts)
    write_definitions(args.output_dir, blocks)
    print(
        "%s eligible %d  volume cuts %s %s m3  bins %s"
        % (class_name, len(targets), format_m3(cuts[0]), format_m3(cuts[1]), stats["size_counts"]),
        flush=True,
    )
    return paths, ap, targets, stats, cuts


def run_class(args, class_name, val_ids, result_by_id, info_by_id):
    """One class from metadata through the audited table."""
    paths = class_paths(args.output_dir, class_name)
    try:
        _run_class(args, class_name, val_ids, result_by_id, info_by_id, paths)
    except RestartChunk:
        raise
    except (Week11Stop, week7.Week7Stop, PilotStop, ValueError) as exc:
        existing = paths["summary"].read_text(encoding="utf-8") if paths["summary"].exists() else ""
        if "correctness failures:" not in existing:
            write_stop_summary(paths, class_name, str(exc))
            if args.max_targets is None:
                copy_outputs(args.output_dir, args.copy_dir)
        raise


def _run_class(args, class_name, val_ids, result_by_id, info_by_id, paths):
    """The class body. run_class records a stop if this raises before the audit summary."""
    _paths, ap, targets, stats, cuts = prepare_class(
        args, class_name, val_ids, result_by_id, info_by_id
    )
    if _paths != paths:
        raise Week11Stop("class paths changed during preparation")
    if args.max_targets is not None:
        targets = targets[: int(args.max_targets)]
        stats = population_stats(targets, cuts[0], cuts[1])
    if args.max_targets is None and class_is_finished(paths, len(targets)):
        print("%s already passed; leaving its table in place" % class_name, flush=True)
        copy_outputs(args.output_dir, args.copy_dir)
        return
    obstacles = build_obstacles(targets, result_by_id, info_by_id)
    if paths["sidecar"].exists():
        print("restoring %s placebo boxes from the sidecar" % class_name, flush=True)
        restore_placebos(paths["sidecar"], class_name, targets, cuts[0], cuts[1])
        applied = sum(1 for target in targets if target["placebo_status"] == "applied")
        unavailable = len(targets) - applied
    else:
        print("placing placebo boxes for %d %s targets" % (len(targets), class_name), flush=True)
        applied, unavailable = attach_placebos(targets, obstacles, args.velodyne_dir)
    stats["placebo_applied"] = applied
    stats["placebo_unavailable"] = unavailable
    stats["joint_counts"] = joint_census(targets, size_labels(cuts[0], cuts[1]))
    estimated = (applied * 13) + (unavailable * 9)
    print(
        "%s placebo applied %d  unavailable %d  estimated forwards for a fresh run %d"
        % (class_name, applied, unavailable, estimated),
        flush=True,
    )
    if not paths["sidecar"].exists():
        week7.atomic_write(
            paths["sidecar"], sidecar_text(class_name, targets, stats, cuts, args.max_targets)
        )
        confirm_sidecar(paths["sidecar"], class_name, targets, cuts[0], cuts[1])
    if args.max_targets is None:
        copy_outputs(args.output_dir, args.copy_dir)
    if args.draw_only:
        print("%s draw-only stop" % class_name, flush=True)
        return

    file_note, stamp_failures = initialize_or_check_stamps(args)
    if stamp_failures:
        raise Week11Stop("%s: %s" % (file_note, "; ".join(stamp_failures)))
    probe_unmodified(args, class_name, targets)
    file_note, stamp_failures = initialize_or_check_stamps(args)
    if stamp_failures:
        raise Week11Stop("%s after the probe: %s" % (file_note, "; ".join(stamp_failures)))

    existing = week7.read_result_rows(paths["results"])
    done = week7.finished_keys(existing)
    print("%s finished cells %d" % (class_name, len(done)), flush=True)
    pending = any(week7.target_has_pending(target, done) for target in targets)
    if pending:
        forwarded = run_forwards(
            args, class_name, targets, paths["results"], done, obstacles
        )
        print("%s forwards this process %d" % (class_name, forwarded), flush=True)
    else:
        print("%s has no pending cells" % class_name, flush=True)

    file_note, stamp_failures = initialize_or_check_stamps(args)
    finished = week7.read_result_rows(paths["results"])
    print("auditing %s" % class_name, flush=True)
    failures = list(stamp_failures)
    failures.extend(week7.audit_csv_rules(finished, targets))
    failures.extend(
        audit_f1_edits(class_name, targets, args.velodyne_dir, obstacles)
    )
    applied_rows = week7.applied_rows(finished) if not failures else []
    text = summary_text(
        class_name, ap, stats, cuts, failures, applied_rows, file_note, args.max_targets
    )
    week7.atomic_write(paths["summary"], text)
    if not failures:
        week7.write_table(paths["table"], build_table_rows(applied_rows, size_labels(cuts[0], cuts[1])))
    if args.max_targets is None:
        copy_outputs(args.output_dir, args.copy_dir)
    print(text, flush=True)
    if failures:
        raise Week11Stop("%s correctness failed" % class_name)


def main():
    """Prepare both classes, then forward whichever cells are still open."""
    args = parse_args()
    self_check_protocol()
    if args.self_check:
        print("week 11 self-check passed", flush=True)
        return
    refuse_preflight_on_official_dir(args)
    if not Path(args.val_txt).exists():
        raise FileNotFoundError("val split missing: %s" % args.val_txt)
    confirm_locked_artifacts(args)
    car_rows = read_csv_rows(args.week3_csv)
    confirm_week3_counts(car_rows, args.walked_txt)
    val_ids = load_val_ids(args.val_txt)
    if len(val_ids) != EXPECTED_FRAMES:
        raise Week11Stop("val split has %d frames, expected %d" % (len(val_ids), EXPECTED_FRAMES))
    results = load_pickle(args.result_pkl)
    infos = load_pickle(args.infos_pkl)
    result_by_id = index_by_frame_id(results, lambda item: item["frame_id"])
    info_by_id = index_by_frame_id(infos, lambda item: item["point_cloud"]["lidar_idx"])
    missing = [frame_id for frame_id in val_ids if frame_id not in result_by_id or frame_id not in info_by_id]
    if missing:
        raise Week11Stop("val frames missing from the saved predictions or infos, e.g. %s" % missing[:5])
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("week 11 classes: %s" % ", ".join(selected_classes(args)), flush=True)
    try:
        for class_name in selected_classes(args):
            run_class(args, class_name, val_ids, result_by_id, info_by_id)
    except RestartChunk as exc:
        print("chunk stop: %s" % exc, flush=True)
        print("rerun the same command to continue", flush=True)
        raise SystemExit(75)
    except (Week11Stop, week7.Week7Stop, PilotStop, ValueError) as exc:
        print("stopped: %s" % exc, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
