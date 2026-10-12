"""Build the week-4 baseline stratification for KITTI val Cars.

Uses the locked week-3 per-detection table and the same week-1 result.pkl.
AP is official KITTI 3D AP with GTs outside a bin ignored. Confidence and
ECE are computed on prediction-side bins. This script does not train, does
not change detector weights, and does not drop points.
"""

import argparse
import copy
import csv
import math
import re
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from pcdet.datasets.kitti.kitti_object_eval_python.eval import get_official_eval_result

from week3_per_detection_table import (
    CLASS_NAME,
    DETECTOR_SCORE_THRESHOLD,
    count_points_in_boxes,
    horizontal_range_m,
    index_by_frame_id,
    load_pickle,
    load_val_ids,
    load_velodyne,
    read_csv_rows,
)

EXPECTED_UNSTRAT_MOD_R11 = 77.2898
EXPECTED_PRED_ROWS = 33133
AP_IOU = 0.7
MIN_GT_PER_BIN = 100
SCORE_THRESHOLD = DETECTOR_SCORE_THRESHOLD

OPENPCDET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    OPENPCDET_ROOT
    / "output/kitti_models/pointpillar/default/eval/epoch_7728/val/default/result.pkl"
)
DEFAULT_INFOS = OPENPCDET_ROOT / "data/kitti/kitti_infos_val.pkl"
DEFAULT_VAL_TXT = OPENPCDET_ROOT / "data/kitti/ImageSets/val.txt"
DEFAULT_VELODYNE = OPENPCDET_ROOT / "data/kitti/training/velodyne"
DEFAULT_WEEK3 = OPENPCDET_ROOT / "output/week3/per_detection_car.csv"
DEFAULT_OUTPUT = OPENPCDET_ROOT / "output/week4"
DEFAULT_COPY = (
    OPENPCDET_ROOT.parent
    / "DIS Notes and Outputs"
    / "outputs by week"
    / "week4-outputs"
)
COPY_NAMES = (
    "ap_by_bin.csv",
    "confidence_ece_by_bin.csv",
    "unstratified_eval.txt",
    "week4_definitions.txt",
    "week4_summary.txt",
)
IMAGE_NAMES = (
    "reliability_range.png",
    "reliability_num_points.png",
    "reliability_size.png",
    "score_dist_range.png",
    "score_dist_num_points.png",
    "score_dist_size.png",
)

# Locked week-4 bins. Overflow range_m >= 80 stays in the last range bin.
AXES = (
    {
        "name": "range_m",
        "labels": ("[0, 20)", "[20, 40)", "[40, 80]"),
        "assign": lambda value: 0 if value < 20.0 else (1 if value < 40.0 else 2),
        "ap_order": "decreasing",
    },
    {
        "name": "num_points",
        "labels": ("[0, 20)", "[20, 100)", "[100, +inf)"),
        "assign": lambda value: 0 if value < 20 else (1 if value < 100 else 2),
        "ap_order": "increasing",
    },
    {
        "name": "size_m3",
        "labels": ("[0, 8)", "[8, 12)", "[12, +inf)"),
        "assign": lambda value: 0 if value < 8.0 else (1 if value < 12.0 else 2),
        "ap_order": "increasing",
    },
)

CUBIC_METER_LABEL = "m" + chr(179)

PRESENTATION = {
    "range_m": {
        "axis_name": "sensor range",
        "bin_labels": ("0-20 m", "20-40 m", "40+ m"),
    },
    "num_points": {
        "axis_name": "interior point count",
        "bin_labels": ("0-19 points", "20-99 points", "100+ points"),
    },
    "size_m3": {
        "axis_name": "predicted box volume",
        "bin_labels": (
            "Under 8 %s" % CUBIC_METER_LABEL,
            "8-12 %s" % CUBIC_METER_LABEL,
            "12+ %s" % CUBIC_METER_LABEL,
        ),
    },
}

RELIABILITY_STYLES = (
    {"color": "black", "marker": "o"},
    {"color": "#0072B2", "marker": "s"},
    {"color": "#E69F00", "marker": "^"},
    {"color": "#009E73", "marker": "D"},
)

HISTOGRAM_STYLES = (
    {"color": "#0072B2"},
    {"color": "#E69F00"},
    {"color": "#009E73"},
)

CONF_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
GT_COLUMNS = [
    "frame_id",
    "gt_idx",
    "range_m",
    "num_points",
    "size_m3",
    "box_l",
    "box_w",
    "box_h",
    "range_bin",
    "num_points_bin",
    "size_bin",
]


def parse_args():
    """Parse input and output paths for the week-4 baseline."""
    parser = argparse.ArgumentParser(description="Week 4 KITTI val baseline stratification")
    parser.add_argument("--result-pkl", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--infos-pkl", type=Path, default=DEFAULT_INFOS)
    parser.add_argument("--val-txt", type=Path, default=DEFAULT_VAL_TXT)
    parser.add_argument("--velodyne-dir", type=Path, default=DEFAULT_VELODYNE)
    parser.add_argument("--week3-csv", type=Path, default=DEFAULT_WEEK3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--copy-dir", type=Path, default=DEFAULT_COPY)
    return parser.parse_args()


def size_m3(length, width, height):
    """Return box volume l*w*h in cubic meters."""
    return float(length) * float(width) * float(height)


def display_axis_name(axis_name):
    """Return the publication-facing name for one locked stratification axis."""
    if axis_name not in PRESENTATION:
        raise ValueError("unknown axis: %s" % axis_name)
    return PRESENTATION[axis_name]["axis_name"]


def display_bin_label(axis_name, bin_index):
    """Return a plain-language bin label without exposing code identifiers."""
    if axis_name not in PRESENTATION:
        raise ValueError("unknown axis: %s" % axis_name)
    labels = PRESENTATION[axis_name]["bin_labels"]
    if int(bin_index) not in range(len(labels)):
        raise ValueError("unknown bin %s for axis %s" % (bin_index, axis_name))
    return labels[int(bin_index)]


def display_bin_from_raw(axis_name, raw_label):
    """Translate a locked raw interval label to its publication-facing label."""
    axis = next((item for item in AXES if item["name"] == axis_name), None)
    if axis is None:
        raise ValueError("unknown axis: %s" % axis_name)
    try:
        bin_index = axis["labels"].index(raw_label)
    except ValueError as exc:
        raise ValueError(
            "unknown raw bin %s for axis %s" % (raw_label, axis_name)
        ) from exc
    return display_bin_label(axis_name, bin_index)


def assign_bins(range_m, num_points, volume):
    """Return the locked (range, points, size) bin indices for one box."""
    return (
        AXES[0]["assign"](float(range_m)),
        AXES[1]["assign"](int(num_points)),
        AXES[2]["assign"](float(volume)),
    )


def parse_car_3d_ap(result_str):
    """Parse Car 3D easy/moderate/hard AP@R11 and AP@R40 from official eval text."""
    r11 = re.search(
        r"Car AP@0\.70, 0\.70, 0\.70:\s*.*?3d   AP:([0-9.]+), ([0-9.]+), ([0-9.]+)",
        result_str,
        flags=re.DOTALL,
    )
    r40 = re.search(
        r"Car AP_R40@0\.70, 0\.70, 0\.70:\s*.*?3d   AP:([0-9.]+), ([0-9.]+), ([0-9.]+)",
        result_str,
        flags=re.DOTALL,
    )
    if r11 is None or r40 is None:
        raise ValueError("failed to parse Car 3D AP from eval output:\n%s" % result_str)
    return {
        "r11_easy": float(r11.group(1)),
        "r11_moderate": float(r11.group(2)),
        "r11_hard": float(r11.group(3)),
        "r40_easy": float(r40.group(1)),
        "r40_moderate": float(r40.group(2)),
        "r40_hard": float(r40.group(3)),
    }


def official_car_3d_ap(gt_annos, dt_annos):
    """Run official KITTI eval for Car and return parsed 3D AP numbers."""
    result_str, _ret = get_official_eval_result(gt_annos, dt_annos, CLASS_NAME)
    parsed = parse_car_3d_ap(result_str)
    parsed["raw"] = result_str
    return parsed


def build_gt_table(val_ids, info_by_id, velodyne_dir):
    """Walk val and build one row per GT Car with week-3 range/point/size defs."""
    rows = []
    walked = []
    for i, frame_id in enumerate(val_ids):
        walked.append(frame_id)
        annos = info_by_id[frame_id]["annos"]
        boxes = np.asarray(annos["gt_boxes_lidar"], dtype=np.float64)
        names = np.asarray(annos["name"])[: boxes.shape[0]]
        car_idx = np.nonzero(names == CLASS_NAME)[0]
        if car_idx.size == 0:
            if (i + 1) % 400 == 0 or (i + 1) == len(val_ids):
                print("gt walk %d/%d rows=%d" % (i + 1, len(val_ids), len(rows)))
            continue
        points = load_velodyne(velodyne_dir, frame_id)
        car_boxes = boxes[car_idx]
        point_counts = count_points_in_boxes(points[:, 0:3], car_boxes)
        ranges = horizontal_range_m(car_boxes)
        for local_i, gt_idx in enumerate(car_idx):
            box = car_boxes[local_i]
            volume = size_m3(box[3], box[4], box[5])
            range_bin, point_bin, size_bin = assign_bins(
                ranges[local_i], point_counts[local_i], volume
            )
            rows.append(
                {
                    "frame_id": frame_id,
                    "gt_idx": int(gt_idx),
                    "range_m": float(ranges[local_i]),
                    "num_points": int(point_counts[local_i]),
                    "size_m3": volume,
                    "box_l": float(box[3]),
                    "box_w": float(box[4]),
                    "box_h": float(box[5]),
                    "range_bin": int(range_bin),
                    "num_points_bin": int(point_bin),
                    "size_bin": int(size_bin),
                }
            )
        if (i + 1) % 400 == 0 or (i + 1) == len(val_ids):
            print("gt walk %d/%d rows=%d" % (i + 1, len(val_ids), len(rows)))
    return rows, walked


def annotate_pred_rows(pred_rows):
    """Add volume, 0.7-match flag, and locked bin indices to week-3 pred rows."""
    annotated = []
    for row in pred_rows:
        volume = size_m3(row["box_l"], row["box_w"], row["box_h"])
        range_bin, point_bin, size_bin = assign_bins(
            row["range_m"], row["num_points_in_box"], volume
        )
        new_row = dict(row)
        new_row["size_m3"] = volume
        new_row["matched_iou07"] = bool(row["iou_3d"] >= AP_IOU)
        new_row["range_bin"] = int(range_bin)
        new_row["num_points_bin"] = int(point_bin)
        new_row["size_bin"] = int(size_bin)
        annotated.append(new_row)
    return annotated


def axis_bin_field(axis_name):
    """Map an axis name to the bin-index column stored on each row."""
    if axis_name == "range_m":
        return "range_bin"
    if axis_name == "num_points":
        return "num_points_bin"
    if axis_name == "size_m3":
        return "size_bin"
    raise ValueError("unknown axis: %s" % axis_name)


def count_bins(rows, field):
    """Return how many rows fall in each of the three locked bins."""
    counts = [0, 0, 0]
    for row in rows:
        counts[int(row[field])] += 1
    return counts


def mark_cars_outside_bin(gt_annos, val_ids, gt_rows, axis_name, bin_index):
    """Copy GT annos and ignore Car objects whose GT box is outside one bin.

    Out-of-bin Cars keep class Car but get occluded=3 so official clean_data
    treats them as ignored GT at every difficulty (same role as a Van).
    """
    keep = {}
    field = axis_bin_field(axis_name)
    for row in gt_rows:
        if int(row[field]) == bin_index:
            keep[(row["frame_id"], int(row["gt_idx"]))] = True
    marked = []
    for frame_id, anno in zip(val_ids, gt_annos):
        anno = copy.deepcopy(anno)
        occluded = np.array(anno["occluded"], dtype=np.float64, copy=True)
        names = np.asarray(anno["name"])
        n_boxes = int(np.asarray(anno["gt_boxes_lidar"]).shape[0])
        for gt_idx in range(n_boxes):
            if names[gt_idx] != CLASS_NAME:
                continue
            if (frame_id, int(gt_idx)) not in keep:
                occluded[gt_idx] = 3.0
        anno["occluded"] = occluded
        marked.append(anno)
    return marked


def reliability_and_ece(scores, correct):
    """Compute ECE and reliability points, omitting empty confidence bins.

    ECE = sum_{k: n_k > 0} (n_k / N) * |acc_k - mean_score_k|
    acc_k = fraction of preds in confidence bin k with iou_3d >= 0.7
    Last bin is [0.9, 1.0] inclusive. N is the number of preds in the slice.
    """
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.bool_)
    n_total = int(scores.shape[0])
    if n_total == 0:
        return 0.0, [], [], []
    ece = 0.0
    mean_conf = []
    accuracy = []
    counts = []
    for i in range(10):
        lo = CONF_EDGES[i]
        hi = CONF_EDGES[i + 1]
        if i < 9:
            mask = (scores >= lo) & (scores < hi)
        else:
            mask = (scores >= lo) & (scores <= hi)
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        acc_k = float(correct[mask].mean())
        conf_k = float(scores[mask].mean())
        ece += (n_k / float(n_total)) * abs(acc_k - conf_k)
        mean_conf.append(conf_k)
        accuracy.append(acc_k)
        counts.append(n_k)
    return ece, mean_conf, accuracy, counts


def pred_slice_stats(rows, label):
    """Summarize confidence and ECE for one prediction-side slice."""
    scores = [row["score"] for row in rows]
    matched = [row["matched_iou07"] for row in rows]
    ece, mean_conf, accuracy, counts = reliability_and_ece(scores, matched)
    return {
        "label": label,
        "n_pred": len(rows),
        "n_matched_iou07": int(sum(1 for flag in matched if flag)),
        "mean_score": float(statistics.mean(scores)) if scores else math.nan,
        "median_score": float(statistics.median(scores)) if scores else math.nan,
        "max_score": float(max(scores)) if scores else math.nan,
        "ece": float(ece),
        "rel_conf": mean_conf,
        "rel_acc": accuracy,
        "rel_n": counts,
        "scores": scores,
    }


def write_csv(path, fieldnames, rows):
    """Write a list of dict rows to CSV."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path, text):
    """Write a UTF-8 text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_outputs(output_dir, copy_dir):
    """Copy the week-4 tables and figures into the notes folder.

    Tables and the summary stay in the week folder. Plots go in its images
    folder. The full ground-truth table stays under OpenPCDet/output.
    """
    copy_dir = Path(copy_dir)
    if not copy_dir.parent.exists():
        print("skip copy; parent missing: %s" % copy_dir.parent)
        return
    copy_dir.mkdir(parents=True, exist_ok=True)
    image_dir = copy_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for name in COPY_NAMES:
        src = Path(output_dir) / name
        if src.exists():
            shutil.copy2(str(src), str(copy_dir / name))
    for name in IMAGE_NAMES:
        src = Path(output_dir) / name
        if src.exists():
            shutil.copy2(str(src), str(image_dir / name))
    print("copied outputs to %s" % copy_dir)


def pyplot():
    """Load matplotlib only when a figure is drawn.

    Week 7 imports the bin labels from this file and does not draw. Importing
    pyplot at load time pulls an old pyparsing call inside matplotlib.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_reliability(path, series, title):
    """Save one reliability diagram with an identity line.

    Data curves are solid. Color and marker shape distinguish the strata.
    The dashed line is only the perfect-calibration reference.
    """
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot(
        [0, 1],
        [0, 1],
        color="0.55",
        linestyle="--",
        linewidth=1.0,
        label="Perfect calibration",
    )
    for index, item in enumerate(series):
        if not item["rel_conf"]:
            continue
        plot_style = RELIABILITY_STYLES[index % len(RELIABILITY_STYLES)]
        ax.plot(
            item["rel_conf"],
            item["rel_acc"],
            color=plot_style["color"],
            linestyle="-",
            marker=plot_style["marker"],
            markersize=5,
            linewidth=1.8,
            label="%s (ECE = %.3f)" % (item["label"], item["ece"]),
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean detector confidence")
    ax.set_ylabel("Accuracy (3D IoU >= 0.7)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.text(
        0.5,
        0.012,
        "PointPillars predictions after post-processing score >= %.1f"
        % SCORE_THRESHOLD,
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_score_hist(path, series, title):
    """Save normalized step histograms for the three bins of one axis.

    Each group is normalized independently to unit area. Every outline is a
    solid line; color is what distinguishes the strata.
    """
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    bins = np.linspace(0.0, 1.0, 21)
    for index, item in enumerate(series):
        if not item["scores"]:
            continue
        plot_style = HISTOGRAM_STYLES[index % len(HISTOGRAM_STYLES)]
        ax.hist(
            item["scores"],
            bins=bins,
            density=True,
            histtype="step",
            color=plot_style["color"],
            linestyle="-",
            linewidth=1.8,
            label="%s (n = %d)" % (item["label"], item["n_pred"]),
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Detector confidence")
    ax.set_ylabel("Normalized density")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.text(
        0.5,
        0.012,
        "Each curve has unit area; predictions shown after score >= %.1f"
        % SCORE_THRESHOLD,
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def definitions_text():
    """Sidecar that freezes bins, AP rule, and the ECE formula."""
    return "\n".join(
        [
            "Week 4 baseline definitions",
            "Sensor range (raw field range_m): horizontal lidar sqrt(x^2+y^2).",
            "  Bins: 0-20 m, 20-40 m, and 40 m or farther.",
            "Interior point count (raw field num_points): points_in_boxes_cpu on the full cloud.",
            "  Bins: 0-19 points, 20-99 points, and 100 or more points.",
            "Predicted box volume (raw field size_m3): l * w * h in cubic meters.",
            "  Bins: under 8, 8 to under 12, and 12 or more cubic meters.",
            "AP: official KITTI 3D AP, Car, overlap 0.7. Quote moderate AP@R11.",
            "AP slice: GT-side. Cars outside the bin are ignored (occluded=3), detections kept.",
            "Confidence / ECE slice: prediction-side week-3 rows after score >= %.1f."
            % SCORE_THRESHOLD,
            "matched_iou07: week-3 stored iou_3d >= 0.7 (not the week-3 0.5 gt_matched flag).",
            "Reliability: 10 equal-width score bins [0.0, 0.1), ..., [0.9, 1.0].",
            "accuracy_k = (preds in bin k with iou_3d >= 0.7) / n_k",
            "conf_k = mean score in bin k",
            "ECE = sum_{k: n_k>0} (n_k / N) * |accuracy_k - conf_k|",
            "Empty confidence bins are omitted. N is the number of preds in the slice.",
            "Confidence histograms: step outlines; each slice is normalized to unit area.",
            "Size AP is not assumed monotonic. Range and interior-point AP must fall toward the weak bins.",
            "Detector: pointpillar_7728.pth. result.pkl is the week-1 file.",
            "",
        ]
    )


def summarize(unstrat, ap_rows, pred_stats):
    """Build the one-page week-4 summary for the update."""
    lines = [
        "Week 4 baseline summary",
        "unstratified Car 3D AP@R11 easy/mod/hard: %.4f / %.4f / %.4f"
        % (unstrat["r11_easy"], unstrat["r11_moderate"], unstrat["r11_hard"]),
        "unstratified Car 3D AP@R40 easy/mod/hard: %.4f / %.4f / %.4f"
        % (unstrat["r40_easy"], unstrat["r40_moderate"], unstrat["r40_hard"]),
        "primary unstratified number: moderate 3D AP@R11 = %.4f (target 77.2898)"
        % unstrat["r11_moderate"],
        "",
        "AP@R11 moderate by GT bin:",
    ]
    for row in ap_rows:
        if row["axis"] == "all":
            continue
        lines.append(
            "  %s %s  n_gt=%s  AP@R11 mod=%.4f  (easy=%.4f hard=%.4f)  R40 mod=%.4f"
            % (
                display_axis_name(row["axis"]),
                display_bin_from_raw(row["axis"], row["bin"]),
                row["n_gt"],
                row["ap_r11_moderate"],
                row["ap_r11_easy"],
                row["ap_r11_hard"],
                row["ap_r40_moderate"],
            )
        )
    lines.append("")
    lines.append(
        "Prediction-side confidence / ECE after score >= %.1f (match at 3D IoU >= 0.7):"
        % SCORE_THRESHOLD
    )
    for item in pred_stats:
        lines.append(
            "  %s  n=%d  matched@0.7=%d  mean_score=%.4f  ECE=%.4f"
            % (
                item["label"],
                item["n_pred"],
                item["n_matched_iou07"],
                item["mean_score"],
                item["ece"],
            )
        )
    lines.append("no bins were merged; every GT bin had at least 100 Cars")
    lines.append(
        "Predicted box volume 12+ %s AP is below the typical-volume bin; edges were not retuned"
        % CUBIC_METER_LABEL
    )
    return "\n".join(lines) + "\n"


def validate(
    val_ids,
    gt_walked,
    gt_rows,
    pred_rows,
    unstrat,
    ap_rows,
    pred_stats,
):
    """Run the week-4 pass criteria. Raise ValueError on a hard failure."""
    if gt_walked != val_ids:
        raise ValueError("GT walk did not match val.txt order or coverage")
    if abs(unstrat["r11_moderate"] - EXPECTED_UNSTRAT_MOD_R11) > 1e-4:
        raise ValueError(
            "unstratified moderate 3D AP@R11 is %.4f, expected %.4f"
            % (unstrat["r11_moderate"], EXPECTED_UNSTRAT_MOD_R11)
        )
    if len(pred_rows) != EXPECTED_PRED_ROWS:
        raise ValueError(
            "pred rows %d != week-3 Car count %d" % (len(pred_rows), EXPECTED_PRED_ROWS)
        )
    for row in pred_rows + gt_rows:
        for field in ("range_bin", "num_points_bin", "size_bin"):
            if int(row[field]) not in (0, 1, 2):
                raise ValueError("row missing a locked bin: %s" % row)
    for item in pred_stats:
        for conf, acc in zip(item["rel_conf"], item["rel_acc"]):
            if not (0.0 <= conf <= 1.0 and 0.0 <= acc <= 1.0):
                raise ValueError("reliability point outside [0, 1]: %s %s" % (conf, acc))
    gt_counts = {
        axis["name"]: count_bins(gt_rows, axis_bin_field(axis["name"])) for axis in AXES
    }
    for axis_name, counts in gt_counts.items():
        if min(counts) < MIN_GT_PER_BIN:
            raise ValueError("GT bin under 100 for %s: %s" % (axis_name, counts))

    ap_by_axis = {}
    for row in ap_rows:
        if row["axis"] == "all":
            continue
        ap_by_axis.setdefault(row["axis"], []).append(row["ap_r11_moderate"])
    range_ap = ap_by_axis["range_m"]
    if not (range_ap[0] >= range_ap[1] >= range_ap[2]):
        raise ValueError(
            "near AP is not >= mid >= far: %s. ignore mask or range assignment is wrong"
            % range_ap
        )
    point_ap = ap_by_axis["num_points"]
    if not (point_ap[2] >= point_ap[1] >= point_ap[0]):
        raise ValueError(
            "dense AP is not >= mid >= sparse: %s. point bins or ignore mask is wrong"
            % point_ap
        )
    # Size volume is reported but not required to be monotonic. The locked
    # [12, +inf) tail is a small set of atypically large Car boxes; AP there
    # can fall even when range/density trends are correct.
    return {
        "unstrat_mod_r11": unstrat["r11_moderate"],
        "n_gt": len(gt_rows),
        "n_pred": len(pred_rows),
        "gt_bin_counts": gt_counts,
    }


def main():
    """Load week-3/week-1 artifacts, score the baseline, write tables and plots."""
    args = parse_args()
    for required in (
        args.result_pkl,
        args.infos_pkl,
        args.val_txt,
        args.velodyne_dir,
        args.week3_csv,
    ):
        if not Path(required).exists():
            raise FileNotFoundError("required artifact missing: %s" % required)

    val_ids = load_val_ids(args.val_txt)
    results = load_pickle(args.result_pkl)
    infos = load_pickle(args.infos_pkl)
    result_by_id = index_by_frame_id(results, lambda item: item["frame_id"])
    info_by_id = index_by_frame_id(infos, lambda item: item["point_cloud"]["lidar_idx"])
    dt_annos = [copy.deepcopy(result_by_id[frame_id]) for frame_id in val_ids]
    base_gt_annos = [copy.deepcopy(info_by_id[frame_id]["annos"]) for frame_id in val_ids]

    print("scoring unstratified official Car 3D AP")
    unstrat = official_car_3d_ap(base_gt_annos, dt_annos)
    print(
        "unstratified moderate 3D AP@R11 = %.4f (R40 = %.4f)"
        % (unstrat["r11_moderate"], unstrat["r40_moderate"])
    )

    print("building GT Car table")
    gt_rows, gt_walked = build_gt_table(val_ids, info_by_id, args.velodyne_dir)
    pred_rows = annotate_pred_rows(read_csv_rows(args.week3_csv))
    print("gt cars", len(gt_rows), "pred cars", len(pred_rows))
    for axis in AXES:
        field = axis_bin_field(axis["name"])
        print(
            "GT",
            axis["name"],
            list(zip(axis["labels"], count_bins(gt_rows, field))),
        )
        print(
            "pred",
            axis["name"],
            list(zip(axis["labels"], count_bins(pred_rows, field))),
        )

    ap_rows = [
        {
            "axis": "all",
            "bin": "all",
            "n_gt": len(gt_rows),
            "ap_r11_easy": unstrat["r11_easy"],
            "ap_r11_moderate": unstrat["r11_moderate"],
            "ap_r11_hard": unstrat["r11_hard"],
            "ap_r40_easy": unstrat["r40_easy"],
            "ap_r40_moderate": unstrat["r40_moderate"],
            "ap_r40_hard": unstrat["r40_hard"],
        }
    ]
    for axis in AXES:
        field = axis_bin_field(axis["name"])
        counts = count_bins(gt_rows, field)
        for bin_index, label in enumerate(axis["labels"]):
            print("scoring AP", axis["name"], label, "n_gt", counts[bin_index])
            marked = mark_cars_outside_bin(
                base_gt_annos, val_ids, gt_rows, axis["name"], bin_index
            )
            parsed = official_car_3d_ap(marked, dt_annos)
            ap_rows.append(
                {
                    "axis": axis["name"],
                    "bin": label,
                    "n_gt": counts[bin_index],
                    "ap_r11_easy": parsed["r11_easy"],
                    "ap_r11_moderate": parsed["r11_moderate"],
                    "ap_r11_hard": parsed["r11_hard"],
                    "ap_r40_easy": parsed["r40_easy"],
                    "ap_r40_moderate": parsed["r40_moderate"],
                    "ap_r40_hard": parsed["r40_hard"],
                }
            )

    pred_stats = [pred_slice_stats(pred_rows, "All Car predictions")]
    pred_table_rows = []
    for item in pred_stats:
        pred_table_rows.append(
            {
                "axis": "all",
                "bin": "all",
                "n_pred": item["n_pred"],
                "n_matched_iou07": item["n_matched_iou07"],
                "mean_score": item["mean_score"],
                "median_score": item["median_score"],
                "max_score": item["max_score"],
                "ece": item["ece"],
            }
        )
    plot_series = {"range_m": [pred_stats[0]], "num_points": [pred_stats[0]], "size_m3": [pred_stats[0]]}
    for axis in AXES:
        field = axis_bin_field(axis["name"])
        for bin_index, label in enumerate(axis["labels"]):
            sliced = [row for row in pred_rows if int(row[field]) == bin_index]
            item = pred_slice_stats(
                sliced, display_bin_label(axis["name"], bin_index)
            )
            pred_stats.append(item)
            plot_series[axis["name"]].append(item)
            pred_table_rows.append(
                {
                    "axis": axis["name"],
                    "bin": label,
                    "n_pred": item["n_pred"],
                    "n_matched_iou07": item["n_matched_iou07"],
                    "mean_score": item["mean_score"],
                    "median_score": item["median_score"],
                    "max_score": item["max_score"],
                    "ece": item["ece"],
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "gt_car.csv", GT_COLUMNS, gt_rows)
    write_csv(
        output_dir / "ap_by_bin.csv",
        [
            "axis",
            "bin",
            "n_gt",
            "ap_r11_easy",
            "ap_r11_moderate",
            "ap_r11_hard",
            "ap_r40_easy",
            "ap_r40_moderate",
            "ap_r40_hard",
        ],
        ap_rows,
    )
    write_csv(
        output_dir / "confidence_ece_by_bin.csv",
        [
            "axis",
            "bin",
            "n_pred",
            "n_matched_iou07",
            "mean_score",
            "median_score",
            "max_score",
            "ece",
        ],
        pred_table_rows,
    )

    plot_reliability(
        output_dir / "reliability_range.png",
        plot_series["range_m"],
        "Reliability by sensor range",
    )
    plot_reliability(
        output_dir / "reliability_num_points.png",
        plot_series["num_points"],
        "Reliability by interior point count",
    )
    plot_reliability(
        output_dir / "reliability_size.png",
        plot_series["size_m3"],
        "Reliability by predicted box volume",
    )
    plot_score_hist(
        output_dir / "score_dist_range.png",
        plot_series["range_m"][1:],
        "Confidence distribution by sensor range",
    )
    plot_score_hist(
        output_dir / "score_dist_num_points.png",
        plot_series["num_points"][1:],
        "Confidence distribution by interior point count",
    )
    plot_score_hist(
        output_dir / "score_dist_size.png",
        plot_series["size_m3"][1:],
        "Confidence distribution by predicted box volume",
    )

    write_text(output_dir / "week4_definitions.txt", definitions_text())
    write_text(output_dir / "week4_summary.txt", summarize(unstrat, ap_rows, pred_stats))
    write_text(output_dir / "unstratified_eval.txt", unstrat["raw"])

    report = validate(val_ids, gt_walked, gt_rows, pred_rows, unstrat, ap_rows, pred_stats)
    copy_outputs(output_dir, args.copy_dir)
    print(Path(output_dir / "week4_summary.txt").read_text(encoding="utf-8"))
    print("validation passed:", report)


if __name__ == "__main__":
    main()
