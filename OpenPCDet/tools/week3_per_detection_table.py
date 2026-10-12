"""Build the week-3 per-detection metadata table for KITTI val Cars.

Reads the locked week-1 PointPillars result.pkl, walks every id in
ImageSets/val.txt, and writes one row per Car prediction with range,
interior point count, box geometry, and 3D-IoU match metadata.

This script does not train, does not change detector weights, and does
not regenerate predictions unless result.pkl is missing (it should not be).
"""

import argparse
import csv
import pickle
import shutil
import statistics
from pathlib import Path

import numpy as np
import torch
from pcdet.datasets.kitti.kitti_object_eval_python.eval import d3_box_overlap
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

# Locked week-3 decisions. Do not change these mid-week.
CLASS_NAME = "Car"
RANGE_DEFINITION = "horizontal_lidar_xy"
# range_m = sqrt(box_x^2 + box_y^2) in OpenPCDet lidar coordinates.
MATCH_IOU_THRESH = 0.5
DETECTOR_SCORE_THRESHOLD = 0.1
# Official KITTI 3D Car AP uses 0.7; week 3 asked for a 0.5 match flag.
# Raw iou_3d is stored so week 4 can rematch at 0.7 without rebuilding.
MATCH_RULE = "independent_best_iou"
# Each prediction takes the single highest-IoU Car GT. Predictions may share a GT.
# Unmatched rows store iou_3d = 0 and an empty gt_idx.

OPENPCDET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    OPENPCDET_ROOT
    / "output/kitti_models/pointpillar/default/eval/epoch_7728/val/default/result.pkl"
)
DEFAULT_INFOS = OPENPCDET_ROOT / "data/kitti/kitti_infos_val.pkl"
DEFAULT_VAL_TXT = OPENPCDET_ROOT / "data/kitti/ImageSets/val.txt"
DEFAULT_VELODYNE = OPENPCDET_ROOT / "data/kitti/training/velodyne"
DEFAULT_OUTPUT = OPENPCDET_ROOT / "output/week3"
DEFAULT_COPY = (
    OPENPCDET_ROOT.parent
    / "DIS Notes and Outputs"
    / "outputs by week"
    / "week3-outputs"
)
COPY_NAMES = (
    "per_detection_car_summary.txt",
    "per_detection_car_definitions.txt",
    "per_detection_car_sample.csv",
)

TABLE_COLUMNS = [
    "frame_id",
    "pred_idx",
    "score",
    "box_x",
    "box_y",
    "box_z",
    "box_l",
    "box_w",
    "box_h",
    "box_yaw",
    "range_m",
    "num_points_in_box",
    "gt_matched",
    "gt_idx",
    "iou_3d",
]

# Near, high-confidence Cars should contain LiDAR points if the cloud loaded.
NEAR_RANGE_M = 15.0
HIGH_SCORE = 0.7


def parse_args():
    """Parse CLI paths and the optional frame-limit used for local backtests."""
    parser = argparse.ArgumentParser(
        description="Week 3 KITTI val per-detection metadata table"
    )
    parser.add_argument("--result-pkl", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--infos-pkl", type=Path, default=DEFAULT_INFOS)
    parser.add_argument("--val-txt", type=Path, default=DEFAULT_VAL_TXT)
    parser.add_argument("--velodyne-dir", type=Path, default=DEFAULT_VELODYNE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--copy-dir", type=Path, default=DEFAULT_COPY)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, walk only the first N val frames (backtest).",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=MATCH_IOU_THRESH,
        help="Minimum 3D IoU to set gt_matched true.",
    )
    return parser.parse_args()


def load_val_ids(val_txt_path):
    """Return KITTI sample ids from val.txt, skipping blank lines."""
    text = Path(val_txt_path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def load_pickle(path):
    """Load a pickle file and return the stored object."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def index_by_frame_id(items, frame_id_fn):
    """Build a dict keyed by zero-padded KITTI frame id strings."""
    indexed = {}
    for item in items:
        frame_id = str(frame_id_fn(item))
        if frame_id in indexed:
            raise ValueError("duplicate frame_id in source: %s" % frame_id)
        indexed[frame_id] = item
    return indexed


def load_velodyne(velodyne_dir, frame_id):
    """Load a KITTI velodyne bin as (M, 4) float32 xyz + intensity."""
    lidar_path = Path(velodyne_dir) / ("%s.bin" % frame_id)
    if not lidar_path.exists():
        raise FileNotFoundError("missing velodyne cloud: %s" % lidar_path)
    points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("unexpected velodyne shape for %s: %s" % (frame_id, points.shape))
    return points


def horizontal_range_m(boxes_lidar):
    """Return sqrt(x^2 + y^2) for each lidar box center."""
    return np.sqrt(np.square(boxes_lidar[:, 0]) + np.square(boxes_lidar[:, 1]))


def camera_boxes_from_anno(location, dimensions, rotation_y):
    """Stack KITTI camera boxes as (N, 7) [x, y, z, l, h, w, ry]."""
    location = np.asarray(location, dtype=np.float64)
    dimensions = np.asarray(dimensions, dtype=np.float64)
    rotation_y = np.asarray(rotation_y, dtype=np.float64)
    if location.size == 0:
        return np.zeros((0, 7), dtype=np.float64)
    return np.concatenate(
        [location.reshape(-1, 3), dimensions.reshape(-1, 3), rotation_y.reshape(-1, 1)],
        axis=1,
    )


def non_dontcare_gt(annos):
    """Return Car-filtered GT aligned to gt_boxes_lidar (DontCare already dropped).

    OpenPCDet stores DontCare only in the label-length arrays. gt_boxes_lidar
    has one row per non-DontCare object, matching names[:n_boxes].
    """
    names = np.asarray(annos["name"])
    gt_boxes_lidar = np.asarray(annos["gt_boxes_lidar"])
    n_boxes = int(gt_boxes_lidar.shape[0])
    names_aligned = names[:n_boxes]
    location = np.asarray(annos["location"])[:n_boxes]
    dimensions = np.asarray(annos["dimensions"])[:n_boxes]
    rotation_y = np.asarray(annos["rotation_y"])[:n_boxes]
    car_mask = names_aligned == CLASS_NAME
    car_indices = np.nonzero(car_mask)[0]
    return {
        "gt_idx": car_indices.astype(np.int64),
        "boxes_lidar": gt_boxes_lidar[car_mask],
        "cam_boxes": camera_boxes_from_anno(
            location[car_mask], dimensions[car_mask], rotation_y[car_mask]
        ),
    }


def count_points_in_boxes(points_xyz, boxes_lidar):
    """Count velodyne points inside each (N, 7) lidar box via OpenPCDet CPU helper."""
    if boxes_lidar.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
        torch.from_numpy(np.ascontiguousarray(points_xyz, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(boxes_lidar, dtype=np.float32)),
    )
    if hasattr(point_indices, "numpy"):
        point_indices = point_indices.numpy()
    counts = (np.asarray(point_indices) > 0).sum(axis=1)
    return counts.astype(np.int64)


def match_preds_to_gt(pred_cam_boxes, gt_cam_boxes, gt_idx_lookup, iou_thresh):
    """Match each prediction to at most one GT by independent best 3D IoU.

    Args:
        pred_cam_boxes: (N, 7) KITTI camera boxes for Car predictions.
        gt_cam_boxes: (K, 7) KITTI camera boxes for Car GT.
        gt_idx_lookup: (K,) indices into the frame non-DontCare GT list.
        iou_thresh: minimum IoU to mark a row matched.

    Returns:
        matched: (N,) bool
        gt_idx: (N,) int64, -1 if unmatched
        iou_3d: (N,) float64, 0.0 if unmatched
    """
    n_pred = int(pred_cam_boxes.shape[0])
    matched = np.zeros((n_pred,), dtype=bool)
    gt_idx = np.full((n_pred,), -1, dtype=np.int64)
    iou_3d = np.zeros((n_pred,), dtype=np.float64)
    if n_pred == 0 or gt_cam_boxes.shape[0] == 0:
        return matched, gt_idx, iou_3d

    overlaps = d3_box_overlap(
        np.ascontiguousarray(pred_cam_boxes, dtype=np.float64),
        np.ascontiguousarray(gt_cam_boxes, dtype=np.float64),
    )
    overlaps = np.asarray(overlaps, dtype=np.float64)
    best_local = np.argmax(overlaps, axis=1)
    best_iou = overlaps[np.arange(n_pred), best_local]
    is_match = best_iou >= iou_thresh
    matched[is_match] = True
    gt_idx[is_match] = gt_idx_lookup[best_local[is_match]]
    iou_3d[is_match] = best_iou[is_match]
    return matched, gt_idx, iou_3d


def process_frame(frame_id, pred_dict, info, velodyne_dir, iou_thresh):
    """Walk one val frame and return Car-prediction rows plus coverage metadata."""
    names = np.asarray(pred_dict["name"])
    if names.size == 0:
        car_mask = np.zeros((0,), dtype=bool)
    else:
        car_mask = names == CLASS_NAME
    car_pred_idx = np.nonzero(car_mask)[0]

    gt = non_dontcare_gt(info["annos"])
    rows = []
    if car_pred_idx.size == 0:
        return rows

    boxes_lidar = np.asarray(pred_dict["boxes_lidar"], dtype=np.float64)[car_pred_idx]
    scores = np.asarray(pred_dict["score"], dtype=np.float64)[car_pred_idx]
    pred_cam = camera_boxes_from_anno(
        np.asarray(pred_dict["location"])[car_pred_idx],
        np.asarray(pred_dict["dimensions"])[car_pred_idx],
        np.asarray(pred_dict["rotation_y"])[car_pred_idx],
    )
    points = load_velodyne(velodyne_dir, frame_id)
    point_counts = count_points_in_boxes(points[:, 0:3], boxes_lidar)
    ranges = horizontal_range_m(boxes_lidar)
    matched, gt_idx, iou_3d = match_preds_to_gt(
        pred_cam, gt["cam_boxes"], gt["gt_idx"], iou_thresh
    )

    for local_i, pred_idx in enumerate(car_pred_idx):
        box = boxes_lidar[local_i]
        row = {
            "frame_id": frame_id,
            "pred_idx": int(pred_idx),
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
        rows.append(row)
    return rows


def write_csv(path, rows):
    """Write the per-detection table with a stable column order."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path, text):
    """Write a UTF-8 text file, creating parents if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_outputs(output_dir, copy_dir):
    """Copy the week-3 summary files into the notes folder.

    The full per-detection CSV and the walked-frame list stay under
    OpenPCDet/output. This copy is the short set kept with the weekly notes.
    """
    copy_dir = Path(copy_dir)
    if not copy_dir.parent.exists():
        print("skip copy; parent missing: %s" % copy_dir.parent)
        return
    copy_dir.mkdir(parents=True, exist_ok=True)
    for name in COPY_NAMES:
        src = Path(output_dir) / name
        if src.exists():
            shutil.copy2(str(src), str(copy_dir / name))
    print("copied outputs to %s" % copy_dir)


def summarize(rows, walked_ids, iou_thresh):
    """Build the printed week-3 summary (counts plus range/point quartiles)."""
    n_rows = len(rows)
    n_matched = sum(1 for row in rows if row["gt_matched"])
    n_unmatched = n_rows - n_matched
    ranges = [row["range_m"] for row in rows]
    points = [row["num_points_in_box"] for row in rows]

    def min_med_max(values):
        if not values:
            return "empty"
        return "min=%.4f median=%.4f max=%.4f" % (
            min(values),
            statistics.median(values),
            max(values),
        )

    lines = [
        "Week 3 per-detection summary",
        "class: %s" % CLASS_NAME,
        "frames walked: %d" % len(walked_ids),
        "car predictions: %d" % n_rows,
        "matched (3D IoU >= %.2f): %d" % (iou_thresh, n_matched),
        "unmatched: %d" % n_unmatched,
        "sensor range in meters (%s): %s"
        % (RANGE_DEFINITION, min_med_max(ranges)),
        "interior points in predicted box: %s"
        % min_med_max([float(v) for v in points]),
        "match_rule: %s" % MATCH_RULE,
        "iou function: OpenPCDet d3_box_overlap (KITTI camera boxes, CPU rotate_iou)",
        "result.pkl: week-1 PointPillars val, Car moderate 3D AP@R11 = 77.2898",
        "prediction rows: PointPillars outputs after score >= %.1f"
        % DETECTOR_SCORE_THRESHOLD,
        "do not bin these values this week",
    ]
    return "\n".join(lines) + "\n"


def definitions_text(iou_thresh):
    """One-page lock file stored next to the table."""
    return "\n".join(
        [
            "Week 3 table definitions",
            "range_m: horizontal Euclidean distance of the predicted box center",
            "  from the lidar origin, sqrt(box_x^2 + box_y^2), lidar frame.",
            "  Not 3D Euclidean. Not camera Z.",
            "num_points_in_box: count of velodyne points inside the predicted box",
            "  using pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils.points_in_boxes_cpu.",
            "  Full cloud, no FOV filter.",
            "gt_matched / iou_3d: independent best 3D IoU to a GT Car.",
            "  Function: d3_box_overlap on KITTI camera boxes [x,y,z,l,h,w,ry].",
            "  Threshold: %.2f. Unmatched rows have iou_3d=0 and empty gt_idx." % iou_thresh,
            "  DontCare is ignored. Van / Pedestrian / Cyclist GT are not match targets.",
            "pred_idx: index into that frame's full detector output (all classes).",
            "gt_idx: index into that frame's non-DontCare GT list (gt_boxes_lidar rows).",
            "boxes: OpenPCDet boxes_lidar order [x,y,z,l,w,h,yaw].",
            "split: KITTI val only, 3769 frames from ImageSets/val.txt.",
            "detector: public PointPillars checkpoint pointpillar_7728.pth.",
            "prediction rows: post-processed detector outputs after score >= %.1f."
            % DETECTOR_SCORE_THRESHOLD,
            "",
        ]
    )


def read_csv_rows(path):
    """Reload the written table for independent validation."""
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                {
                    "frame_id": raw["frame_id"],
                    "pred_idx": int(raw["pred_idx"]),
                    "score": float(raw["score"]),
                    "box_x": float(raw["box_x"]),
                    "box_y": float(raw["box_y"]),
                    "box_z": float(raw["box_z"]),
                    "box_l": float(raw["box_l"]),
                    "box_w": float(raw["box_w"]),
                    "box_h": float(raw["box_h"]),
                    "box_yaw": float(raw["box_yaw"]),
                    "range_m": float(raw["range_m"]),
                    "num_points_in_box": int(raw["num_points_in_box"]),
                    "gt_matched": raw["gt_matched"].strip().lower() == "true",
                    "gt_idx": raw["gt_idx"].strip(),
                    "iou_3d": float(raw["iou_3d"]),
                }
            )
    return rows


def validate_table(rows, walked_ids, val_ids, expected_car_preds, iou_thresh):
    """Run the week-3 pass criteria. Raise ValueError on the first hard failure."""
    walked_set = set(walked_ids)
    missing = [frame_id for frame_id in val_ids if frame_id not in walked_set]
    if missing:
        raise ValueError("walk dropped %d val frames, e.g. %s" % (len(missing), missing[:5]))
    extra = [frame_id for frame_id in walked_ids if frame_id not in set(val_ids)]
    if extra:
        raise ValueError("walk included non-val frames, e.g. %s" % extra[:5])

    if expected_car_preds is not None and len(rows) != expected_car_preds:
        raise ValueError(
            "row count %d != Car predictions in result.pkl %d"
            % (len(rows), expected_car_preds)
        )

    for row in rows:
        if row["num_points_in_box"] < 0:
            raise ValueError("negative num_points_in_box at %s" % row["frame_id"])
        if row["gt_matched"]:
            if not (0.0 < row["iou_3d"] <= 1.0 + 1e-6):
                raise ValueError(
                    "matched iou_3d out of (0, 1] at %s pred %s: %s"
                    % (row["frame_id"], row["pred_idx"], row["iou_3d"])
                )
            if row["gt_idx"] == "":
                raise ValueError("matched row missing gt_idx at %s" % row["frame_id"])
        else:
            if row["gt_idx"] != "":
                raise ValueError("unmatched row has gt_idx at %s" % row["frame_id"])
            if abs(row["iou_3d"]) > 1e-12:
                raise ValueError(
                    "unmatched iou_3d must be 0 at %s pred %s: %s"
                    % (row["frame_id"], row["pred_idx"], row["iou_3d"])
                )

    near_high = [
        row
        for row in rows
        if row["score"] >= HIGH_SCORE and row["range_m"] <= NEAR_RANGE_M
    ]
    zero_near_high = [row for row in near_high if row["num_points_in_box"] == 0]
    if zero_near_high:
        sample = zero_near_high[0]
        raise ValueError(
            "high-confidence near Car has 0 interior points; cloud or box frame is wrong. "
            "example frame=%s pred_idx=%s score=%.3f range_m=%.3f"
            % (sample["frame_id"], sample["pred_idx"], sample["score"], sample["range_m"])
        )

    return {
        "frames_walked": len(walked_ids),
        "rows": len(rows),
        "matched": sum(1 for row in rows if row["gt_matched"]),
        "near_high_checked": len(near_high),
    }


def count_car_predictions(result_by_id, val_ids):
    """Count Car predictions in result.pkl over the walked val ids."""
    total = 0
    for frame_id in val_ids:
        names = np.asarray(result_by_id[frame_id]["name"])
        if names.size == 0:
            continue
        total += int((names == CLASS_NAME).sum())
    return total


def main():
    """Load artifacts, walk val, write the table, then validate the written files."""
    args = parse_args()
    for required in (args.result_pkl, args.infos_pkl, args.val_txt, args.velodyne_dir):
        if not Path(required).exists():
            raise FileNotFoundError("required week-1 artifact missing: %s" % required)

    val_ids = load_val_ids(args.val_txt)
    if args.limit > 0:
        val_ids = val_ids[: args.limit]

    results = load_pickle(args.result_pkl)
    infos = load_pickle(args.infos_pkl)
    result_by_id = index_by_frame_id(results, lambda item: item["frame_id"])
    info_by_id = index_by_frame_id(infos, lambda item: item["point_cloud"]["lidar_idx"])

    missing_results = [frame_id for frame_id in val_ids if frame_id not in result_by_id]
    missing_infos = [frame_id for frame_id in val_ids if frame_id not in info_by_id]
    if missing_results:
        raise FileNotFoundError("result.pkl missing val frames: %s" % missing_results[:5])
    if missing_infos:
        raise FileNotFoundError("kitti_infos_val.pkl missing val frames: %s" % missing_infos[:5])

    expected_car_preds = count_car_predictions(result_by_id, val_ids)
    print("walking %d val frames, expected Car rows=%d" % (len(val_ids), expected_car_preds))

    rows = []
    walked_ids = []
    for i, frame_id in enumerate(val_ids):
        walked_ids.append(frame_id)
        frame_rows = process_frame(
            frame_id,
            result_by_id[frame_id],
            info_by_id[frame_id],
            args.velodyne_dir,
            args.iou_thresh,
        )
        rows.extend(frame_rows)
        if (i + 1) % 200 == 0 or (i + 1) == len(val_ids):
            print("processed %d/%d frames, rows=%d" % (i + 1, len(val_ids), len(rows)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "per_detection_car.csv"
    summary_path = output_dir / "per_detection_car_summary.txt"
    definitions_path = output_dir / "per_detection_car_definitions.txt"
    walked_path = output_dir / "walked_frames.txt"
    sample_path = output_dir / "per_detection_car_sample.csv"

    write_csv(table_path, rows)
    write_csv(sample_path, rows[:50])
    write_text(summary_path, summarize(rows, walked_ids, args.iou_thresh))
    write_text(definitions_path, definitions_text(args.iou_thresh))
    write_text(walked_path, "\n".join(walked_ids) + "\n")

    reloaded = read_csv_rows(table_path)
    report = validate_table(reloaded, walked_ids, val_ids, expected_car_preds, args.iou_thresh)
    copy_outputs(output_dir, args.copy_dir)
    print(summary_path.read_text(encoding="utf-8"))
    print("validation passed:", report)
    print("wrote", table_path)


if __name__ == "__main__":
    main()
