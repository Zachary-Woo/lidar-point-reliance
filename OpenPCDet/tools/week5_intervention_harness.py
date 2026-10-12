"""Week-5 intervention harness for one target Car detection.

Deletes a locked fraction of points inside one predicted box, re-runs the
week-1 PointPillars eval path on that in-memory cloud, and associates the
new Car prediction back to the original box by 3D IoU.

Removal and thinning are the same random delete. This script does not train,
does not edit velodyne bins, and does not overwrite result.pkl.
"""

import argparse
import copy
import csv
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _init_path  # noqa: F401

import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.datasets.kitti.kitti_object_eval_python.eval import d3_box_overlap
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, common_utils

from week3_per_detection_table import (
    CLASS_NAME,
    DETECTOR_SCORE_THRESHOLD,
    camera_boxes_from_anno,
    count_points_in_boxes,
    index_by_frame_id,
    load_pickle,
    load_velodyne,
    read_csv_rows,
)

# Locked week-5 decisions. Do not change these mid-week.
SEED = 6908
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
ASSOC_IOU = 0.5
F0_IOU_MIN = 0.99
F0_SCORE_TOL = 0.01
EXPECTED_ROWS = 33133
EXPECTED_MATCHED = 13458
EXPECTED_FRAMES = 3769
EXPECTED_AP_TOKEN = "77.2898"
MASK_DEFINITION = "points_in_boxes_cpu on the full velodyne cloud"
OPERATION_NOTE = (
    "Removal and thinning are the same random delete of interior points. "
    "Farthest-point and voxel-wise stay week 12."
)

OPENPCDET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    OPENPCDET_ROOT
    / "output/kitti_models/pointpillar/default/eval/epoch_7728/val/default/result.pkl"
)
DEFAULT_INFOS = OPENPCDET_ROOT / "data/kitti/kitti_infos_val.pkl"
DEFAULT_WEEK3 = OPENPCDET_ROOT / "output/week3/per_detection_car.csv"
DEFAULT_WALKED = OPENPCDET_ROOT / "output/week3/walked_frames.txt"
DEFAULT_WEEK4 = OPENPCDET_ROOT / "output/week4/week4_summary.txt"
DEFAULT_VELODYNE = OPENPCDET_ROOT / "data/kitti/training/velodyne"
DEFAULT_CKPT = OPENPCDET_ROOT / "checkpoints/pointpillar_7728.pth"
DEFAULT_CFG = OPENPCDET_ROOT / "tools/cfgs/kitti_models/pointpillar.yaml"
DEFAULT_DATA_ROOT = OPENPCDET_ROOT / "data/kitti"
DEFAULT_OUTPUT = OPENPCDET_ROOT / "output/week5"
DEFAULT_COPY = (
    OPENPCDET_ROOT.parent
    / "DIS Notes and Outputs"
    / "outputs by week"
    / "week5-outputs"
)

SMOKE_COLUMNS = [
    "frame_id",
    "pred_idx",
    "slot",
    "fraction",
    "score_before",
    "num_points_before",
    "num_removed",
    "num_points_after",
    "outside_unchanged",
    "score_after",
    "iou_to_original",
    "survived_iou05",
    "n_deleted_in_fov",
]


def parse_args():
    """Parse CLI paths. Defaults are inside this repository."""
    parser = argparse.ArgumentParser(description="Week 5 interior-removal harness")
    parser.add_argument("--result-pkl", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--infos-pkl", type=Path, default=DEFAULT_INFOS)
    parser.add_argument("--week3-csv", type=Path, default=DEFAULT_WEEK3)
    parser.add_argument("--walked-txt", type=Path, default=DEFAULT_WALKED)
    parser.add_argument("--week4-summary", type=Path, default=DEFAULT_WEEK4)
    parser.add_argument("--velodyne-dir", type=Path, default=DEFAULT_VELODYNE)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--cfg-file", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--copy-dir", type=Path, default=DEFAULT_COPY)
    return parser.parse_args()


def write_text(path, text):
    """Write a UTF-8 text file, creating parents if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def bool_text(flag):
    """CSV bool spelling used by the week-3 table."""
    return "true" if flag else "false"


def draw_seed(frame_id, pred_idx, fraction):
    """Stable 32-bit seed for one (frame, prediction, fraction) draw.

    The base seed is mixed with the target identity so each cell is
    reproducible and independent of the order the smoke rows are visited.
    """
    token = "%d|%s|%d|%.2f" % (SEED, frame_id, int(pred_idx), float(fraction))
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def num_to_remove(num_points, fraction):
    """How many interior points a fraction deletes. Python round, half even."""
    return int(round(float(fraction) * int(num_points)))


def interior_index(points_xyz, box_lidar):
    """Indices of full-cloud points inside one lidar box.

    Uses the week-3 helper path: points_in_boxes_cpu, no FOV filter.
    """
    box = np.asarray(box_lidar, dtype=np.float32).reshape(1, 7)
    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
        torch.from_numpy(np.ascontiguousarray(points_xyz, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(box, dtype=np.float32)),
    )
    if hasattr(point_indices, "numpy"):
        point_indices = point_indices.numpy()
    mask = np.asarray(point_indices)[0] > 0
    return np.flatnonzero(mask).astype(np.int64)


def choose_delete_indices(interior_idx, num_removed, seed):
    """Pick num_removed interior indices without replacement.

    Interior indices are sorted before the draw so the seed, not point
    storage order, decides the subset. An empty draw returns an empty array.
    """
    interior_idx = np.sort(np.asarray(interior_idx, dtype=np.int64))
    if num_removed < 0 or num_removed > int(interior_idx.size):
        raise ValueError(
            "num_removed %d is outside interior count %d" % (num_removed, interior_idx.size)
        )
    if num_removed == 0:
        return np.zeros((0,), dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(interior_idx, size=int(num_removed), replace=False)
    return np.sort(np.asarray(chosen, dtype=np.int64))


def apply_delete(points, delete_idx):
    """Return a copy of points with delete_idx rows removed. Order is kept."""
    if delete_idx.size == 0:
        return np.array(points, copy=True)
    keep = np.ones(points.shape[0], dtype=bool)
    keep[delete_idx] = False
    return points[keep]


def box_from_row(row):
    """Week-3 lidar box [x, y, z, l, w, h, yaw] for one CSV row."""
    return np.array(
        [
            row["box_x"],
            row["box_y"],
            row["box_z"],
            row["box_l"],
            row["box_w"],
            row["box_h"],
            row["box_yaw"],
        ],
        dtype=np.float32,
    )


def count_deleted_in_fov(points, delete_idx, calib, image_shape):
    """How many deleted points the eval FOV filter would have kept.

    Informational only. The removal mask stays the full-cloud interior set.
    """
    if delete_idx.size == 0:
        return 0
    pts_rect = calib.lidar_to_rect(points[delete_idx, 0:3])
    fov_flag = KittiDataset.get_fov_flag(pts_rect, image_shape, calib)
    return int(np.asarray(fov_flag).sum())


def cloud_keeps_outside(original, modified, delete_idx, interior_idx):
    """True when the edit only drops interior rows and leaves every other row.

    Comparison is exact array equality on the kept rows, in original order.
    """
    inside = np.zeros(original.shape[0], dtype=bool)
    if interior_idx.size:
        inside[interior_idx] = True
    if delete_idx.size and not bool(np.all(inside[delete_idx])):
        return False
    keep = np.ones(original.shape[0], dtype=bool)
    if delete_idx.size:
        keep[delete_idx] = False
    return bool(np.array_equal(modified, original[keep]))


def file_stamp(path):
    """Size and mtime for an unchanged-file check. Does not hash the cloud."""
    stat = Path(path).stat()
    return (int(stat.st_size), int(stat.st_mtime_ns))


def stamp_tree(paths):
    """Map path string to (size, mtime_ns) for each existing path."""
    stamped = {}
    for path in paths:
        stamped[str(path)] = file_stamp(path)
    return stamped


def velodyne_bin_paths(velodyne_dir):
    """Every .bin in the training velodyne directory, sorted by name."""
    root = Path(velodyne_dir)
    return sorted(root.glob("*.bin"))


def confirm_locked_artifacts(args):
    """Require the week-1/3/4 files. Do not rebuild them here."""
    required = [
        args.result_pkl,
        args.infos_pkl,
        args.week3_csv,
        args.week4_summary,
        args.velodyne_dir,
        args.ckpt,
        args.cfg_file,
        args.data_root,
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("required artifact missing: %s" % missing)
    summary = Path(args.week4_summary).read_text(encoding="utf-8")
    if EXPECTED_AP_TOKEN not in summary:
        raise ValueError(
            "week-4 summary does not contain %s; refusing to rescore" % EXPECTED_AP_TOKEN
        )


def confirm_week3_counts(rows, walked_txt):
    """Check the week-3 closeout counts without rebuilding the table."""
    if len(rows) != EXPECTED_ROWS:
        raise ValueError("week-3 CSV has %d rows, expected %d" % (len(rows), EXPECTED_ROWS))
    n_matched = sum(1 for row in rows if row["gt_matched"])
    if n_matched != EXPECTED_MATCHED:
        raise ValueError(
            "week-3 matched count %d, expected %d" % (n_matched, EXPECTED_MATCHED)
        )
    if Path(walked_txt).exists():
        walked = [
            line.strip()
            for line in Path(walked_txt).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(walked) != EXPECTED_FRAMES:
            raise ValueError(
                "walked_frames.txt has %d ids, expected %d" % (len(walked), EXPECTED_FRAMES)
            )


def _highest_score(cands):
    """Highest score, then smaller pred_idx, then frame_id, for a stable pick."""
    return max(cands, key=lambda row: (row["score"], -row["pred_idx"], row["frame_id"]))


def select_smoke(rows):
    """Pick the five smoke rows. Record a fallback note if a filter is empty.

    Filters are the week-5 todo. A fallback is written next to the slot; the
    filter itself is not silently rewritten.
    """
    notes = []

    near = [row for row in rows if row["range_m"] < 20.0 and row["num_points_in_box"] >= 100]
    if near:
        near_row = _highest_score(near)
    else:
        relaxed = [row for row in rows if row["range_m"] < 20.0]
        if not relaxed:
            relaxed = [row for row in rows if row["num_points_in_box"] >= 100]
        if not relaxed:
            raise ValueError("near-dense filter matched nothing and no closest row exists")
        near_row = _highest_score(relaxed)
        notes.append("near dense: filter matched nothing; picked closest high-score row")

    far = [row for row in rows if row["range_m"] >= 40.0]
    if far:
        far_row = _highest_score(far)
    else:
        far_row = max(rows, key=lambda row: (row["range_m"], row["score"]))
        notes.append("far: filter matched nothing; picked the farthest row")

    sparse = [row for row in rows if 1 <= row["num_points_in_box"] < 20]
    if sparse:
        sparse_row = _highest_score(sparse)
    else:
        positive = [row for row in rows if row["num_points_in_box"] > 0]
        if not positive:
            raise ValueError("sparse filter matched nothing and no positive point row exists")
        sparse_row = min(positive, key=lambda row: (row["num_points_in_box"], -row["score"]))
        notes.append("sparse: filter matched nothing; picked the smallest positive point count")

    empty = [row for row in rows if row["num_points_in_box"] == 0]
    if empty:
        empty_row = _highest_score(empty)
    else:
        empty_row = min(rows, key=lambda row: (row["num_points_in_box"], -row["score"]))
        notes.append("empty box: filter matched nothing; picked the smallest point count")

    mid_slice = [
        row
        for row in rows
        if row["gt_matched"]
        and 20.0 <= row["range_m"] < 40.0
        and 20 <= row["num_points_in_box"] < 100
    ]
    if mid_slice:
        median_score = float(np.median([row["score"] for row in mid_slice]))
        mid_row = min(
            mid_slice,
            key=lambda row: (abs(row["score"] - median_score), row["pred_idx"], row["frame_id"]),
        )
    else:
        matched = [row for row in rows if row["gt_matched"]]
        if not matched:
            raise ValueError("mid filter matched nothing and no matched row exists")
        mid_row = min(
            matched,
            key=lambda row: (
                0 if 20.0 <= row["range_m"] < 40.0 else min(abs(row["range_m"] - 20.0), abs(row["range_m"] - 40.0)),
                0 if 20 <= row["num_points_in_box"] < 100 else abs(row["num_points_in_box"] - 20),
                -row["score"],
            ),
        )
        notes.append("mid: filter matched nothing; picked the closest matched row")

    slots = [
        ("near_dense", near_row),
        ("far", far_row),
        ("sparse", sparse_row),
        ("empty_box", empty_row),
        ("mid", mid_row),
    ]
    return slots, notes


def sidecar_text(slots, notes):
    """Definitions and the five targets, written before any forward pass."""
    lines = [
        "Week 5 intervention sidecar",
        "seed: %d" % SEED,
        "fractions: %s" % ", ".join("%.2f" % fraction for fraction in FRACTIONS),
        "num_removed: Python round(fraction * num_points_before), half even",
        "operation: %s" % OPERATION_NOTE,
        "association: highest 3D IoU of a new Car box to the original predicted box",
        "iou helper: d3_box_overlap on KITTI camera boxes",
        "association bar: %.2f (week-3 match bar, not official AP 0.7)" % ASSOC_IOU,
        "sensor range (raw field range_m): horizontal lidar sqrt(x^2+y^2)",
        "interior mask: %s" % MASK_DEFINITION,
        "fov: n_deleted_in_fov is informational. The mask is not FOV-filtered.",
        "detector: pointpillar_7728.pth through the PointPillars evaluation path",
        "source predictions: post-processed detector outputs after score >= %.1f"
        % DETECTOR_SCORE_THRESHOLD,
        "no confidence survival threshold this week",
        "targets:",
    ]
    for slot, row in slots:
        lines.append(
            "  %s frame_id=%s pred_idx=%d score=%.4f range_m=%.4f num_points=%d gt_matched=%s"
            % (
                slot,
                row["frame_id"],
                row["pred_idx"],
                row["score"],
                row["range_m"],
                row["num_points_in_box"],
                bool_text(row["gt_matched"]),
            )
        )
    if notes:
        lines.append("filter notes:")
        for note in notes:
            lines.append("  %s" % note)
    else:
        lines.append("filter notes: every slot matched its filter")
    lines.append("")
    return "\n".join(lines)


def other_car_boxes(pred_dict, target_pred_idx):
    """Lidar boxes of Car predictions in the frame other than the target."""
    names = np.asarray(pred_dict["name"])
    boxes = np.asarray(pred_dict["boxes_lidar"], dtype=np.float32)
    if names.size == 0:
        return np.zeros((0, 7), dtype=np.float32)
    keep = []
    for pred_idx, name in enumerate(names):
        if name == CLASS_NAME and int(pred_idx) != int(target_pred_idx):
            keep.append(pred_idx)
    if not keep:
        return np.zeros((0, 7), dtype=np.float32)
    return boxes[np.asarray(keep, dtype=np.int64)]


def other_car_edit_ok(original_xyz, modified_xyz, delete_idx, other_boxes):
    """True when other-car count changes equal shared deleted points only.

    A point that sits in the target and in another box is part of the target
    delete, so that other box's count may drop by exactly those shared points.
    Any other drop means a point outside the target was removed.
    """
    if other_boxes.shape[0] == 0:
        return True
    before = count_points_in_boxes(original_xyz, other_boxes)
    after = count_points_in_boxes(modified_xyz, other_boxes)
    if delete_idx.size == 0:
        shared = np.zeros((other_boxes.shape[0],), dtype=np.int64)
    else:
        shared = count_points_in_boxes(original_xyz[delete_idx], other_boxes)
    return bool(np.array_equal(before - after, shared))


def prepare_removal(points, box, fraction, frame_id, pred_idx):
    """Delete one fraction of interior points. Returns the edited cloud and facts.

    The returned cloud is a new array. `points` is not modified.
    """
    inside = interior_index(points[:, 0:3], box)
    n_before = int(inside.size)
    n_removed = num_to_remove(n_before, fraction)
    delete_idx = choose_delete_indices(inside, n_removed, draw_seed(frame_id, pred_idx, fraction))
    modified = apply_delete(points, delete_idx)
    n_after = int(count_points_in_boxes(modified[:, 0:3], box.reshape(1, 7))[0])
    deleted_are_interior = bool(delete_idx.size == 0 or np.all(np.isin(delete_idx, inside)))
    outside_ok = cloud_keeps_outside(points, modified, delete_idx, inside)
    return {
        "modified": modified,
        "delete_idx": delete_idx,
        "interior_idx": inside,
        "num_points_before": n_before,
        "num_removed": n_removed,
        "num_points_after": n_after,
        "outside_unchanged": outside_ok,
        "deleted_are_interior": deleted_are_interior,
    }


def build_eval_data_dict(dataset, info, points):
    """Same eval item as KittiDataset.__getitem__, with an injected cloud.

    training is already False on `dataset`, so prepare_data does not augment
    and does not shuffle. FOV filtering still happens here, after the delete,
    because that is the week-1 eval order.
    """
    info = copy_info(info)
    sample_idx = info["point_cloud"]["lidar_idx"]
    img_shape = info["image"]["image_shape"]
    calib = dataset.get_calib(sample_idx)
    get_item_list = dataset.dataset_cfg.get("GET_ITEM_LIST", ["points"])
    input_dict = {
        "frame_id": sample_idx,
        "calib": calib,
    }
    if "annos" in info:
        annos = common_utils.drop_info_with_name(info["annos"], name="DontCare")
        loc = annos["location"]
        dims = annos["dimensions"]
        rots = annos["rotation_y"]
        gt_names = annos["name"]
        gt_boxes_camera = np.concatenate(
            [loc, dims, rots[..., np.newaxis]], axis=1
        ).astype(np.float32)
        gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
        input_dict.update(
            {
                "gt_names": gt_names,
                "gt_boxes": gt_boxes_lidar,
            }
        )
        if "gt_boxes2d" in get_item_list:
            input_dict["gt_boxes2d"] = annos["bbox"]
        road_plane = dataset.get_road_plane(sample_idx)
        if road_plane is not None:
            input_dict["road_plane"] = road_plane
    if "points" in get_item_list:
        cloud = np.array(points, dtype=np.float32, copy=True)
        if dataset.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(cloud[:, 0:3])
            fov_flag = dataset.get_fov_flag(pts_rect, img_shape, calib)
            cloud = cloud[fov_flag]
        input_dict["points"] = cloud
    input_dict["calib"] = calib
    data_dict = dataset.prepare_data(data_dict=input_dict)
    data_dict["image_shape"] = img_shape
    return data_dict


def copy_info(info):
    """Deep-copy one KITTI info so drop_info_with_name cannot mutate the cache."""
    return copy.deepcopy(info)


def forward_cloud(dataset, model, info, points):
    """One eval forward. Returns the OpenPCDet prediction dict for that frame."""
    data_dict = build_eval_data_dict(dataset, info, points)
    batch = dataset.collate_batch([data_dict])
    load_data_to_gpu(batch)
    with torch.no_grad():
        pred_dicts, _recall = model(batch)
    annos = dataset.generate_prediction_dicts(
        batch, pred_dicts, dataset.class_names, output_path=None
    )
    if len(annos) != 1:
        raise RuntimeError("expected one prediction dict, got %d" % len(annos))
    return annos[0]


def associate_to_original(pred_dict, original_box_lidar, calib):
    """Best new Car box against the original predicted box.

    Returns (score_after or None, iou, survived). No Car prediction yields
    score None, iou 0, survived False.
    """
    names = np.asarray(pred_dict["name"])
    if names.size == 0:
        return None, 0.0, False
    car_mask = names == CLASS_NAME
    if not np.any(car_mask):
        return None, 0.0, False
    new_cam = camera_boxes_from_anno(
        np.asarray(pred_dict["location"])[car_mask],
        np.asarray(pred_dict["dimensions"])[car_mask],
        np.asarray(pred_dict["rotation_y"])[car_mask],
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
    scores = np.asarray(pred_dict["score"], dtype=np.float64)[car_mask]
    return float(scores[best]), iou, bool(iou >= ASSOC_IOU)


def load_model(cfg_file, data_root, ckpt):
    """Build PointPillars in eval mode and load the week-1 checkpoint."""
    cfg_from_yaml_file(str(cfg_file), cfg)
    logger = common_utils.create_logger()
    dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=Path(data_root),
        training=False,
        logger=logger,
    )
    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset
    )
    model.load_params_from_file(filename=str(ckpt), logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    return dataset, model


def index_infos(infos):
    """Map zero-padded frame id to one KITTI info dict."""
    return index_by_frame_id(infos, lambda item: item["point_cloud"]["lidar_idx"])


def run_point_checks(slots, velodyne_dir, result_by_id, info_by_id):
    """Accounting checks that do not need a forward pass.

    Returns (check_lines, prepared_by_key). prepared_by_key maps
    (frame_id, pred_idx, fraction) to the removal facts and modified cloud.
    """
    checks = []
    prepared = {}
    seen_identity = {}
    for slot, row in slots:
        frame_id = row["frame_id"]
        pred_idx = int(row["pred_idx"])
        original = load_velodyne(velodyne_dir, frame_id)
        box = box_from_row(row)
        info = info_by_id[frame_id]
        image_shape = info["image"]["image_shape"]
        other_boxes = other_car_boxes(result_by_id[frame_id], pred_idx)
        for fraction in FRACTIONS:
            facts = prepare_removal(original, box, fraction, frame_id, pred_idx)
            key = (frame_id, pred_idx, float(fraction))
            prepared[key] = {
                "slot": slot,
                "row": row,
                "original": original,
                "box": box,
                "other_boxes": other_boxes,
                "image_shape": image_shape,
                **facts,
            }
            prefix = "%s %s pred %d f=%.2f" % (slot, frame_id, pred_idx, fraction)
            if facts["num_points_before"] != int(row["num_points_in_box"]):
                checks.append(
                    (
                        False,
                        "%s num_points_before %d != csv %d"
                        % (prefix, facts["num_points_before"], int(row["num_points_in_box"])),
                    )
                )
            else:
                checks.append((True, "%s num_points_before matches week-3 CSV" % prefix))
            if facts["num_points_after"] != facts["num_points_before"] - facts["num_removed"]:
                checks.append((False, "%s interior count after delete is wrong" % prefix))
            else:
                checks.append((True, "%s num_points_after = before - removed" % prefix))
            if not facts["deleted_are_interior"]:
                checks.append((False, "%s deleted a point outside the target box" % prefix))
            else:
                checks.append((True, "%s deleted indices are inside the target" % prefix))
            if not facts["outside_unchanged"]:
                checks.append((False, "%s outside points changed" % prefix))
            else:
                checks.append((True, "%s outside points unchanged" % prefix))
            if not other_car_edit_ok(
                original[:, 0:3],
                facts["modified"][:, 0:3],
                facts["delete_idx"],
                other_boxes,
            ):
                checks.append((False, "%s other-car interior counts changed beyond shared points" % prefix))
            else:
                checks.append((True, "%s other cars change only by shared target points" % prefix))
            if float(fraction) == 0.0:
                identical = np.array_equal(facts["modified"], original) and facts["num_removed"] == 0
                checks.append(
                    (identical, "%s f=0 cloud identical and num_removed=0" % prefix)
                    if identical
                    else (False, "%s f=0 cloud was modified" % prefix)
                )
            if float(fraction) == 1.0 and facts["num_points_before"] > 0:
                emptied = facts["num_points_after"] == 0
                checks.append(
                    (emptied, "%s f=1 interior count is 0" % prefix)
                    if emptied
                    else (False, "%s f=1 left interior points" % prefix)
                )
            if slot == "empty_box":
                unchanged = np.array_equal(facts["modified"], original) and facts["num_removed"] == 0
                checks.append(
                    (unchanged, "%s empty box left the cloud unchanged" % prefix)
                    if unchanged
                    else (False, "%s empty box changed the cloud" % prefix)
                )
        # Determinism on the first non-empty fraction that removes something,
        # otherwise on 0.50 of this slot. The todo asks for one pair; we check
        # every slot at 0.50 so a single unlucky row cannot hide a bug.
        first = prepare_removal(original, box, 0.5, frame_id, pred_idx)
        second = prepare_removal(original, box, 0.5, frame_id, pred_idx)
        same = np.array_equal(first["delete_idx"], second["delete_idx"])
        checks.append(
            (same, "%s %s pred %d f=0.50 delete indices match on a second draw" % (slot, frame_id, pred_idx))
            if same
            else (False, "%s %s pred %d f=0.50 delete indices differed" % (slot, frame_id, pred_idx))
        )
        identity = (frame_id, pred_idx)
        if identity in seen_identity and seen_identity[identity] != slot:
            checks.append(
                (
                    True,
                    "note: %s and %s selected the same detection; both are still checked"
                    % (seen_identity[identity], slot),
                )
            )
        seen_identity[identity] = slot
    return checks, prepared


def score_text(score):
    """Empty string when the target did not produce a Car prediction."""
    if score is None:
        return ""
    return "%.6f" % float(score)


def run_forwards(dataset, model, prepared, info_by_id):
    """Run one forward per prepared cell and attach association fields."""
    checks = []
    rows_out = []
    # Cache calib per frame. get_calib reads the calib file; that file is not edited.
    calib_by_frame = {}
    near_dense_f1_done = False
    for key in sorted(prepared.keys(), key=lambda item: (item[0], item[1], item[2])):
        facts = prepared[key]
        frame_id, pred_idx, fraction = key
        slot = facts["slot"]
        info = info_by_id[frame_id]
        if frame_id not in calib_by_frame:
            calib_by_frame[frame_id] = dataset.get_calib(info["point_cloud"]["lidar_idx"])
        calib = calib_by_frame[frame_id]
        n_fov = count_deleted_in_fov(
            facts["original"], facts["delete_idx"], calib, facts["image_shape"]
        )
        pred_dict = forward_cloud(dataset, model, info, facts["modified"])
        score_after, iou, survived = associate_to_original(pred_dict, facts["box"], calib)
        prefix = "%s %s pred %d f=%.2f" % (slot, frame_id, pred_idx, fraction)
        if slot == "near_dense" and float(fraction) == 1.0:
            near_dense_f1_done = True
            checks.append((True, "%s f=1 forward returned" % prefix))
        if float(fraction) == 0.0:
            if score_after is None:
                checks.append((False, "%s f=0 forward produced no Car" % prefix))
            else:
                iou_ok = iou >= F0_IOU_MIN
                score_ok = abs(score_after - float(facts["row"]["score"])) < F0_SCORE_TOL
                checks.append(
                    (
                        iou_ok and score_ok,
                        "%s f=0 iou=%.4f score_after=%.4f score_before=%.4f"
                        % (prefix, iou, score_after, float(facts["row"]["score"])),
                    )
                )
        rows_out.append(
            {
                "frame_id": frame_id,
                "pred_idx": pred_idx,
                "slot": slot,
                "fraction": "%.2f" % fraction,
                "score_before": "%.6f" % float(facts["row"]["score"]),
                "num_points_before": facts["num_points_before"],
                "num_removed": facts["num_removed"],
                "num_points_after": facts["num_points_after"],
                "outside_unchanged": bool_text(facts["outside_unchanged"]),
                "score_after": score_text(score_after),
                "iou_to_original": "%.6f" % iou,
                "survived_iou05": bool_text(survived),
                "n_deleted_in_fov": n_fov,
            }
        )
    if not near_dense_f1_done:
        checks.append((False, "near-dense f=1 forward did not run"))
    return checks, rows_out


def write_smoke_csv(path, rows):
    """Write the smoke table with the locked column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SMOKE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_checks(checks):
    """One line per check, PASS or FAIL, plus a final count."""
    lines = ["Week 5 correctness checks"]
    n_fail = 0
    for passed, message in checks:
        if passed:
            lines.append("PASS %s" % message)
        else:
            n_fail += 1
            lines.append("FAIL %s" % message)
    lines.append("failed: %d" % n_fail)
    lines.append("passed: %d" % (len(checks) - n_fail))
    lines.append("")
    return "\n".join(lines), n_fail


def copy_outputs(output_dir, copy_dir):
    """Copy the smoke table, sidecar, and check log into the notes folder."""
    copy_dir = Path(copy_dir)
    if not copy_dir.parent.exists():
        print("skip copy; parent missing: %s" % copy_dir.parent)
        return
    copy_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "week5_smoke.csv",
        "week5_sidecar.txt",
        "week5_check_log.txt",
    ):
        src = Path(output_dir) / name
        if src.exists():
            shutil.copy2(str(src), str(copy_dir / name))
    print("copied outputs to %s" % copy_dir)


def main():
    """Select five targets, check the delete, forward each fraction, write the log."""
    args = parse_args()
    confirm_locked_artifacts(args)
    rows = read_csv_rows(args.week3_csv)
    confirm_week3_counts(rows, args.walked_txt)

    infos = load_pickle(args.infos_pkl)
    results = load_pickle(args.result_pkl)
    info_by_id = index_infos(infos)
    result_by_id = index_by_frame_id(results, lambda item: item["frame_id"])

    slots, notes = select_smoke(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = output_dir / "week5_sidecar.txt"
    write_text(sidecar_path, sidecar_text(slots, notes))
    print(sidecar_path.read_text(encoding="utf-8"))

    before_bins = stamp_tree(velodyne_bin_paths(args.velodyne_dir))
    before_result = file_stamp(args.result_pkl)

    point_checks, prepared = run_point_checks(
        slots, args.velodyne_dir, result_by_id, info_by_id
    )
    point_log, point_fail = format_checks(point_checks)
    if point_fail:
        write_text(output_dir / "week5_check_log.txt", point_log)
        copy_outputs(output_dir, args.copy_dir)
        print(point_log)
        raise SystemExit(1)

    dataset, model = load_model(args.cfg_file, args.data_root, args.ckpt)
    forward_checks, smoke_rows = run_forwards(dataset, model, prepared, info_by_id)

    after_bins = stamp_tree(velodyne_bin_paths(args.velodyne_dir))
    after_result = file_stamp(args.result_pkl)
    bins_ok = before_bins == after_bins
    result_ok = before_result == after_result
    file_checks = [
        (
            bins_ok,
            "training/velodyne bin size and mtime unchanged (%d files)" % len(before_bins),
        ),
        (
            result_ok,
            "result.pkl size and mtime unchanged",
        ),
    ]
    if not bins_ok:
        file_checks[0] = (False, "training/velodyne bin size or mtime changed")
    if not result_ok:
        file_checks[1] = (False, "result.pkl size or mtime changed")

    all_checks = point_checks + forward_checks + file_checks
    check_log, n_fail = format_checks(all_checks)
    write_smoke_csv(output_dir / "week5_smoke.csv", smoke_rows)
    write_text(output_dir / "week5_check_log.txt", check_log)
    copy_outputs(output_dir, args.copy_dir)
    print(check_log)
    if n_fail:
        raise SystemExit(1)
    print("wrote", output_dir / "week5_smoke.csv")


if __name__ == "__main__":
    main()
