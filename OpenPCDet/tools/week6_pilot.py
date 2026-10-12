"""Week-6 pilot on the locked week-5 interior delete.

Draws 200 Car rows from the week-3 table, stratified by the week-4 range
bins, and runs one in-memory forward per row per locked fraction. Retention
counts a miss as zero. This driver does not train, does not edit velodyne
bins, does not overwrite result.pkl, and does not run the week-7 controls.
"""

import argparse
import csv
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _init_path  # noqa: F401

import numpy as np
from pcdet.models import load_data_to_gpu

from week3_per_detection_table import (
    load_pickle,
    load_velodyne,
    read_csv_rows,
)
from week4_baseline_stratification import (
    assign_bins,
    display_bin_label,
    size_m3,
)
from week5_intervention_harness import (
    ASSOC_IOU,
    EXPECTED_AP_TOKEN,
    F0_IOU_MIN,
    F0_SCORE_TOL,
    FRACTIONS,
    SEED,
    associate_to_original,
    bool_text,
    box_from_row,
    build_eval_data_dict,
    confirm_locked_artifacts,
    confirm_week3_counts,
    file_stamp,
    index_infos,
    load_model,
    num_to_remove,
    prepare_removal,
    score_text,
    stamp_tree,
    velodyne_bin_paths,
)

# Locked week-6 sample. Do not lower the floor to fill a bin.
SCORE_FLOOR = 0.7
MIN_POINTS = 1
SURVIVAL_SCORE = 0.5
QUOTAS = (67, 67, 66)
EXPECTED_ELIGIBLE = 11125
EXPECTED_RANGE_COUNTS = (4864, 4868, 1393)
EXPECTED_ELIGIBLE_MATCHED = 10478
EXPECTED_NEAR_AT_LEAST_100 = 4825
EXPECTED_NEAR_UNDER_20 = 0
EXPECTED_SMOKE_FORWARDS = 25
EXPECTED_SMOKE_PASSED = 152
PILOT_N = 200
# result.pkl was written by test.py --batch_size 4. A batch of 1 does not
# reproduce every stored box, so each pilot forward uses that same batch.
EVAL_BATCH_SIZE = 4

RANGE_LABELS = tuple(display_bin_label("range_m", index) for index in range(3))
FRACTION_LABELS = tuple("%.2f" % fraction for fraction in FRACTIONS)

PILOT_COLUMNS = [
    "frame_id",
    "pred_idx",
    "range_bin",
    "num_points_bin",
    "size_bin",
    "fraction",
    "score_before",
    "num_points_before",
    "num_removed",
    "best_iou",
    "best_score",
    "survived_iou05",
    "score_after",
    "retained",
    "survived_score05",
]

IDENTITY_COLUMNS = [
    "frame_id",
    "pred_idx",
    "rank_token",
    "rank",
    "score",
    "range_m",
    "num_points_in_box",
    "size_m3",
    "range_bin",
    "num_points_bin",
    "size_bin",
    "gt_matched",
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
DEFAULT_OUTPUT = OPENPCDET_ROOT / "output/week6"
DEFAULT_COPY = (
    OPENPCDET_ROOT.parent
    / "DIS Notes and Outputs"
    / "outputs by week"
    / "week6-outputs"
)

COPY_NAMES = (
    "week6_pilot.csv",
    "week6_sidecar.txt",
    "week6_summary.txt",
)


class PilotStop(RuntimeError):
    """A locked check failed. The partial CSV is already on disk."""


def parse_args():
    """Parse CLI paths. Defaults are inside this repository."""
    parser = argparse.ArgumentParser(description="Week 6 retention pilot")
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
    return parser.parse_args()


def require_equal(name, actual, expected):
    """Stop when a locked count does not match. Do not relax the filter."""
    if actual != expected:
        raise ValueError("%s is %s, locked value is %s" % (name, actual, expected))


def rank_token(range_label, frame_id, pred_idx):
    """Hash input for one eligible row. The label is the week-4 range bin label."""
    return "%d|%s|%s|%d" % (SEED, range_label, frame_id, int(pred_idx))


def rank_digest(token):
    """Lowercase sha256 hex of a rank token. Sort order is this string."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def annotate_row(row):
    """Add week-4 bin labels and box volume. Does not change the raw fields."""
    volume = size_m3(row["box_l"], row["box_w"], row["box_h"])
    range_index, point_index, size_index = assign_bins(
        row["range_m"], row["num_points_in_box"], volume
    )
    annotated = dict(row)
    annotated["size_m3"] = float(volume)
    annotated["range_bin_index"] = int(range_index)
    annotated["range_bin"] = display_bin_label("range_m", range_index)
    annotated["num_points_bin"] = display_bin_label("num_points", point_index)
    annotated["size_bin"] = display_bin_label("size_m3", size_index)
    return annotated


def assert_unique_identities(rows, label):
    """Raise if the same week-3 (frame_id, pred_idx) appears twice."""
    seen = set()
    for row in rows:
        key = (row["frame_id"], int(row["pred_idx"]))
        if key in seen:
            raise ValueError("duplicate %s identity %s pred %s" % (label, key[0], key[1]))
        seen.add(key)


def select_pilot(rows):
    """Draw the quota in each range bin. Ranking is sha256, then frame, then pred.

    Returns (chosen, eligible_stats). chosen is bin 0, then 1, then 2, and
    inside a bin it is increasing rank digest. A short bin stops the draw.
    """
    assert_unique_identities(rows, "week-3")
    eligible = [
        annotate_row(row)
        for row in rows
        if float(row["score"]) >= SCORE_FLOOR and int(row["num_points_in_box"]) >= MIN_POINTS
    ]
    grouped = [[] for _ in RANGE_LABELS]
    for row in eligible:
        grouped[int(row["range_bin_index"])].append(row)

    require_equal("eligible rows", len(eligible), EXPECTED_ELIGIBLE)
    for index, label in enumerate(RANGE_LABELS):
        require_equal("eligible %s" % label, len(grouped[index]), EXPECTED_RANGE_COUNTS[index])
    matched = sum(1 for row in eligible if row["gt_matched"])
    require_equal("eligible gt_matched", matched, EXPECTED_ELIGIBLE_MATCHED)
    near = grouped[0]
    near_dense = sum(1 for row in near if int(row["num_points_in_box"]) >= 100)
    near_sparse = sum(1 for row in near if int(row["num_points_in_box"]) < 20)
    require_equal("near eligible with >=100 points", near_dense, EXPECTED_NEAR_AT_LEAST_100)
    require_equal("near eligible with <20 points", near_sparse, EXPECTED_NEAR_UNDER_20)

    chosen = []
    for index, label in enumerate(RANGE_LABELS):
        pool = grouped[index]
        quota = QUOTAS[index]
        if len(pool) < quota:
            raise ValueError(
                "range bin %s has %d eligible rows, quota is %d"
                % (label, len(pool), quota)
            )
        decorated = []
        for row in pool:
            token = rank_token(label, row["frame_id"], row["pred_idx"])
            decorated.append(
                (rank_digest(token), row["frame_id"], int(row["pred_idx"]), token, row)
            )
        decorated.sort(key=lambda item: (item[0], item[1], item[2]))
        for digest, _frame_id, _pred_idx, token, row in decorated[:quota]:
            picked = dict(row)
            picked["rank_token"] = token
            picked["rank"] = digest
            chosen.append(picked)

    if len(chosen) != PILOT_N:
        raise RuntimeError("pilot draw returned %d rows, expected %d" % (len(chosen), PILOT_N))
    assert_unique_identities(chosen, "pilot")
    for index, label in enumerate(RANGE_LABELS):
        got = sum(1 for row in chosen if row["range_bin"] == label)
        require_equal("pilot %s" % label, got, QUOTAS[index])

    stats = {
        "eligible": len(eligible),
        "eligible_matched": matched,
        "range_counts": tuple(len(group) for group in grouped),
        "pilot_matched": sum(1 for row in chosen if row["gt_matched"]),
    }
    return chosen, stats


def sidecar_text(chosen, stats):
    """Definitions and the 200 identities. Written before any forward."""
    lines = [
        "Week 6 pilot sidecar",
        "seed: %d" % SEED,
        "fractions: %s" % ", ".join(FRACTION_LABELS),
        "num_removed: Python round(fraction * num_points_before), half even",
        "operation: one random interior delete on an in-memory copy",
        "eval batch: %d (week-1 test.py batch). Only the target frame is edited."
        % EVAL_BATCH_SIZE,
        "association: highest 3D IoU of a new Car box to the original predicted box",
        "iou helper: d3_box_overlap on KITTI camera boxes",
        "association bar: %.2f" % ASSOC_IOU,
        "best_score: score of that box, or empty when no new Car box exists",
        "score_after: best_score only when best_iou >= %.2f, otherwise empty" % ASSOC_IOU,
        "retained: score_after / score_before when the IoU bar is met, otherwise 0",
        "survived_score05: IoU bar met and score_after >= %.2f" % SURVIVAL_SCORE,
        "survival score threshold: %.2f" % SURVIVAL_SCORE,
        "rank token: sha256(\"%d|<range bin label>|<frame_id>|<pred_idx>\")" % SEED,
        "range bin labels: %s" % " | ".join(RANGE_LABELS),
        "quotas: %s" % ", ".join(
            "%s %d" % (label, QUOTAS[index]) for index, label in enumerate(RANGE_LABELS)
        ),
        "eligible filter: score >= %.2f and num_points_in_box >= %d" % (SCORE_FLOOR, MIN_POINTS),
        "eligible rows: %d" % stats["eligible"],
        "eligible by range: %s" % ", ".join(
            "%s %d" % (label, stats["range_counts"][index])
            for index, label in enumerate(RANGE_LABELS)
        ),
        "eligible gt_matched: %d" % stats["eligible_matched"],
        "pilot gt_matched: %d" % stats["pilot_matched"],
        "draw: inside each range bin, sort by rank digest, then frame_id, then pred_idx",
        "identity_table:",
        ",".join(IDENTITY_COLUMNS),
    ]
    for row in chosen:
        fields = [
            row["frame_id"],
            str(int(row["pred_idx"])),
            row["rank_token"],
            row["rank"],
            "%.6f" % float(row["score"]),
            "%.6f" % float(row["range_m"]),
            str(int(row["num_points_in_box"])),
            "%.6f" % float(row["size_m3"]),
            row["range_bin"],
            row["num_points_bin"],
            row["size_bin"],
            bool_text(row["gt_matched"]),
        ]
        if len(fields) != len(IDENTITY_COLUMNS):
            raise RuntimeError("identity row width does not match the header")
        lines.append(",".join(fields))
    lines.append("")
    return "\n".join(lines)


def atomic_write(path, text):
    """Replace path in one step so a crash cannot leave a partial sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def parse_sidecar_identities(path):
    """Return (frame_id, pred_idx) in sidecar order."""
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
        identities.append((raw["frame_id"], int(raw["pred_idx"])))
    return identities


def confirm_week5_smoke(week5_dir):
    """Require the locked smoke outputs. Do not rerun those 25 forwards."""
    week5_dir = Path(week5_dir)
    smoke_csv = week5_dir / "week5_smoke.csv"
    check_log = week5_dir / "week5_check_log.txt"
    missing = [str(path) for path in (smoke_csv, check_log) if not path.exists()]
    if missing:
        raise FileNotFoundError("week-5 smoke artifact missing: %s" % missing)
    log_text = check_log.read_text(encoding="utf-8")
    if "failed: 0" not in log_text or "passed: %d" % EXPECTED_SMOKE_PASSED not in log_text:
        raise ValueError("week-5 check log is not the locked 152 passed / 0 failed result")
    with smoke_csv.open("r", encoding="utf-8", newline="") as handle:
        n_rows = sum(1 for _row in csv.DictReader(handle))
    require_equal("week-5 smoke rows", n_rows, EXPECTED_SMOKE_FORWARDS)
    if EXPECTED_AP_TOKEN != "77.2898":
        raise RuntimeError("week-4 AP token constant drifted")


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


def read_pilot_rows(path):
    """Load finished pilot rows. An empty or missing file returns no rows."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    repair_trailing_partial(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PILOT_COLUMNS:
            raise ValueError("pilot CSV header does not match the locked columns")
        rows = []
        for raw in reader:
            if None in raw or any(raw[column] is None for column in PILOT_COLUMNS):
                raise ValueError("pilot CSV has a row with the wrong width")
            rows.append(raw)
    return rows


def finished_keys(rows):
    """Set of (frame_id, pred_idx, fraction label) already stored."""
    keys = set()
    for row in rows:
        key = (row["frame_id"], int(row["pred_idx"]), row["fraction"])
        if key in keys:
            raise ValueError(
                "duplicate pilot row %s pred %s fraction %s" % (key[0], key[1], key[2])
            )
        keys.add(key)
    return keys


def open_pilot_writer(path, already_has_rows):
    """Append rows. Write the header only when the file is still empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=PILOT_COLUMNS, lineterminator="\n")
    if not already_has_rows:
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())
    return handle, writer


def append_pilot_row(handle, writer, row):
    """Write one finished cell and flush it before the next forward."""
    writer.writerow(row)
    handle.flush()
    os.fsync(handle.fileno())


def cached_cloud(cache, velodyne_dir, frame_id):
    """Load each frame's velodyne bin once. The cached array is not edited."""
    if frame_id not in cache:
        cache[frame_id] = load_velodyne(velodyne_dir, frame_id)
    return cache[frame_id]


def removal_or_stop(points, target, fraction):
    """Delete one fraction. Stop if the edit does not match the locked accounting."""
    facts = prepare_removal(
        points,
        box_from_row(target),
        fraction,
        target["frame_id"],
        int(target["pred_idx"]),
    )
    prefix = "%s pred %d f=%.2f" % (target["frame_id"], int(target["pred_idx"]), fraction)
    csv_points = int(target["num_points_in_box"])
    if facts["num_points_before"] != csv_points:
        raise PilotStop(
            "%s num_points_before %d != week-3 %d"
            % (prefix, facts["num_points_before"], csv_points)
        )
    expected_removed = num_to_remove(csv_points, fraction)
    if facts["num_removed"] != expected_removed:
        raise PilotStop(
            "%s num_removed %d != round result %d"
            % (prefix, facts["num_removed"], expected_removed)
        )
    if facts["num_points_after"] != facts["num_points_before"] - facts["num_removed"]:
        raise PilotStop("%s interior count after delete is wrong" % prefix)
    if not facts["deleted_are_interior"]:
        raise PilotStop("%s deleted a point outside the target box" % prefix)
    if not facts["outside_unchanged"]:
        raise PilotStop("%s outside points changed" % prefix)
    if float(fraction) == 0.0:
        identical = facts["num_removed"] == 0 and np.array_equal(facts["modified"], points)
        if not identical:
            raise PilotStop("%s f=0 cloud was modified" % prefix)
    if float(fraction) == 1.0:
        if facts["num_points_before"] <= 0:
            raise PilotStop("%s f=1 target is empty" % prefix)
        if facts["num_points_after"] != 0:
            raise PilotStop("%s f=1 left interior points" % prefix)
    return facts


def build_pilot_row(target, fraction_label, facts, best_score, best_iou, survived):
    """Build one CSV row. An unassociated box does not count as retained confidence.

    best_score keeps the diagnostic score of the highest-IoU Car box.
    score_after is empty, and retained is 0, when survived_iou05 is false.
    """
    if survived and best_score is None:
        raise PilotStop(
            "%s pred %d f=%s survived with no score"
            % (target["frame_id"], int(target["pred_idx"]), fraction_label)
        )
    score_before = float(target["score"])
    if score_before <= 0.0:
        raise PilotStop(
            "%s pred %d score_before is not positive" % (target["frame_id"], int(target["pred_idx"]))
        )
    if survived:
        score_after_text = score_text(best_score)
        retained = float(best_score) / score_before
        survived_score = float(best_score) >= SURVIVAL_SCORE
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
        "fraction": fraction_label,
        "score_before": "%.6f" % score_before,
        "num_points_before": str(int(facts["num_points_before"])),
        "num_removed": str(int(facts["num_removed"])),
        "best_iou": "%.6f" % float(best_iou),
        "best_score": score_text(best_score),
        "survived_iou05": bool_text(survived),
        "score_after": score_after_text,
        "retained": "%.6f" % retained,
        "survived_score05": bool_text(survived_score),
    }


def check_f0_forward(target, best_score, best_iou, survived):
    """f=0 must come back as the same box and nearly the same score."""
    prefix = "%s pred %d f=0.00" % (target["frame_id"], int(target["pred_idx"]))
    if not survived or best_score is None:
        raise PilotStop("%s forward produced no associated Car" % prefix)
    if float(best_iou) < F0_IOU_MIN:
        raise PilotStop("%s iou %.4f is below %.2f" % (prefix, float(best_iou), F0_IOU_MIN))
    delta = abs(float(best_score) - float(target["score"]))
    if delta >= F0_SCORE_TOL:
        raise PilotStop(
            "%s |score_after %.4f - score_before %.4f| = %.4f"
            % (prefix, float(best_score), float(target["score"]), delta)
        )


def load_runtime(args):
    """Load infos and the week-1 checkpoint once. training stays off.

    The dataset index map is the val order used to rebuild each eval batch.
    """
    infos = load_pickle(args.infos_pkl)
    info_by_id = index_infos(infos)
    dataset, model = load_model(args.cfg_file, args.data_root, args.ckpt)
    index_by_frame = {}
    for index, info in enumerate(dataset.kitti_infos):
        frame_id = str(info["point_cloud"]["lidar_idx"])
        if frame_id in index_by_frame:
            raise RuntimeError("duplicate dataset frame id %s" % frame_id)
        index_by_frame[frame_id] = index
    if set(index_by_frame) != set(info_by_id):
        raise RuntimeError("dataset frame ids do not match kitti_infos_val.pkl")
    return dataset, model, info_by_id, index_by_frame


def forward_in_eval_batch(dataset, model, info_by_id, index_by_frame, frame_id, points):
    """Forward one edited cloud inside its original week-1 batch.

    Other frames in the batch are loaded from disk and are not edited.
    The returned prediction dict is only the target frame.
    """
    index = index_by_frame[frame_id]
    start = index - (index % EVAL_BATCH_SIZE)
    end = min(start + EVAL_BATCH_SIZE, len(dataset))
    items = []
    for cursor in range(start, end):
        if cursor == index:
            items.append(build_eval_data_dict(dataset, info_by_id[frame_id], points))
        else:
            items.append(dataset[cursor])
    batch = dataset.collate_batch(items)
    load_data_to_gpu(batch)
    with torch.no_grad():
        pred_dicts, _recall = model(batch)
    annos = dataset.generate_prediction_dicts(
        batch, pred_dicts, dataset.class_names, output_path=None
    )
    if len(annos) != len(items):
        raise RuntimeError("expected %d prediction dicts, got %d" % (len(items), len(annos)))
    return annos[index - start]


def run_pending_forwards(args, pending, writer_handle, writer):
    """Forward each unfinished cell from the original cloud, then append the row.

    Two selected cars in one frame are two forwards. Each starts from the
    unmodified bin. The cached cloud is never written back to disk.
    """
    dataset, model, info_by_id, index_by_frame = load_runtime(args)
    clouds = {}
    calib_by_frame = {}
    started = time.perf_counter()
    for step, (target, fraction, fraction_label) in enumerate(pending, start=1):
        frame_id = target["frame_id"]
        points = cached_cloud(clouds, args.velodyne_dir, frame_id)
        facts = removal_or_stop(points, target, fraction)
        info = info_by_id[frame_id]
        if frame_id not in calib_by_frame:
            calib_by_frame[frame_id] = dataset.get_calib(info["point_cloud"]["lidar_idx"])
        pred_dict = forward_in_eval_batch(
            dataset,
            model,
            info_by_id,
            index_by_frame,
            frame_id,
            facts["modified"],
        )
        best_score, best_iou, survived = associate_to_original(
            pred_dict, box_from_row(target), calib_by_frame[frame_id]
        )
        if float(fraction) == 0.0:
            check_f0_forward(target, best_score, best_iou, survived)
        row = build_pilot_row(
            target, fraction_label, facts, best_score, best_iou, survived
        )
        append_pilot_row(writer_handle, writer, row)
        elapsed = time.perf_counter() - started
        print(
            "%d/%d %s pred %s f=%s iou=%s survived=%s elapsed=%.1fs"
            % (
                step,
                len(pending),
                frame_id,
                row["pred_idx"],
                fraction_label,
                row["best_iou"],
                row["survived_iou05"],
                elapsed,
            ),
            flush=True,
        )


def audit_csv_rules(rows, chosen):
    """Check the stored table against the locked metric rules. Return failure lines."""
    failures = []
    if len(rows) != PILOT_N * len(FRACTIONS):
        failures.append("row count %d, expected %d" % (len(rows), PILOT_N * len(FRACTIONS)))
    expected_keys = []
    by_identity = {}
    for target in chosen:
        by_identity[(target["frame_id"], int(target["pred_idx"]))] = target
        for fraction_label in FRACTION_LABELS:
            expected_keys.append((target["frame_id"], int(target["pred_idx"]), fraction_label))
    try:
        got = finished_keys(rows)
    except ValueError as exc:
        failures.append(str(exc))
        got = set()
    for key in expected_keys:
        if key not in got:
            failures.append("missing cell %s pred %s f=%s" % key)
    if len(failures) > 30:
        return failures[:30]

    for row in rows:
        identity = (row["frame_id"], int(row["pred_idx"]))
        target = by_identity.get(identity)
        prefix = "%s pred %s f=%s" % (row["frame_id"], row["pred_idx"], row["fraction"])
        if target is None:
            failures.append("%s is not in the sidecar draw" % prefix)
            continue
        if row["range_bin"] != target["range_bin"]:
            failures.append("%s range_bin does not match the sidecar" % prefix)
        if int(row["num_points_before"]) != int(target["num_points_in_box"]):
            failures.append("%s num_points_before does not match week-3" % prefix)
        if abs(float(row["score_before"]) - float(target["score"])) > 5e-7:
            failures.append("%s score_before does not match week-3" % prefix)
        expected_removed = num_to_remove(int(target["num_points_in_box"]), float(row["fraction"]))
        if int(row["num_removed"]) != expected_removed:
            failures.append("%s num_removed does not match round(fraction * n)" % prefix)
        survived = row["survived_iou05"] == "true"
        iou = float(row["best_iou"])
        if survived and iou < ASSOC_IOU - 1e-6:
            failures.append("%s survived_iou05 is true below the IoU bar" % prefix)
        if (not survived) and iou >= ASSOC_IOU:
            failures.append("%s survived_iou05 is false at or above the IoU bar" % prefix)
        score_after = row["score_after"].strip()
        if not survived:
            if score_after != "":
                failures.append("%s missed the IoU bar but score_after is set" % prefix)
            if float(row["retained"]) != 0.0:
                failures.append("%s missed the IoU bar but retained is not 0" % prefix)
            if row["survived_score05"] != "false":
                failures.append("%s missed the IoU bar but survived_score05 is true" % prefix)
        else:
            if score_after == "":
                failures.append("%s met the IoU bar but score_after is empty" % prefix)
            elif row["best_score"].strip() != score_after:
                failures.append("%s score_after does not equal best_score" % prefix)
            else:
                expected_retained = float(score_after) / float(row["score_before"])
                # Both scores are already rounded to 6 decimals, so the ratio
                # can differ from the stored ratio by about one digit.
                if abs(float(row["retained"]) - expected_retained) > 2e-6:
                    failures.append("%s retained does not equal score_after / score_before" % prefix)
                expect_flag = float(score_after) >= SURVIVAL_SCORE
                if (row["survived_score05"] == "true") != expect_flag:
                    failures.append("%s survived_score05 does not match the 0.50 rule" % prefix)
        if row["fraction"] == "0.00":
            if iou + 1e-6 < F0_IOU_MIN:
                failures.append("%s f=0 best_iou %.6f is below %.2f" % (prefix, iou, F0_IOU_MIN))
            if score_after == "":
                failures.append("%s f=0 score_after is empty" % prefix)
            elif abs(float(score_after) - float(row["score_before"])) >= F0_SCORE_TOL + 1e-5:
                failures.append("%s f=0 score moved by %.6f or more" % (prefix, F0_SCORE_TOL))
        if len(failures) > 30:
            return failures[:30]
    return failures


def audit_f1_removal(chosen, velodyne_dir, cache):
    """Recompute the f=1 delete. Every pilot target is nonempty by the filter."""
    failures = []
    for target in chosen:
        points = cached_cloud(cache, velodyne_dir, target["frame_id"])
        try:
            facts = removal_or_stop(points, target, 1.0)
        except PilotStop as exc:
            failures.append(str(exc))
            if len(failures) > 20:
                break
            continue
        if facts["num_points_after"] != 0 or not facts["outside_unchanged"]:
            failures.append(
                "%s pred %d f=1 did not empty the box or changed outside points"
                % (target["frame_id"], int(target["pred_idx"]))
            )
    return failures


def slice_rows(rows, range_label, fraction_label):
    """Rows for one range bin and one fraction label."""
    return [
        row
        for row in rows
        if row["range_bin"] == range_label and row["fraction"] == fraction_label
    ]


def mean_retained(rows):
    """Mean of stored retained. Misses are already 0 in that column."""
    if not rows:
        raise ValueError("cannot average an empty slice")
    return sum(float(row["retained"]) for row in rows) / float(len(rows))


def flag_rate(rows, field):
    """Fraction of rows whose flag column is the string true."""
    if not rows:
        raise ValueError("cannot rate an empty slice")
    hits = sum(1 for row in rows if row[field] == "true")
    return hits, hits / float(len(rows))


def metric_lines(rows):
    """One block per range bin, with retained and both survival rates."""
    lines = []
    for label in RANGE_LABELS:
        block = slice_rows(rows, label, FRACTION_LABELS[0])
        lines.append("range bin %s  n=%d" % (label, len(block)))
        for fraction_label in FRACTION_LABELS:
            group = slice_rows(rows, label, fraction_label)
            if len(group) != len(block):
                raise ValueError(
                    "range bin %s fraction %s has %d rows, expected %d"
                    % (label, fraction_label, len(group), len(block))
                )
            retained = mean_retained(group)
            score_hits, score_rate = flag_rate(group, "survived_score05")
            iou_hits, iou_rate = flag_rate(group, "survived_iou05")
            lines.append(
                "  f=%s  mean_retained=%.6f  score_survival=%.6f (%d/%d)  iou_survival=%.6f (%d/%d)"
                % (
                    fraction_label,
                    retained,
                    score_rate,
                    score_hits,
                    len(group),
                    iou_rate,
                    iou_hits,
                    len(group),
                )
            )
    return lines


def decision_lines(rows):
    """GO only when every range bin drops both metrics from f=0 to f=1.

    Returns (lines, decision). decision is GO or NO-GO. The caller must
    skip this when a correctness check has already failed.
    """
    lines = ["comparisons:"]
    failed = []
    for label in RANGE_LABELS:
        base = slice_rows(rows, label, "0.00")
        emptied = slice_rows(rows, label, "1.00")
        retained_0 = mean_retained(base)
        retained_1 = mean_retained(emptied)
        _score_hits_0, score_0 = flag_rate(base, "survived_score05")
        _score_hits_1, score_1 = flag_rate(emptied, "survived_score05")
        retained_lower = retained_1 < retained_0
        score_lower = score_1 < score_0
        lines.append(
            "  %s mean_retained f1=%.6f f0=%.6f lower=%s"
            % (label, retained_1, retained_0, bool_text(retained_lower))
        )
        lines.append(
            "  %s score_survival f1=%.6f f0=%.6f lower=%s"
            % (label, score_1, score_0, bool_text(score_lower))
        )
        if not retained_lower:
            failed.append(
                "%s mean_retained (f1=%.6f, f0=%.6f)" % (label, retained_1, retained_0)
            )
        if not score_lower:
            failed.append(
                "%s score_survival (f1=%.6f, f0=%.6f)" % (label, score_1, score_0)
            )
    if failed:
        lines.append("decision: NO-GO")
        for item in failed:
            lines.append("failed: %s" % item)
        return lines, "NO-GO"
    lines.append("decision: GO")
    lines.append("failed: none")
    return lines, "GO"


def summary_text(stats, correctness, rows, file_note):
    """Counts, correctness, then the metric table and GO or NO-GO.

    When correctness fails, the metric comparison is omitted.
    """
    lines = [
        "Week 6 pilot summary",
        "eligible: %d (score >= %.2f, num_points_in_box >= %d)"
        % (stats["eligible"], SCORE_FLOOR, MIN_POINTS),
        "eligible by range: %s"
        % ", ".join(
            "%s %d" % (label, stats["range_counts"][index])
            for index, label in enumerate(RANGE_LABELS)
        ),
        "eligible gt_matched: %d" % stats["eligible_matched"],
        "pilot: %d" % PILOT_N,
        "pilot by range: %s"
        % ", ".join(
            "%s %d" % (label, QUOTAS[index]) for index, label in enumerate(RANGE_LABELS)
        ),
        "pilot gt_matched: %d" % stats["pilot_matched"],
        "forwards: %d" % (PILOT_N * len(FRACTIONS)),
        "seed: %d" % SEED,
        "fractions: %s" % " ".join(FRACTION_LABELS),
        "association bar: %.2f" % ASSOC_IOU,
        "score survival threshold: %.2f" % SURVIVAL_SCORE,
        "eval batch size: %d" % EVAL_BATCH_SIZE,
        "correctness:",
    ]
    if correctness:
        lines.append("  failures: %d" % len(correctness))
        for item in correctness:
            lines.append("  FAIL %s" % item)
        lines.append("decision: INVALID")
        lines.append(file_note)
        lines.append("")
        return "\n".join(lines)
    lines.append("  failures: 0")
    lines.append("  %s" % file_note)
    lines.extend(metric_lines(rows))
    comparison, _decision = decision_lines(rows)
    lines.extend(comparison)
    lines.append("")
    return "\n".join(lines)


def copy_outputs(output_dir, copy_dir):
    """Copy the pilot table, sidecar, and summary into the notes folder."""
    output_dir = Path(output_dir)
    copy_dir = Path(copy_dir)
    if not copy_dir.parent.exists():
        print("skip copy; parent missing: %s" % copy_dir.parent, flush=True)
        return
    copy_dir.mkdir(parents=True, exist_ok=True)
    for name in COPY_NAMES:
        src = output_dir / name
        if src.exists():
            shutil.copy2(str(src), str(copy_dir / name))
    print("copied outputs to %s" % copy_dir, flush=True)


def file_change_note(before_bins, after_bins, before_result, after_result):
    """One summary line for the unchanged-input check. A mismatch is a failure."""
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


def pending_cells(chosen, done):
    """Cells in sidecar order that are not already in the CSV."""
    pending = []
    for target in chosen:
        for fraction, fraction_label in zip(FRACTIONS, FRACTION_LABELS):
            key = (target["frame_id"], int(target["pred_idx"]), fraction_label)
            if key not in done:
                pending.append((target, fraction, fraction_label))
    return pending


def write_stop_summary(output_dir, message):
    """Record an early stop without a GO or NO-GO line."""
    text = "\n".join(
        [
            "Week 6 pilot summary",
            "decision: INVALID",
            "stopped: %s" % message,
            "",
        ]
    )
    atomic_write(Path(output_dir) / "week6_summary.txt", text)


def main():
    """Draw 200 targets, forward unfinished cells, then write the decision."""
    args = parse_args()
    if sum(QUOTAS) != PILOT_N:
        raise RuntimeError("quotas sum to %d, expected %d" % (sum(QUOTAS), PILOT_N))
    confirm_locked_artifacts(args)
    confirm_week5_smoke(args.week5_dir)
    rows = read_csv_rows(args.week3_csv)
    confirm_week3_counts(rows, args.walked_txt)
    chosen, stats = select_pilot(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = output_dir / "week6_sidecar.txt"
    pilot_path = output_dir / "week6_pilot.csv"
    fresh = [(row["frame_id"], int(row["pred_idx"])) for row in chosen]
    if sidecar_path.exists():
        saved = parse_sidecar_identities(sidecar_path)
        if saved != fresh:
            raise ValueError("existing sidecar does not match the deterministic draw")
    atomic_write(sidecar_path, sidecar_text(chosen, stats))
    print("sidecar %s identities %d" % (sidecar_path, len(chosen)), flush=True)
    copy_outputs(output_dir, args.copy_dir)
    if args.draw_only:
        print("draw-only stop", flush=True)
        return

    existing = read_pilot_rows(pilot_path)
    done = finished_keys(existing)
    pending = pending_cells(chosen, done)
    print("finished cells %d  pending %d" % (len(done), len(pending)), flush=True)

    print("stamping velodyne and result.pkl", flush=True)
    before_bins = stamp_tree(velodyne_bin_paths(args.velodyne_dir))
    before_result = file_stamp(args.result_pkl)
    print("stamp done, %d velodyne files" % len(before_bins), flush=True)
    writer_handle = None
    try:
        if pending:
            header_present = pilot_path.exists() and pilot_path.stat().st_size > 0
            writer_handle, writer = open_pilot_writer(pilot_path, already_has_rows=header_present)
            run_pending_forwards(args, pending, writer_handle, writer)
    except PilotStop as exc:
        write_stop_summary(output_dir, str(exc))
        copy_outputs(output_dir, args.copy_dir)
        print("stopped: %s" % exc, flush=True)
        raise SystemExit(1)
    finally:
        if writer_handle is not None:
            writer_handle.flush()
            writer_handle.close()

    finished = read_pilot_rows(pilot_path)
    clouds = {}
    correctness = []
    correctness.extend(audit_csv_rules(finished, chosen))
    correctness.extend(audit_f1_removal(chosen, args.velodyne_dir, clouds))
    after_bins = stamp_tree(velodyne_bin_paths(args.velodyne_dir))
    after_result = file_stamp(args.result_pkl)
    file_note, file_failures = file_change_note(
        before_bins, after_bins, before_result, after_result
    )
    correctness.extend(file_failures)
    text = summary_text(stats, correctness, finished, file_note)
    atomic_write(output_dir / "week6_summary.txt", text)
    copy_outputs(output_dir, args.copy_dir)
    print(text, flush=True)
    if correctness:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
