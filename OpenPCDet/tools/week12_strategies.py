"""Week-12 removal strategies, association rules, and table math.

numpy only. Nothing here loads a model, a cloud, or a pickle, so the
self-test runs on any machine:

    python week12_strategies.py

week12_ablations.py imports these helpers and checks that the random
seed and the half-even count still match the week-5 and week-7 code.
"""

import hashlib

import numpy as np

SEED = 6908
STRATEGIES = ("random", "fps", "voxel")
NEW_STRATEGIES = ("fps", "voxel")
RESULT_FRACTIONS = (0.25, 0.5, 0.75)


# ---------------------------------------------------------------------------
# Seeds and counts
# ---------------------------------------------------------------------------


def token_seed(token):
    """32-bit seed from the first four sha256 bytes, little endian."""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def random_token(frame_id, pred_idx, fraction):
    """The week-7 interior token. Random keeps it unchanged."""
    return "%d|%s|%s|%d|%.2f" % (SEED, "interior", frame_id, int(pred_idx), float(fraction))


def strategy_token(strategy, frame_id, pred_idx, fraction):
    """Week-12 token for a new strategy. The strategy name is part of it."""
    if strategy not in NEW_STRATEGIES:
        raise ValueError("strategy token is only for %s, got %s" % (NEW_STRATEGIES, strategy))
    return "%d|%s|interior|%s|%d|%.2f" % (SEED, strategy, frame_id, int(pred_idx), float(fraction))


def random_seed(frame_id, pred_idx, fraction):
    return token_seed(random_token(frame_id, pred_idx, fraction))


def strategy_seed(strategy, frame_id, pred_idx, fraction):
    return token_seed(strategy_token(strategy, frame_id, pred_idx, fraction))


def num_to_remove(num_points, fraction):
    """Python round, half even. Same rule as week 5."""
    return int(round(float(fraction) * int(num_points)))


# ---------------------------------------------------------------------------
# Farthest-point sampling
# ---------------------------------------------------------------------------


def _as_xyz(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError("expected an (n, 3) array, got %s" % (xyz.shape,))
    return xyz[:, 0:3]


def fps_order(xyz, k, start):
    """First k indices chosen by farthest-point sampling from start.

    Distance is squared Euclidean in x, y, z. A chosen index is never chosen
    again, even when the remaining points are duplicates of chosen ones.
    Ties break by the lower index (np.argmax returns the first maximum).
    """
    xyz = _as_xyz(xyz)
    n = int(xyz.shape[0])
    k = int(k)
    if k < 0 or k > n:
        raise ValueError("fps k=%d outside 0..%d" % (k, n))
    if k == 0:
        return np.zeros((0,), dtype=np.int64)
    start = int(start)
    if start < 0 or start >= n:
        raise ValueError("fps start %d outside 0..%d" % (start, n - 1))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    order = np.empty((k,), dtype=np.int64)
    order[0] = start
    dist = (x - x[start]) ** 2 + (y - y[start]) ** 2 + (z - z[start]) ** 2
    dist[start] = -1.0
    for i in range(1, k):
        nxt = int(np.argmax(dist))
        if dist[nxt] < 0.0:
            raise RuntimeError("fps ran out of unchosen points")
        order[i] = nxt
        step = (x - x[nxt]) ** 2 + (y - y[nxt]) ** 2 + (z - z[nxt]) ** 2
        np.minimum(dist, step, out=dist)
        dist[nxt] = -1.0
    return order


def fps_reference(xyz, k, start):
    """Plain-Python farthest-point sampling. Used only by the audit on small boxes."""
    xyz = _as_xyz(xyz)
    pts = [(float(row[0]), float(row[1]), float(row[2])) for row in xyz]
    n = len(pts)
    k = int(k)
    if k == 0:
        return np.zeros((0,), dtype=np.int64)
    chosen = [int(start)]
    used = [False] * n
    used[int(start)] = True
    sx, sy, sz = pts[int(start)]
    best = [(p[0] - sx) ** 2 + (p[1] - sy) ** 2 + (p[2] - sz) ** 2 for p in pts]
    while len(chosen) < k:
        pick = -1
        pick_d = -1.0
        for index in range(n):
            if used[index]:
                continue
            if best[index] > pick_d:
                pick = index
                pick_d = best[index]
        chosen.append(pick)
        used[pick] = True
        px, py, pz = pts[pick]
        for index in range(n):
            d = (pts[index][0] - px) ** 2 + (pts[index][1] - py) ** 2 + (pts[index][2] - pz) ** 2
            if d < best[index]:
                best[index] = d
    return np.asarray(chosen, dtype=np.int64)


def fps_start(n, seed):
    """Seeded start position in the sorted interior."""
    rng = np.random.default_rng(int(seed))
    return int(rng.integers(int(n)))


def fps_delete(interior_idx, xyz_interior, num_removed, seed):
    """Keep n - num_removed interior points chosen by FPS. Delete the rest.

    interior_idx must be sorted, and xyz_interior must be the same rows in the
    same order. Returns (sorted global delete indices, start position).
    """
    interior_idx = np.asarray(interior_idx, dtype=np.int64)
    n = int(interior_idx.size)
    if xyz_interior.shape[0] != n:
        raise ValueError("xyz rows %d != interior %d" % (xyz_interior.shape[0], n))
    if n and not bool(np.all(np.diff(interior_idx) > 0)):
        raise ValueError("interior indices are not sorted and unique")
    num_removed = int(num_removed)
    if num_removed < 0 or num_removed > n:
        raise ValueError("num_removed %d outside 0..%d" % (num_removed, n))
    if num_removed == 0:
        return np.zeros((0,), dtype=np.int64), None
    if num_removed == n:
        return interior_idx.copy(), None
    start = fps_start(n, seed)
    keep_local = fps_order(xyz_interior, n - num_removed, start)
    mask = np.ones((n,), dtype=bool)
    mask[keep_local] = False
    return interior_idx[mask], start


# ---------------------------------------------------------------------------
# Voxel-wise (whole pillars on the detector grid)
# ---------------------------------------------------------------------------


def pillar_keys(xyz, pc_range, voxel_size):
    """Integer grid cell of each point. Same floor rule as the voxelizer."""
    xyz = _as_xyz(xyz)
    origin = np.asarray(pc_range[0:3], dtype=np.float64)
    size = np.asarray(voxel_size, dtype=np.float64)
    if size.shape != (3,) or bool(np.any(size <= 0.0)):
        raise ValueError("voxel size must be three positive numbers, got %s" % (voxel_size,))
    return np.floor((xyz - origin) / size).astype(np.int64)


def group_pillars(xyz, pc_range, voxel_size):
    """Return (unique keys sorted, member lists in local index order)."""
    keys = pillar_keys(xyz, pc_range, voxel_size)
    if keys.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int64), []
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(uniq.shape[0] + 1))
    members = [order[bounds[i]:bounds[i + 1]] for i in range(uniq.shape[0])]
    return uniq, members


def voxel_delete(interior_idx, xyz_interior, num_removed, seed, pc_range, voxel_size):
    """Delete whole occupied pillars in seeded order, then part of one more.

    Pillars are visited in a seeded permutation of their sorted keys. A
    pillar is deleted whole while it fits in the remaining count. The first
    pillar that would overshoot gives up a seeded random subset of exactly
    the remainder, and the walk stops. Returns
    (sorted global delete indices, pillars emptied, points from the partial pillar).
    """
    interior_idx = np.asarray(interior_idx, dtype=np.int64)
    n = int(interior_idx.size)
    if xyz_interior.shape[0] != n:
        raise ValueError("xyz rows %d != interior %d" % (xyz_interior.shape[0], n))
    num_removed = int(num_removed)
    if num_removed < 0 or num_removed > n:
        raise ValueError("num_removed %d outside 0..%d" % (num_removed, n))
    if num_removed == 0:
        return np.zeros((0,), dtype=np.int64), 0, 0
    uniq, members = group_pillars(xyz_interior, pc_range, voxel_size)
    rng = np.random.default_rng(int(seed))
    visit = rng.permutation(len(members))
    remaining = num_removed
    chosen = []
    emptied = 0
    partial = 0
    for pillar in visit:
        group = members[int(pillar)]
        if group.size <= remaining:
            chosen.append(group)
            remaining -= int(group.size)
            emptied += 1
        else:
            picked = rng.choice(np.sort(group), size=remaining, replace=False)
            chosen.append(np.asarray(picked, dtype=np.int64))
            partial = remaining
            remaining = 0
        if remaining == 0:
            break
    if remaining != 0:
        raise RuntimeError("voxel walk ended with %d points left" % remaining)
    local = np.sort(np.concatenate(chosen)) if chosen else np.zeros((0,), dtype=np.int64)
    return interior_idx[local], emptied, partial


def check_voxel_delete(interior_idx, xyz_interior, delete_idx, num_removed, pc_range, voxel_size):
    """Return a list of problems. Empty means the delete has the pillar shape."""
    problems = []
    interior_idx = np.asarray(interior_idx, dtype=np.int64)
    delete_idx = np.asarray(delete_idx, dtype=np.int64)
    if int(delete_idx.size) != int(num_removed):
        problems.append("deleted %d, expected %d" % (delete_idx.size, num_removed))
    if delete_idx.size and not bool(np.all(np.isin(delete_idx, interior_idx))):
        problems.append("deleted a point outside the interior")
    if delete_idx.size != np.unique(delete_idx).size:
        problems.append("deleted a point twice")
    _uniq, members = group_pillars(xyz_interior, pc_range, voxel_size)
    deleted = np.isin(interior_idx, delete_idx)
    partial = 0
    for group in members:
        hit = int(np.sum(deleted[group]))
        if 0 < hit < int(group.size):
            partial += 1
    if partial > 1:
        problems.append("%d pillars were partly deleted, at most 1 allowed" % partial)
    return problems


# ---------------------------------------------------------------------------
# Association rules on the saved returned boxes
# ---------------------------------------------------------------------------

# Each returned Car box is stored as [score, iou3d_to_original, bev_center_distance_m],
# in the detector's output order.

ASSOCIATION_RULES = ("highest_iou", "highest_score_above_bar", "nearest_center")


def rule_highest_iou(boxes, bar):
    """Week-7 rule. First maximum IoU in output order. Returns (score, iou, survived)."""
    if not boxes:
        return None, 0.0, False
    ious = [float(box[1]) for box in boxes]
    best = int(np.argmax(np.asarray(ious, dtype=np.float64)))
    return float(boxes[best][0]), ious[best], bool(ious[best] >= float(bar))


def rule_highest_score(boxes, bar):
    """Highest score among boxes with IoU at least bar. Returns (score, iou, survived)."""
    candidates = [box for box in boxes if float(box[1]) >= float(bar)]
    if not candidates:
        return None, 0.0, False
    scores = np.asarray([float(box[0]) for box in candidates], dtype=np.float64)
    best = int(np.argmax(scores))
    return float(candidates[best][0]), float(candidates[best][1]), True


def rule_nearest_center(boxes, radius):
    """Nearest BEV center within radius, no IoU requirement. Returns (score, dist, survived)."""
    if not boxes:
        return None, float("inf"), False
    dists = np.asarray([float(box[2]) for box in boxes], dtype=np.float64)
    best = int(np.argmin(dists))
    return float(boxes[best][0]), float(dists[best]), bool(dists[best] <= float(radius))


def retained_from(score_before, score_after, survived):
    """Week-7 retention. A miss is 0 and stays in the mean."""
    if not survived:
        return 0.0
    if score_after is None:
        raise ValueError("survived with no score")
    return float(score_after) / float(score_before)


# ---------------------------------------------------------------------------
# Table math
# ---------------------------------------------------------------------------

TABLE_COLUMNS = [
    "ablation",
    "setting",
    "condition",
    "stratum_type",
    "stratum",
    "fraction",
    "n",
    "mean_num_removed",
    "mean_retained",
    "score_survival_hits",
    "score_survival",
    "iou_survival_hits",
    "iou_survival",
]


def record(row_bins, fraction_label, num_removed, retained, score_ok, iou_ok, condition="interior"):
    """One scored cell. row_bins holds range_bin, num_points_bin, size_bin."""
    return {
        "range_bin": row_bins["range_bin"],
        "num_points_bin": row_bins["num_points_bin"],
        "size_bin": row_bins["size_bin"],
        "condition": condition,
        "fraction": fraction_label,
        "num_removed": None if num_removed is None else int(num_removed),
        "retained": float(retained),
        "score_ok": bool(score_ok),
        "iou_ok": bool(iou_ok),
    }


def table_rows(ablation, setting, records, labels, fraction_labels, conditions=("interior",)):
    """Marginal rows: every condition, axis bin, and fraction. n = 0 rows stay."""
    axes = (
        ("range", "range_bin", labels["range"]),
        ("points", "num_points_bin", labels["points"]),
        ("size", "size_bin", labels["size"]),
    )
    out = []
    for condition in conditions:
        for stratum_type, field, bins in axes:
            for label in bins:
                for fraction_label in fraction_labels:
                    group = [
                        item
                        for item in records
                        if item["condition"] == condition
                        and item["fraction"] == fraction_label
                        and item[field] == label
                    ]
                    out.append(summarize(ablation, setting, condition, stratum_type, label, fraction_label, group))
    return out


def summarize(ablation, setting, condition, stratum_type, stratum, fraction_label, group):
    row = {
        "ablation": ablation,
        "setting": setting,
        "condition": condition,
        "stratum_type": stratum_type,
        "stratum": stratum,
        "fraction": fraction_label,
        "n": str(len(group)),
        "mean_num_removed": "",
        "mean_retained": "",
        "score_survival_hits": "",
        "score_survival": "",
        "iou_survival_hits": "",
        "iou_survival": "",
    }
    if not group:
        return row
    n = len(group)
    removed = [item["num_removed"] for item in group if item["num_removed"] is not None]
    if len(removed) == n:
        row["mean_num_removed"] = "%.6f" % (sum(removed) / float(n))
    score_hits = sum(1 for item in group if item["score_ok"])
    iou_hits = sum(1 for item in group if item["iou_ok"])
    row["mean_retained"] = "%.6f" % (sum(item["retained"] for item in group) / float(n))
    row["score_survival_hits"] = "%d/%d" % (score_hits, n)
    row["score_survival"] = "%.6f" % (score_hits / float(n))
    row["iou_survival_hits"] = "%d/%d" % (iou_hits, n)
    row["iou_survival"] = "%.6f" % (iou_hits / float(n))
    return row


# ---------------------------------------------------------------------------
# Self-test (no model, no data)
# ---------------------------------------------------------------------------


def self_test():
    """Fail fast if a strategy or rule does not do what the definitions say."""
    if num_to_remove(1, 0.5) != 0 or num_to_remove(3, 0.5) != 2:
        raise AssertionError("half-even rounding changed")
    if random_token("000002", 0, 0.25) != "6908|interior|000002|0|0.25":
        raise AssertionError("random token is not the week-7 interior token")
    if strategy_token("fps", "000002", 0, 0.25) != "6908|fps|interior|000002|0|0.25":
        raise AssertionError("fps token changed")

    rng = np.random.default_rng(1)
    xyz = rng.normal(size=(120, 3))
    xyz[10] = xyz[3]  # a duplicate point must not break FPS
    for k in (1, 2, 37, 119, 120):
        fast = fps_order(xyz, k, 5)
        slow = fps_reference(xyz, k, 5)
        if not np.array_equal(fast, slow):
            raise AssertionError("fps_order disagrees with the reference at k=%d" % k)
        if np.unique(fast).size != k:
            raise AssertionError("fps chose a point twice at k=%d" % k)

    interior = np.arange(1000, 1120, dtype=np.int64)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        n_removed = num_to_remove(interior.size, fraction)
        deleted, _start = fps_delete(interior, xyz, n_removed, 42)
        if deleted.size != n_removed or not np.all(np.isin(deleted, interior)):
            raise AssertionError("fps_delete count or region is wrong at f=%.2f" % fraction)
        again, _start = fps_delete(interior, xyz, n_removed, 42)
        if not np.array_equal(deleted, again):
            raise AssertionError("fps_delete is not deterministic")

    pc_range = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)
    voxel = (0.16, 0.16, 4.0)
    box_xyz = np.column_stack(
        [
            rng.uniform(10.0, 11.0, size=300),
            rng.uniform(2.0, 2.6, size=300),
            rng.uniform(-1.5, 0.0, size=300),
        ]
    )
    interior = np.arange(5000, 5300, dtype=np.int64)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        n_removed = num_to_remove(interior.size, fraction)
        deleted, emptied, partial = voxel_delete(interior, box_xyz, n_removed, 7, pc_range, voxel)
        problems = check_voxel_delete(interior, box_xyz, deleted, n_removed, pc_range, voxel)
        if problems:
            raise AssertionError("voxel f=%.2f: %s" % (fraction, problems))
        if fraction == 1.0 and not np.array_equal(deleted, interior):
            raise AssertionError("voxel f=1 did not delete the whole interior")
        if fraction > 0.0 and emptied == 0 and partial != n_removed:
            raise AssertionError("voxel accounting is inconsistent")

    boxes = [[0.9, 0.2, 0.3], [0.6, 0.8, 1.5], [0.8, 0.55, 2.5], [0.95, 0.0, 9.0]]
    if rule_highest_iou(boxes, 0.5) != (0.6, 0.8, True):
        raise AssertionError("highest-IoU rule changed")
    if rule_highest_score(boxes, 0.5)[0] != 0.8:
        raise AssertionError("highest-score rule changed")
    if rule_nearest_center(boxes, 2.0)[0] != 0.9:
        raise AssertionError("nearest-center rule changed")
    if rule_nearest_center([[0.9, 0.9, 2.5]], 2.0)[2]:
        raise AssertionError("nearest-center radius is not applied")
    if rule_highest_iou([], 0.5) != (None, 0.0, False):
        raise AssertionError("empty box list must be a miss")
    return True


if __name__ == "__main__":
    self_test()
    print("week12_strategies self-test passed")
