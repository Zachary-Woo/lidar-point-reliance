#####################
# Based on https://github.com/hongzhenwang/RRPN-revise
# Licensed under The MIT License
# Author: yanyan, scrin@foxmail.com
#
# Notes:
# The upstream OpenPCDet file runs rotated-box IoU with numba.cuda.
# On WSL that CUDA path can segfault even when torch CUDA works.
# This file keeps the same public API (rotate_iou_gpu_eval) but
# computes IoU with CPU numba, which is fast enough for KITTI eval.
#####################
import math

import numba
import numpy as np


@numba.jit(nopython=True)
def _trangle_area(a_x, a_y, b_x, b_y, c_x, c_y):
    """Return signed triangle area for three 2D points."""
    return ((a_x - c_x) * (b_y - c_y) - (a_y - c_y) * (b_x - c_x)) / 2.0


@numba.jit(nopython=True)
def _area(int_pts, num_of_inter):
    """Return polygon area from an ordered intersection vertex list."""
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            _trangle_area(
                int_pts[0],
                int_pts[1],
                int_pts[2 * i + 2],
                int_pts[2 * i + 3],
                int_pts[2 * i + 4],
                int_pts[2 * i + 5],
            )
        )
    return area_val


@numba.jit(nopython=True)
def _sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    """Sort intersection vertices into convex-polygon order in place."""
    if num_of_inter <= 0:
        return

    center_x = 0.0
    center_y = 0.0
    for i in range(num_of_inter):
        center_x += int_pts[2 * i]
        center_y += int_pts[2 * i + 1]
    center_x /= num_of_inter
    center_y /= num_of_inter

    vs = np.zeros((num_of_inter,), dtype=np.float32)
    for i in range(num_of_inter):
        v0 = int_pts[2 * i] - center_x
        v1 = int_pts[2 * i + 1] - center_y
        d = math.sqrt(v0 * v0 + v1 * v1)
        # Duplicate vertices sit on the centroid. Dividing by that length
        # aborts the kernel, which is the identical-box case.
        if d < 1e-8:
            vs[i] = 0.0
            continue
        v0 = v0 / d
        v1 = v1 / d
        if v1 < 0:
            v0 = -2.0 - v0
        vs[i] = v0

    for i in range(1, num_of_inter):
        if vs[i - 1] > vs[i]:
            temp = vs[i]
            tx = int_pts[2 * i]
            ty = int_pts[2 * i + 1]
            j = i
            while j > 0 and vs[j - 1] > temp:
                vs[j] = vs[j - 1]
                int_pts[j * 2] = int_pts[j * 2 - 2]
                int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                j -= 1
            vs[j] = temp
            int_pts[j * 2] = tx
            int_pts[j * 2 + 1] = ty


@numba.jit(nopython=True)
def _line_segment_intersection(pts1, pts2, i, j, temp_pts):
    """Find intersection of edge i on pts1 with edge j on pts2."""
    a0 = pts1[2 * i]
    a1 = pts1[2 * i + 1]
    b0 = pts1[2 * ((i + 1) % 4)]
    b1 = pts1[2 * ((i + 1) % 4) + 1]
    c0 = pts2[2 * j]
    c1 = pts2[2 * j + 1]
    d0 = pts2[2 * ((j + 1) % 4)]
    d1 = pts2[2 * ((j + 1) % 4) + 1]

    ba0 = b0 - a0
    ba1 = b1 - a1
    da0 = d0 - a0
    ca0 = c0 - a0
    da1 = d1 - a1
    ca1 = c1 - a1
    acd = da1 * ca0 > ca1 * da0
    bcd = (d1 - b1) * (c0 - b0) > (c1 - b1) * (d0 - b0)
    if acd != bcd:
        abc = ca1 * ba0 > ba1 * ca0
        abd = da1 * ba0 > ba1 * da0
        if abc != abd:
            dc0 = d0 - c0
            dc1 = d1 - c1
            abba = a0 * b1 - b0 * a1
            cddc = c0 * d1 - d0 * c1
            dh = ba1 * dc0 - ba0 * dc1
            # Parallel or coincident edges have no single intersection point.
            if abs(dh) < 1e-8:
                return False
            dx = abba * dc0 - ba0 * cddc
            dy = abba * dc1 - ba1 * cddc
            temp_pts[0] = dx / dh
            temp_pts[1] = dy / dh
            return True
    return False


@numba.jit(nopython=True)
def _point_in_quadrilateral(pt_x, pt_y, corners):
    """Return whether a point lies inside an axis-aligned-in-corner quad."""
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]
    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]
    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]
    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1
    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0


@numba.jit(nopython=True)
def _quadrilateral_intersection(pts1, pts2, int_pts):
    """Collect intersection polygon vertices for two quads."""
    num_of_inter = 0
    for i in range(4):
        if _point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if _point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1

    temp_pts = np.zeros((2,), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            has_pts = _line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1
    return num_of_inter


@numba.jit(nopython=True)
def _rbbox_to_corners(corners, rbbox):
    """Convert a rotated box (x, y, dx, dy, angle) into 4 corner xy pairs."""
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = np.array([-x_d / 2.0, -x_d / 2.0, x_d / 2.0, x_d / 2.0], dtype=np.float32)
    corners_y = np.array([-y_d / 2.0, y_d / 2.0, y_d / 2.0, -y_d / 2.0], dtype=np.float32)
    for i in range(4):
        corners[2 * i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i + 1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y


@numba.jit(nopython=True)
def _inter(rbbox1, rbbox2):
    """Return intersection area of two rotated boxes."""
    corners1 = np.zeros((8,), dtype=np.float32)
    corners2 = np.zeros((8,), dtype=np.float32)
    intersection_corners = np.zeros((16,), dtype=np.float32)

    _rbbox_to_corners(corners1, rbbox1)
    _rbbox_to_corners(corners2, rbbox2)
    num_intersection = _quadrilateral_intersection(
        corners1, corners2, intersection_corners
    )
    _sort_vertex_in_convex_polygon(intersection_corners, num_intersection)
    return _area(intersection_corners, num_intersection)


@numba.jit(nopython=True)
def _rbbox_almost_equal(rbox1, rbox2):
    """True when two rotated boxes match within a millimeter and a milliradian.

    The polygon clipper returns a nonsense area when two boxes share edges.
    A reproduced detection is that case. Boxes this close are the same
    footprint, so the intersection area is the box area.
    """
    for i in range(5):
        if abs(rbox1[i] - rbox2[i]) > 1e-3:
            return False
    return True


@numba.jit(nopython=True)
def _dev_rotate_iou_eval(rbox1, rbox2, criterion=-1):
    """Return IoU / IoF for one rotated-box pair."""
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    if _rbbox_almost_equal(rbox1, rbox2):
        area_inter = area1
    else:
        area_inter = _inter(rbox1, rbox2)
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    if criterion == 0:
        return area_inter / area1
    if criterion == 1:
        return area_inter / area2
    return area_inter


@numba.jit(nopython=True, parallel=True)
def _rotate_iou_kernel_eval(boxes, query_boxes, criterion=-1):
    """Fill an NxK IoU matrix for all box pairs on CPU."""
    n = boxes.shape[0]
    k = query_boxes.shape[0]
    iou = np.zeros((n, k), dtype=np.float32)
    for i in numba.prange(n):
        for j in range(k):
            iou[i, j] = _dev_rotate_iou_eval(boxes[i], query_boxes[j], criterion)
    return iou


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """Compute rotated-box IoU on CPU (name kept for OpenPCDet callers).

    Args:
        boxes: float array [N, 5] as x, y, dx, dy, angle.
        query_boxes: float array [K, 5] in the same format.
        criterion: -1 for IoU, 0/1 for directed IoF, else intersection area.
        device_id: unused; retained for API compatibility with upstream.

    Returns:
        float array [N, K] of pairwise scores.
    """
    del device_id
    box_dtype = boxes.dtype
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    n = boxes.shape[0]
    k = query_boxes.shape[0]
    if n == 0 or k == 0:
        return np.zeros((n, k), dtype=box_dtype)
    iou = _rotate_iou_kernel_eval(boxes, query_boxes, criterion)
    return iou.astype(box_dtype)
