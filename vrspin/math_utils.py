"""Quaternion mathematical utilities.

Standalone functions for common quaternion operations used throughout the
VRSpin attention system.  All quaternions use the ``[x, y, z, w]`` convention.
"""

from __future__ import annotations

__all__ = [
    "quaternion_distance",
    "forward_vector_from_quaternion",
    "direction_to_quaternion",
    "slerp",
]

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp as ScipySlerp


def quaternion_distance(q1: ArrayLike, q2: ArrayLike) -> float:
    """Angular distance in radians between two orientations.

    Args:
        q1: First quaternion ``[x, y, z, w]``.
        q2: Second quaternion ``[x, y, z, w]``.

    Returns:
        Angle in radians in the range ``[0, π]``.
    """
    a = np.asarray(q1, dtype=float)
    b = np.asarray(q2, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return float(np.pi)
    a, b = a / na, b / nb
    return float((R.from_quat(a).inv() * R.from_quat(b)).magnitude())


def forward_vector_from_quaternion(q: ArrayLike) -> np.ndarray:
    """Derive the forward direction vector from a quaternion.

    Rotates the local ``+Z`` axis ``[0, 0, 1]`` by *q* to produce the
    world-space forward direction.  Intended for rendering and debug
    visualisation — **not** for primary attention comparisons.

    Args:
        q: Quaternion ``[x, y, z, w]``.

    Returns:
        Unit vector of shape ``(3,)``.
    """
    arr = np.asarray(q, dtype=float)
    norm = np.linalg.norm(arr)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    return R.from_quat(arr / norm).apply(np.array([0.0, 0.0, 1.0]))


def direction_to_quaternion(direction: ArrayLike) -> np.ndarray:
    """Convert a 3-D direction vector to a quaternion orientation.

    The returned quaternion rotates the local ``+Z`` axis to align with
    *direction*.

    Args:
        direction: Target direction vector ``[x, y, z]``.

    Returns:
        Unit quaternion ``[x, y, z, w]`` of shape ``(4,)``.
    """
    d = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    d = d / norm
    forward = np.array([0.0, 0.0, 1.0])
    dot = np.clip(np.dot(forward, d), -1.0, 1.0)
    if dot > 1.0 - 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot < -1.0 + 1e-8:
        # 180° rotation — pick an arbitrary perpendicular axis
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(forward, d)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(dot)
    return R.from_rotvec(axis * angle).as_quat()


def slerp(q1: ArrayLike, q2: ArrayLike, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions.

    Args:
        q1: Start quaternion ``[x, y, z, w]``.
        q2: End quaternion ``[x, y, z, w]``.
        t: Interpolation parameter in ``[0, 1]``.

    Returns:
        Interpolated unit quaternion ``[x, y, z, w]`` of shape ``(4,)``.
    """
    a = np.asarray(q1, dtype=float)
    b = np.asarray(q2, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    a, b = a / na, b / nb
    key_rots = R.from_quat(np.stack([a, b]))
    interp = ScipySlerp([0.0, 1.0], key_rots)
    result = interp([float(t)])[0]
    return result.as_quat()
