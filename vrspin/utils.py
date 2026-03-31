"""VRSpin utility functions.

Provides helper functions that complement the quaternion utilities in
:mod:`spinstep.utils`, most notably :func:`slerp` for smooth quaternion
interpolation.
"""

from __future__ import annotations

__all__ = ["slerp"]

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R, Slerp as ScipySlerp


def slerp(q1: ArrayLike, q2: ArrayLike, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions.

    Args:
        q1: Start quaternion ``[x, y, z, w]``.
        q2: End quaternion ``[x, y, z, w]``.
        t: Interpolation parameter in ``[0, 1]``.  ``0`` returns *q1*,
            ``1`` returns *q2*.

    Returns:
        Interpolated quaternion as a NumPy array of shape ``(4,)`` in
        ``[x, y, z, w]`` order.

    Example::

        from vrspin.utils import slerp
        result = slerp([0, 0, 0, 1], [0, 0.707, 0, 0.707], 0.5)
    """
    a = np.asarray(q1, dtype=float)
    b = np.asarray(q2, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    key_rots = R.from_quat(np.stack([a, b]))
    interp = ScipySlerp([0.0, 1.0], key_rots)
    return interp([float(t)])[0].as_quat()
