"""Orientation-based attention cone powered by SpinStep quaternion math.

An AttentionCone models a directional perception volume (like a field of view)
as a quaternion orientation plus a half-angle aperture.  SpinStep's
:class:`~spinstep.DiscreteOrientationSet` is used for efficient batch queries
over many entity orientations.
"""

from __future__ import annotations

__all__ = ["AttentionCone"]

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R

from spinstep import DiscreteOrientationSet


class AttentionCone:
    """A directional attention cone defined by a quaternion and a half-angle.

    The cone points in the direction encoded by *orientation* and accepts
    anything within *half_angle_rad* radians of that direction.

    Args:
        orientation: Quaternion ``[x, y, z, w]`` for the cone's pointing
            direction.
        half_angle_rad: Half-aperture of the cone in radians.
        label: Human-readable label (e.g. ``"visual"``, ``"audio"``,
            ``"haptic"``).

    Raises:
        ValueError: If the orientation quaternion has near-zero norm.

    Attributes:
        orientation: Normalised quaternion as a NumPy array of shape ``(4,)``.
        half_angle: Half-aperture in radians.
        label: Modality label string.

    Example::

        import numpy as np
        from vrspin import AttentionCone

        cone = AttentionCone([0, 0, 0, 1], half_angle_rad=np.deg2rad(60))
        print(cone.is_in_cone([0, 0, 0.1, 0.995]))  # True — nearly aligned
    """

    def __init__(
        self,
        orientation: ArrayLike,
        half_angle_rad: float,
        label: str = "visual",
    ) -> None:
        arr = np.asarray(orientation, dtype=float)
        norm = np.linalg.norm(arr)
        if norm < 1e-8:
            raise ValueError("Orientation quaternion must be non-zero")
        self.orientation: np.ndarray = arr / norm
        self.half_angle: float = float(half_angle_rad)
        self.label: str = label

    # ------------------------------------------------------------------
    # Orientation management
    # ------------------------------------------------------------------

    def update_orientation(self, orientation: ArrayLike) -> None:
        """Update the cone's pointing direction.

        Args:
            orientation: New quaternion ``[x, y, z, w]``.
        """
        arr = np.asarray(orientation, dtype=float)
        norm = np.linalg.norm(arr)
        if norm >= 1e-8:
            self.orientation = arr / norm

    # ------------------------------------------------------------------
    # Membership tests
    # ------------------------------------------------------------------

    def is_in_cone(self, target_quat: ArrayLike) -> bool:
        """Return ``True`` if *target_quat* falls within this cone's aperture.

        The check measures the quaternion geodesic distance (angle between the
        two orientations) and compares it against :attr:`half_angle`.

        Args:
            target_quat: Quaternion ``[x, y, z, w]`` to test.

        Returns:
            ``True`` when the angular distance is less than
            :attr:`half_angle`.
        """
        target = np.asarray(target_quat, dtype=float)
        norm = np.linalg.norm(target)
        if norm < 1e-8:
            return False
        target = target / norm
        angle = (R.from_quat(self.orientation).inv() * R.from_quat(target)).magnitude()
        return bool(angle < self.half_angle)

    def filter_within_cone(self, orientation_set: DiscreteOrientationSet) -> np.ndarray:
        """Return indices of orientations from *orientation_set* inside this cone.

        Delegates to :meth:`~spinstep.DiscreteOrientationSet.query_within_angle`
        for efficient batch lookup.

        Args:
            orientation_set: :class:`~spinstep.DiscreteOrientationSet` to
                query.

        Returns:
            Integer index array into ``orientation_set.orientations``.
        """
        return orientation_set.query_within_angle(self.orientation, self.half_angle)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def get_forward_vector(self) -> np.ndarray:
        """Return the 3-D unit vector this cone is pointing toward.

        The local ``+Z`` axis ``[0, 0, 1]`` is rotated by the cone's
        quaternion orientation to produce the world-space forward direction.

        Returns:
            NumPy array of shape ``(3,)``.
        """
        return R.from_quat(self.orientation).apply(np.array([0.0, 0.0, 1.0]))

    def angular_distance_to(self, target_quat: ArrayLike) -> float:
        """Return the angular distance in radians between this cone and *target_quat*.

        Args:
            target_quat: Quaternion ``[x, y, z, w]``.

        Returns:
            Angle in radians ``[0, π]``.
        """
        target = np.asarray(target_quat, dtype=float)
        norm = np.linalg.norm(target)
        if norm < 1e-8:
            return float(np.pi)
        target = target / norm
        return float(
            (R.from_quat(self.orientation).inv() * R.from_quat(target)).magnitude()
        )
