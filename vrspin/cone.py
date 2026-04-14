"""Orientation-based attention cone powered by SpinStep quaternion math.

An AttentionCone models a directional perception volume (like a field of view)
as a quaternion orientation plus a half-angle aperture.  SpinStep's
:class:`~spinstep.DiscreteOrientationSet` is used for efficient batch queries
over many entity orientations.
"""

from __future__ import annotations

__all__ = ["AttentionCone"]

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from spinstep import DiscreteOrientationSet
from spinstep.utils import (
    is_within_angle_threshold,
    batch_quaternion_angle,
    forward_vector_from_quaternion,
    quaternion_distance as _spinstep_quat_distance,
)


class AttentionCone:
    """A directional attention cone defined by a quaternion and a half-angle.

    The cone points in the direction encoded by *orientation* and accepts
    anything within *half_angle_rad* radians of that direction.

    Supports two construction styles::

        # Original style — positional half_angle_rad
        cone = AttentionCone([0, 0, 0, 1], half_angle_rad=np.deg2rad(60))

        # Instruction-style — keyword half_angle
        cone = AttentionCone([0, 0, 0, 1], half_angle=np.deg2rad(45))

    Args:
        orientation: Quaternion ``[x, y, z, w]`` for the cone's pointing
            direction.
        half_angle_rad: Half-aperture of the cone in radians.
        half_angle: Alias for *half_angle_rad* (used when called as keyword).
        label: Human-readable label (e.g. ``"visual"``, ``"audio"``,
            ``"haptic"``).
        falloff: Attenuation curve applied inside the cone.  One of
            ``'linear'``, ``'cosine'``, or ``None`` (step function —
            1.0 inside, 0.0 outside).

    Raises:
        ValueError: If the orientation quaternion has near-zero norm.

    Attributes:
        orientation: Normalised quaternion as a NumPy array of shape ``(4,)``.
        half_angle: Half-aperture in radians.
        label: Modality label string.
        falloff: Attenuation curve name or ``None``.

    Example::

        import numpy as np
        from vrspin import AttentionCone

        cone = AttentionCone([0, 0, 0, 1], half_angle_rad=np.deg2rad(60))
        print(cone.is_in_cone([0, 0, 0.1, 0.995]))  # True — nearly aligned
        print(cone.contains([0, 0, 0.1, 0.995]))     # True — alias
        print(cone.attenuation([0, 0, 0.1, 0.995]))  # 0.0–1.0 strength
    """

    _VALID_FALLOFFS = {None, "linear", "cosine"}

    def __init__(
        self,
        orientation: ArrayLike,
        half_angle_rad: Optional[float] = None,
        label: str = "visual",
        *,
        half_angle: Optional[float] = None,
        falloff: Optional[str] = None,
    ) -> None:
        arr = np.asarray(orientation, dtype=float)
        norm = np.linalg.norm(arr)
        if norm < 1e-8:
            raise ValueError("Orientation quaternion must be non-zero")
        self.orientation: np.ndarray = arr / norm

        # Accept either half_angle_rad (positional) or half_angle (keyword).
        resolved = half_angle_rad if half_angle_rad is not None else half_angle
        if resolved is None:
            raise TypeError(
                "AttentionCone requires a half-angle: "
                "pass half_angle_rad or half_angle (keyword)"
            )
        self.half_angle: float = float(resolved)

        self.label: str = label

        if falloff not in self._VALID_FALLOFFS:
            raise ValueError(
                f"falloff must be one of {self._VALID_FALLOFFS!r}, got {falloff!r}"
            )
        self.falloff: Optional[str] = falloff

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

    def update_origin(self, new_quat: ArrayLike) -> None:
        """Update the cone's origin orientation.

        Alias for :meth:`update_orientation` matching the instruction-style
        API (``cone.update_origin(new_quat)``).

        Args:
            new_quat: New quaternion ``[x, y, z, w]``.
        """
        self.update_orientation(new_quat)

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
        return bool(
            is_within_angle_threshold(self.orientation, target, self.half_angle)
        )

    def contains(self, target_quat: ArrayLike) -> bool:
        """Return ``True`` if *target_quat* is inside this cone.

        Alias for :meth:`is_in_cone` matching the instruction-style API.

        Args:
            target_quat: Quaternion ``[x, y, z, w]`` to test.
        """
        return self.is_in_cone(target_quat)

    # ------------------------------------------------------------------
    # Attenuation
    # ------------------------------------------------------------------

    def attenuation(self, target_quat: ArrayLike) -> float:
        """Return a ``[0, 1]`` attention strength for *target_quat*.

        The value is ``1.0`` when *target_quat* is exactly aligned with the
        cone origin and decays toward ``0.0`` at the edge.  Targets outside
        the cone always return ``0.0``.

        The decay curve is determined by :attr:`falloff`:

        * ``None`` — step function: ``1.0`` inside, ``0.0`` outside.
        * ``'linear'`` — ``1 - (angle / half_angle)``.
        * ``'cosine'`` — ``cos(angle / half_angle * π/2)``.

        Args:
            target_quat: Quaternion ``[x, y, z, w]``.

        Returns:
            Attention strength in ``[0.0, 1.0]``.
        """
        angle = self.angular_distance_to(target_quat)
        if angle >= self.half_angle:
            return 0.0
        if self.falloff is None:
            return 1.0
        ratio = angle / self.half_angle
        if self.falloff == "linear":
            return float(max(0.0, 1.0 - ratio))
        # cosine
        return float(np.cos(ratio * np.pi / 2.0))

    # ------------------------------------------------------------------
    # Batch queries
    # ------------------------------------------------------------------

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

    def query_batch(self, entity_quats: ArrayLike) -> np.ndarray:
        """Return a boolean mask indicating which quaternions are inside this cone.

        Uses :func:`~spinstep.utils.batch_quaternion_angle` for vectorised
        distance computation.

        Args:
            entity_quats: Array of shape ``(N, 4)`` of ``[x, y, z, w]``
                quaternions.

        Returns:
            Boolean NumPy array of shape ``(N,)``.
        """
        quats = np.atleast_2d(np.asarray(entity_quats, dtype=float))
        origin = self.orientation.reshape(1, 4)
        # batch_quaternion_angle returns shape (1, N)
        angles = batch_quaternion_angle(origin, quats, np).ravel()
        return angles < self.half_angle

    def query_batch_with_attenuation(self, entity_quats: ArrayLike) -> np.ndarray:
        """Return per-entity attenuation values.

        Combines :meth:`query_batch` filtering with :attr:`falloff`-based
        decay.

        Args:
            entity_quats: Array of shape ``(N, 4)`` of ``[x, y, z, w]``
                quaternions.

        Returns:
            NumPy array of shape ``(N,)`` with values in ``[0, 1]``.
            Entries outside the cone are ``0.0``.
        """
        quats = np.atleast_2d(np.asarray(entity_quats, dtype=float))
        origin = self.orientation.reshape(1, 4)
        angles = batch_quaternion_angle(origin, quats, np).ravel()
        inside = angles < self.half_angle
        result = np.zeros(len(angles), dtype=float)
        if self.falloff is None:
            result[inside] = 1.0
        elif self.falloff == "linear":
            result[inside] = np.clip(1.0 - angles[inside] / self.half_angle, 0.0, 1.0)
        else:  # cosine
            result[inside] = np.cos(angles[inside] / self.half_angle * np.pi / 2.0)
        return result

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def get_forward_vector(self) -> np.ndarray:
        """Return the 3-D unit vector this cone is pointing toward.

        The local ``-Z`` axis ``[0, 0, -1]`` is rotated by the cone's
        quaternion orientation to produce the world-space forward direction,
        matching SpinStep's :func:`~spinstep.utils.forward_vector_from_quaternion`
        convention.

        Returns:
            NumPy array of shape ``(3,)``.
        """
        return forward_vector_from_quaternion(self.orientation)

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
        return float(_spinstep_quat_distance(self.orientation, target))
