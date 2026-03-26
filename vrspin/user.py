"""VR User with multi-modal attention cones.

The :class:`VRUser` holds a head-orientation quaternion and maintains three
independent :class:`~vrspin.cone.AttentionCone` instances — one per sensory
modality — matching the *multi-head SpinStep* concept from the design spec:

* **visual** — narrow cone (60°), tight gaze focus
* **audio** — wide cone (120°), peripheral sound awareness
* **haptic** — very narrow cone (30°), precise touch-feedback zone
"""

from __future__ import annotations

__all__ = ["VRUser"]

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R

from .cone import AttentionCone


class VRUser:
    """A VR user whose perception is driven by quaternion head-orientation.

    The user carries three :class:`~vrspin.cone.AttentionCone` objects
    (visual, audio, haptic) that are all updated together whenever
    :meth:`set_orientation` is called.

    Args:
        name: Display name for this user.
        orientation: Initial head orientation quaternion ``[x, y, z, w]``.
            Defaults to the identity rotation (facing ``+Z``).

    Attributes:
        name: User identifier.
        orientation: Current head orientation as a normalised NumPy array.
        visual_cone: Narrow visual attention cone (half-angle 60°).
        audio_cone: Wide auditory attention cone (half-angle 120°).
        haptic_cone: Tight haptic feedback cone (half-angle 30°).

    Example::

        import numpy as np
        from scipy.spatial.transform import Rotation as R
        from vrspin import VRUser

        user = VRUser("Alice")
        # Rotate 45° left (around Y axis)
        q = R.from_euler("y", 45, degrees=True).as_quat()
        user.set_orientation(q)
        print(user.get_forward_vector())
    """

    VISUAL_HALF_ANGLE: float = np.deg2rad(60.0)
    AUDIO_HALF_ANGLE: float = np.deg2rad(120.0)
    HAPTIC_HALF_ANGLE: float = np.deg2rad(30.0)

    def __init__(
        self,
        name: str,
        orientation: ArrayLike = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        self.name: str = name
        arr = np.asarray(orientation, dtype=float)
        norm = np.linalg.norm(arr)
        if norm < 1e-8:
            raise ValueError("Initial orientation quaternion must be non-zero")
        self.orientation: np.ndarray = arr / norm

        self.visual_cone = AttentionCone(self.orientation, self.VISUAL_HALF_ANGLE, label="visual")
        self.audio_cone = AttentionCone(self.orientation, self.AUDIO_HALF_ANGLE, label="audio")
        self.haptic_cone = AttentionCone(self.orientation, self.HAPTIC_HALF_ANGLE, label="haptic")

    # ------------------------------------------------------------------
    # Orientation control
    # ------------------------------------------------------------------

    def set_orientation(self, orientation: ArrayLike) -> None:
        """Update head orientation and all three attention cones.

        Args:
            orientation: New quaternion ``[x, y, z, w]``.
        """
        arr = np.asarray(orientation, dtype=float)
        norm = np.linalg.norm(arr)
        if norm < 1e-8:
            return
        self.orientation = arr / norm
        for cone in (self.visual_cone, self.audio_cone, self.haptic_cone):
            cone.update_orientation(self.orientation)

    def rotate_by(self, delta_quat: ArrayLike) -> None:
        """Compose the current orientation with *delta_quat*.

        Equivalent to applying an additional rotation on top of the current
        head pose, useful for simulating continuous head-turns.

        Args:
            delta_quat: Quaternion ``[x, y, z, w]`` describing the additional
                rotation.
        """
        new_rot = R.from_quat(self.orientation) * R.from_quat(delta_quat)
        self.set_orientation(new_rot.as_quat())

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_forward_vector(self) -> np.ndarray:
        """Return the 3-D world-space direction the user is looking toward.

        Returns:
            Unit vector of shape ``(3,)``.
        """
        return self.visual_cone.get_forward_vector()

    def sees(self, target_quat: ArrayLike) -> bool:
        """Return ``True`` if *target_quat* is inside the visual cone.

        Args:
            target_quat: Orientation of the target entity.
        """
        return self.visual_cone.is_in_cone(target_quat)

    def hears(self, target_quat: ArrayLike) -> bool:
        """Return ``True`` if *target_quat* is inside the audio cone.

        Args:
            target_quat: Orientation of the audio source.
        """
        return self.audio_cone.is_in_cone(target_quat)

    def feels(self, target_quat: ArrayLike) -> bool:
        """Return ``True`` if *target_quat* is inside the haptic cone.

        Args:
            target_quat: Orientation of the haptic source.
        """
        return self.haptic_cone.is_in_cone(target_quat)

    def cone_for(self, modality: str) -> AttentionCone:
        """Return the :class:`~vrspin.cone.AttentionCone` for *modality*.

        Args:
            modality: One of ``"visual"``, ``"audio"``, or ``"haptic"``.

        Raises:
            KeyError: If *modality* is not recognised.
        """
        cones = {
            "visual": self.visual_cone,
            "audio": self.audio_cone,
            "haptic": self.haptic_cone,
        }
        if modality not in cones:
            raise KeyError(f"Unknown modality '{modality}'. Choose from {list(cones)}")
        return cones[modality]

    def __repr__(self) -> str:
        fwd = self.get_forward_vector()
        return (
            f"VRUser({self.name!r}, forward=[{fwd[0]:.2f}, {fwd[1]:.2f}, {fwd[2]:.2f}])"
        )
