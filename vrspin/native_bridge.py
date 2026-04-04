"""Python ctypes bridge to the VRSpin native C library.

Loads ``libvrspin_native.so`` (Linux), ``libvrspin_native.dylib`` (macOS),
or ``vrspin_native.dll`` (Windows) and exposes a Pythonic API that mirrors
:class:`~vrspin.cone.AttentionCone`.

Usage::

    from vrspin.native_bridge import NativeAttentionCone, native_slerp

    cone = NativeAttentionCone([0, 0, 0, 1], half_angle=0.5, falloff="linear")
    print(cone.contains([0.1, 0, 0, 0.995]))     # True/False
    print(cone.attenuation([0.1, 0, 0, 0.995]))   # 0.0–1.0

The native library must be compiled first — see ``vrspin/native/Makefile``.
"""

from __future__ import annotations

__all__ = [
    "NativeAttentionCone",
    "native_slerp",
    "native_forward_vector",
    "native_quat_distance",
    "native_process_frame",
    "load_native_library",
]

import ctypes
import os
import platform
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

# ------------------------------------------------------------------ #
# Library loading                                                      #
# ------------------------------------------------------------------ #

_FALLOFF_MAP = {
    None: 0,        # VRSPIN_FALLOFF_NONE
    "linear": 1,    # VRSPIN_FALLOFF_LINEAR
    "cosine": 2,    # VRSPIN_FALLOFF_COSINE
}


def _lib_filename() -> str:
    """Return the platform-specific shared library filename."""
    system = platform.system()
    if system == "Darwin":
        return "libvrspin_native.dylib"
    if system == "Windows":
        return "vrspin_native.dll"
    return "libvrspin_native.so"


def _search_paths() -> List[Path]:
    """Return candidate directories where the native library might live."""
    here = Path(__file__).resolve().parent
    return [
        here / "native",                 # vrspin/native/
        here,                            # vrspin/
        Path.cwd(),                      # working directory
        Path.cwd() / "vrspin" / "native",
    ]


def load_native_library(path: Optional[Union[str, Path]] = None) -> ctypes.CDLL:
    """Load the VRSpin native shared library.

    Args:
        path: Explicit path to the library file.  If ``None``, the
            function searches standard locations relative to the package.

    Returns:
        A :class:`ctypes.CDLL` handle.

    Raises:
        OSError: If the library cannot be found or loaded.
    """
    if path is not None:
        return ctypes.CDLL(str(path))

    fname = _lib_filename()
    for directory in _search_paths():
        candidate = directory / fname
        if candidate.exists():
            return ctypes.CDLL(str(candidate))

    raise OSError(
        f"Cannot find {fname!r}. Build it first:\n"
        f"  cd vrspin/native && make"
    )


# Lazy singleton — loaded on first use.
_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    """Return the cached library handle, loading it on first call."""
    global _lib
    if _lib is None:
        _lib = load_native_library()
        _setup_signatures(_lib)
    return _lib


def _setup_signatures(lib: ctypes.CDLL) -> None:
    """Declare C function signatures for type-safety."""
    c_double_p = ctypes.POINTER(ctypes.c_double)
    c_int_p = ctypes.POINTER(ctypes.c_int)

    # vrspin_version
    lib.vrspin_version.restype = ctypes.c_char_p
    lib.vrspin_version.argtypes = []

    # vrspin_quat_normalize
    lib.vrspin_quat_normalize.restype = None
    lib.vrspin_quat_normalize.argtypes = [ctypes.c_double * 4]

    # vrspin_quat_distance
    lib.vrspin_quat_distance.restype = ctypes.c_double
    lib.vrspin_quat_distance.argtypes = [ctypes.c_double * 4, ctypes.c_double * 4]

    # vrspin_forward_vector
    lib.vrspin_forward_vector.restype = None
    lib.vrspin_forward_vector.argtypes = [ctypes.c_double * 4, ctypes.c_double * 3]

    # vrspin_slerp
    lib.vrspin_slerp.restype = None
    lib.vrspin_slerp.argtypes = [
        ctypes.c_double * 4, ctypes.c_double * 4,
        ctypes.c_double, ctypes.c_double * 4,
    ]

    # vrspin_cone_create
    lib.vrspin_cone_create.restype = ctypes.c_void_p
    lib.vrspin_cone_create.argtypes = [
        ctypes.c_double * 4, ctypes.c_double, ctypes.c_int,
    ]

    # vrspin_cone_destroy
    lib.vrspin_cone_destroy.restype = None
    lib.vrspin_cone_destroy.argtypes = [ctypes.c_void_p]

    # vrspin_cone_update_origin
    lib.vrspin_cone_update_origin.restype = None
    lib.vrspin_cone_update_origin.argtypes = [ctypes.c_void_p, ctypes.c_double * 4]

    # vrspin_cone_contains
    lib.vrspin_cone_contains.restype = ctypes.c_int
    lib.vrspin_cone_contains.argtypes = [ctypes.c_void_p, ctypes.c_double * 4]

    # vrspin_cone_attenuation
    lib.vrspin_cone_attenuation.restype = ctypes.c_double
    lib.vrspin_cone_attenuation.argtypes = [ctypes.c_void_p, ctypes.c_double * 4]

    # vrspin_cone_query_batch
    lib.vrspin_cone_query_batch.restype = None
    lib.vrspin_cone_query_batch.argtypes = [
        ctypes.c_void_p, c_double_p, ctypes.c_int, c_int_p,
    ]

    # vrspin_cone_query_batch_attenuation
    lib.vrspin_cone_query_batch_attenuation.restype = None
    lib.vrspin_cone_query_batch_attenuation.argtypes = [
        ctypes.c_void_p, c_double_p, ctypes.c_int, c_double_p,
    ]

    # vrspin_process_frame
    lib.vrspin_process_frame.restype = ctypes.c_int
    lib.vrspin_process_frame.argtypes = [
        ctypes.c_double * 4,
        c_double_p, ctypes.c_int,
        ctypes.c_double, ctypes.c_double,
        c_double_p, c_double_p, c_int_p,
    ]


# ------------------------------------------------------------------ #
# Helper: Python array → ctypes array                                  #
# ------------------------------------------------------------------ #

def _to_c4(arr: ArrayLike) -> "ctypes.c_double * 4":
    """Convert a 4-element array-like to a ctypes double[4]."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    out = (ctypes.c_double * 4)()
    for i in range(4):
        out[i] = a[i]
    return out


def _to_c3(arr: ArrayLike) -> "ctypes.c_double * 3":
    """Convert a 3-element array-like to a ctypes double[3]."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    out = (ctypes.c_double * 3)()
    for i in range(3):
        out[i] = a[i]
    return out


# ------------------------------------------------------------------ #
# Public utilities                                                      #
# ------------------------------------------------------------------ #

def native_quat_distance(q1: ArrayLike, q2: ArrayLike) -> float:
    """Compute angular distance between two quaternions using the native library.

    Args:
        q1: Quaternion ``[x, y, z, w]``.
        q2: Quaternion ``[x, y, z, w]``.

    Returns:
        Angle in radians in ``[0, pi]``.
    """
    lib = _get_lib()
    return float(lib.vrspin_quat_distance(_to_c4(q1), _to_c4(q2)))


def native_forward_vector(q: ArrayLike) -> np.ndarray:
    """Extract the forward direction vector from a quaternion using the native library.

    Uses the -Z convention: identity [0,0,0,1] yields [0,0,-1].

    Args:
        q: Quaternion ``[x, y, z, w]``.

    Returns:
        Direction vector as a NumPy array of shape ``(3,)``.
    """
    lib = _get_lib()
    out = (ctypes.c_double * 3)()
    lib.vrspin_forward_vector(_to_c4(q), out)
    return np.array([out[0], out[1], out[2]])


def native_slerp(q1: ArrayLike, q2: ArrayLike, t: float) -> np.ndarray:
    """Spherical linear interpolation using the native library.

    Args:
        q1: Start quaternion ``[x, y, z, w]``.
        q2: End quaternion ``[x, y, z, w]``.
        t: Interpolation factor in ``[0, 1]``.

    Returns:
        Interpolated quaternion as a NumPy array of shape ``(4,)``.
    """
    lib = _get_lib()
    out = (ctypes.c_double * 4)()
    lib.vrspin_slerp(_to_c4(q1), _to_c4(q2), ctypes.c_double(t), out)
    return np.array([out[0], out[1], out[2], out[3]])


def native_process_frame(
    user_quat: ArrayLike,
    entity_orientations: ArrayLike,
    visual_half_angle: float,
    audio_half_angle: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Process a full attention frame using the native library.

    Args:
        user_quat: User head quaternion ``[x, y, z, w]``.
        entity_orientations: Array of shape ``(N, 4)`` entity quaternions.
        visual_half_angle: Visual cone half-angle in radians.
        audio_half_angle: Audio cone half-angle in radians.

    Returns:
        Tuple of ``(visual_strengths, audio_gains, visual_attended, count)``
        where *count* is the number of visually attended entities.
    """
    lib = _get_lib()
    quats = np.asarray(entity_orientations, dtype=np.float64)
    if quats.ndim == 1:
        quats = quats.reshape(1, 4)
    n = quats.shape[0]

    vis_str = (ctypes.c_double * n)()
    aud_gain = (ctypes.c_double * n)()
    vis_att = (ctypes.c_int * n)()

    count = lib.vrspin_process_frame(
        _to_c4(user_quat),
        quats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        ctypes.c_double(visual_half_angle),
        ctypes.c_double(audio_half_angle),
        ctypes.cast(vis_str, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(aud_gain, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(vis_att, ctypes.POINTER(ctypes.c_int)),
    )

    return (
        np.array(vis_str[:]),
        np.array(aud_gain[:]),
        np.array(vis_att[:], dtype=bool),
        count,
    )


# ------------------------------------------------------------------ #
# NativeAttentionCone — drop-in for AttentionCone (hot path only)      #
# ------------------------------------------------------------------ #

class NativeAttentionCone:
    """A native-C-backed attention cone, API-compatible with :class:`~vrspin.cone.AttentionCone`.

    This class wraps the compiled ``libvrspin_native`` shared library.
    It provides the same public methods as the Python
    :class:`~vrspin.cone.AttentionCone` but executes all math in optimised
    C, suitable for production VR loops with sub-millisecond budgets.

    Args:
        orientation: Centre quaternion ``[x, y, z, w]``.
        half_angle: Half-aperture in radians.
        falloff: ``'linear'``, ``'cosine'``, or ``None``.

    Example::

        from vrspin.native_bridge import NativeAttentionCone
        import numpy as np

        cone = NativeAttentionCone([0, 0, 0, 1], half_angle=0.5)
        print(cone.contains([0.1, 0, 0, 0.995]))
    """

    def __init__(
        self,
        orientation: ArrayLike,
        half_angle: float,
        falloff: Optional[str] = None,
    ) -> None:
        self._lib = _get_lib()
        falloff_int = _FALLOFF_MAP.get(falloff)
        if falloff_int is None:
            raise ValueError(
                f"falloff must be one of {set(_FALLOFF_MAP)!r}, got {falloff!r}"
            )
        self._handle = self._lib.vrspin_cone_create(
            _to_c4(orientation),
            ctypes.c_double(half_angle),
            ctypes.c_int(falloff_int),
        )
        if not self._handle:
            raise MemoryError("Failed to allocate native attention cone")

        self.half_angle = half_angle
        self.falloff = falloff

    def __del__(self) -> None:
        if hasattr(self, "_handle") and self._handle and hasattr(self, "_lib"):
            self._lib.vrspin_cone_destroy(self._handle)
            self._handle = None

    # -- Orientation management -----------------------------------------

    def update_origin(self, new_quat: ArrayLike) -> None:
        """Update the cone's pointing direction.

        Args:
            new_quat: New centre quaternion ``[x, y, z, w]``.
        """
        self._lib.vrspin_cone_update_origin(self._handle, _to_c4(new_quat))

    update_orientation = update_origin  # alias

    # -- Membership tests -----------------------------------------------

    def contains(self, target_quat: ArrayLike) -> bool:
        """Return ``True`` if *target_quat* falls inside this cone.

        Args:
            target_quat: Quaternion ``[x, y, z, w]``.
        """
        return bool(self._lib.vrspin_cone_contains(self._handle, _to_c4(target_quat)))

    is_in_cone = contains  # alias

    # -- Attenuation ----------------------------------------------------

    def attenuation(self, target_quat: ArrayLike) -> float:
        """Return attention strength in ``[0, 1]``.

        Args:
            target_quat: Quaternion ``[x, y, z, w]``.
        """
        return float(
            self._lib.vrspin_cone_attenuation(self._handle, _to_c4(target_quat))
        )

    # -- Batch queries --------------------------------------------------

    def query_batch(self, entity_quats: ArrayLike) -> np.ndarray:
        """Return boolean mask of which quaternions are inside the cone.

        Args:
            entity_quats: Array of shape ``(N, 4)``.

        Returns:
            Boolean NumPy array of shape ``(N,)``.
        """
        quats = np.atleast_2d(np.asarray(entity_quats, dtype=np.float64))
        n = quats.shape[0]
        results = (ctypes.c_int * n)()
        self._lib.vrspin_cone_query_batch(
            self._handle,
            quats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n),
            ctypes.cast(results, ctypes.POINTER(ctypes.c_int)),
        )
        return np.array(results[:], dtype=bool)

    def query_batch_with_attenuation(self, entity_quats: ArrayLike) -> np.ndarray:
        """Return per-entity attenuation values.

        Args:
            entity_quats: Array of shape ``(N, 4)``.

        Returns:
            NumPy array of shape ``(N,)`` with values in ``[0, 1]``.
        """
        quats = np.atleast_2d(np.asarray(entity_quats, dtype=np.float64))
        n = quats.shape[0]
        results = (ctypes.c_double * n)()
        self._lib.vrspin_cone_query_batch_attenuation(
            self._handle,
            quats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n),
            ctypes.cast(results, ctypes.POINTER(ctypes.c_double)),
        )
        return np.array(results[:])

    # -- Forward vector -------------------------------------------------

    def get_forward_vector(self) -> np.ndarray:
        """Return the 3-D unit vector this cone points toward.

        Returns:
            NumPy array of shape ``(3,)``.
        """
        # Re-read orientation from the cone handle by computing forward vector
        # of the identity and then using the C function directly.
        # We need the current orientation — but we only store the handle.
        # Workaround: compute via the library.
        # Since we can't read the orientation from the opaque handle,
        # we track it on the Python side as well.
        raise NotImplementedError(
            "get_forward_vector() is not yet supported on NativeAttentionCone. "
            "Use native_forward_vector(quat) instead."
        )

    def __repr__(self) -> str:
        return (
            f"NativeAttentionCone(half_angle={self.half_angle:.4f}, "
            f"falloff={self.falloff!r})"
        )
