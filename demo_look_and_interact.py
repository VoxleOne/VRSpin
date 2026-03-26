#!/usr/bin/env python3
"""SpinStep VR Demo: "Look & Interact"

Demonstrates an orientation-driven virtual plaza where all perception and
interaction emerge from the user's head orientation — no menus, no clicks.

Run::

    python demo_look_and_interact.py

The demo simulates a complete interaction loop:

1. User Maya enters the plaza facing north (toward the Fountain).
2. Elena (fountain NPC) notices Maya and greets her.
3. Audio and object highlights respond to Maya's visual and audio cones.
4. Maya pivots left — a VR Art knowledge panel becomes visible.
5. Maya rotates further left — the panel advances to a new page.
6. Maya turns right toward the east Market Stand — Kai (market NPC) engages.
7. Maya looks directly at the fountain — haptic feedback fires.
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from vrspin import VirtualPlaza, VRUser

# ANSI colour helpers (gracefully degrade on terminals without colour support)
_RESET = "\033[0m"
_BOLD = "\033[1m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_MAGENTA = "\033[95m"
_BLUE = "\033[94m"
_RED = "\033[91m"

_MODALITY_COLOURS = {
    "visual": _GREEN,
    "audio": _YELLOW,
    "haptic": _MAGENTA,
    "npc": _CYAN,
    "knowledge": _BLUE,
    "system": _RESET,
}


def _print_header(text: str) -> None:
    print(f"\n{_BOLD}{_CYAN}{'─' * 60}{_RESET}")
    print(f"{_BOLD}{_CYAN} {text}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'─' * 60}{_RESET}")


def _print_event(event) -> None:
    colour = _MODALITY_COLOURS.get(event.modality, _RESET)
    print(f"  {colour}{event}{_RESET}")


def _print_step(step: int, description: str) -> None:
    print(f"\n{_BOLD}[Step {step:02d}]{_RESET} {description}")


def _y_rot(deg: float) -> np.ndarray:
    """Quaternion for a rotation of *deg* degrees around +Y (yaw)."""
    return R.from_euler("y", deg, degrees=True).as_quat()


def run_demo() -> None:
    _print_header("SpinStep VR Demo  —  'Look & Interact'")
    print(
        "\n"
        "  Attention drives perception. Objects, audio, NPCs, and virtual knowledge\n"
        "  appear based on orientation cones powered by SpinStep quaternions.\n"
        "\n"
        "  Tech: Python · SpinStep · NumPy · SciPy (no VR headset needed — simulated)\n"
    )

    plaza = VirtualPlaza()
    user = VRUser("Maya")

    # -----------------------------------------------------------------------
    # Step 1 — User enters the plaza facing NORTH (toward the Fountain)
    # -----------------------------------------------------------------------
    _print_step(1, "Maya enters the plaza facing NORTH (toward the Fountain)")
    user.set_orientation(_y_rot(0.0))
    events = plaza.tick(user)
    for e in events:
        _print_event(e)

    # One extra tick so Elena finishes turning
    for e in plaza.tick(user):
        _print_event(e)

    # -----------------------------------------------------------------------
    # Step 2 — Maya pivots left (15°) — VR Art panel just comes into view
    # -----------------------------------------------------------------------
    _print_step(2, "Maya pivots LEFT ~15° — VR Art knowledge panel becomes visible")
    user.set_orientation(_y_rot(15.0))
    events = plaza.tick(user)
    for e in events:
        _print_event(e)

    # -----------------------------------------------------------------------
    # Step 3 — Maya rotates to 70° — fully centred on VR Art panel
    # -----------------------------------------------------------------------
    _print_step(3, "Maya rotates further LEFT to 70° — panel content updates")
    user.set_orientation(_y_rot(70.0))
    events = plaza.tick(user)
    for e in events:
        _print_event(e)

    # -----------------------------------------------------------------------
    # Step 4 — Maya turns to 85° — Digital Sculpture panel appears
    # -----------------------------------------------------------------------
    _print_step(4, "Maya turns hard LEFT to 85° — Digital Sculpture panel")
    user.set_orientation(_y_rot(85.0))
    events = plaza.tick(user)
    for e in events:
        _print_event(e)

    # -----------------------------------------------------------------------
    # Step 5 — Maya turns to face EAST toward the Market Stand
    # -----------------------------------------------------------------------
    _print_step(5, "Maya turns RIGHT toward the East Market Stand (−70°)")
    user.set_orientation(_y_rot(-70.0))
    events = plaza.tick(user)
    for e in events:
        _print_event(e)

    # Run extra ticks so market NPC engages
    for _ in range(2):
        for e in plaza.tick(user):
            _print_event(e)

    # -----------------------------------------------------------------------
    # Step 6 — Maya looks directly at the Fountain (haptic trigger)
    # -----------------------------------------------------------------------
    _print_step(6, "Maya faces NORTH directly at the Fountain (haptic pulse)")
    user.set_orientation(_y_rot(0.0))
    # Haptic cone is narrow (30°) — set orientation very precisely
    events = plaza.tick(user)
    for e in events:
        _print_event(e)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    _print_header("Demo Complete")
    print(
        "\n"
        "  All interactions above were driven purely by head orientation.\n"
        "  No menus, no clicks — just natural turning and looking.\n"
        "\n"
        "  SpinStep quaternion primitives used:\n"
        "    · Node           — scene-graph entities with orientation\n"
        "    · QuaternionDepthIterator — orientation-aligned tree traversal\n"
        "    · DiscreteOrientationSet  — batch cone membership queries\n"
        "    · AttentionCone  — multi-modal perception (visual / audio / haptic)\n"
    )


if __name__ == "__main__":
    run_demo()
