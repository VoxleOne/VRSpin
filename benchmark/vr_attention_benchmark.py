#!/usr/bin/env python3
"""Performance benchmarks for VRSpin attention-cone operations.

Measures throughput of the core per-frame operations:

* Single ``AttentionCone.is_in_cone()`` call
* Single ``AttentionCone.attenuation()`` call
* ``AttentionCone.query_batch()`` over N entities (vectorised)
* ``AttentionManager.update()`` over N entities
* ``MultiHeadAttention.update()`` with 3 heads over N entities

Usage::

    python benchmark/vr_attention_benchmark.py
    python benchmark/vr_attention_benchmark.py --n 1000 --reps 10000
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation as R

from vrspin import AttentionCone
from vrspin.multihead import MultiHeadAttention
from vrspin.scene import AttentionManager, SceneEntity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_quat(rng: np.random.Generator) -> np.ndarray:
    """Return a uniformly random unit quaternion."""
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def _make_entities(n: int, rng: np.random.Generator) -> list[SceneEntity]:
    """Create *n* randomly-oriented ``SceneEntity`` objects."""
    return [
        SceneEntity(
            name=f"entity_{i}",
            orientation=_random_quat(rng),
            position=rng.uniform(-10, 10, 3).tolist(),
        )
        for i in range(n)
    ]


def _bench(label: str, fn: Callable[[], object], reps: int) -> None:
    """Time *fn* over *reps* repetitions and print throughput."""
    # Warm-up
    for _ in range(min(100, reps)):
        fn()

    start = time.perf_counter()
    for _ in range(reps):
        fn()
    elapsed = time.perf_counter() - start

    calls_per_sec = reps / elapsed
    us_per_call = elapsed / reps * 1e6
    print(f"  {label:<55s} {calls_per_sec:>10,.0f} calls/s  ({us_per_call:.2f} µs/call)")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_single_cone(reps: int, rng: np.random.Generator) -> None:
    """Benchmark single-entity cone queries."""
    origin = _random_quat(rng)
    target = _random_quat(rng)
    cone = AttentionCone(origin, half_angle_rad=np.radians(45))

    print("\n── Single-entity queries ──────────────────────────────────────")
    _bench("AttentionCone.is_in_cone(target)", lambda: cone.is_in_cone(target), reps)
    _bench("AttentionCone.contains(target)  [alias]", lambda: cone.contains(target), reps)
    _bench("AttentionCone.attenuation(target) [step]", lambda: cone.attenuation(target), reps)

    cone_linear = AttentionCone(origin, half_angle_rad=np.radians(45), falloff="linear")
    _bench("AttentionCone.attenuation(target) [linear]", lambda: cone_linear.attenuation(target), reps)

    cone_cosine = AttentionCone(origin, half_angle_rad=np.radians(45), falloff="cosine")
    _bench("AttentionCone.attenuation(target) [cosine]", lambda: cone_cosine.attenuation(target), reps)

    _bench("AttentionCone.angular_distance_to(target)", lambda: cone.angular_distance_to(target), reps)


def bench_batch(n_entities: int, reps: int, rng: np.random.Generator) -> None:
    """Benchmark batch cone queries over N entities."""
    origin = _random_quat(rng)
    entity_quats = np.array([_random_quat(rng) for _ in range(n_entities)])
    cone = AttentionCone(origin, half_angle_rad=np.radians(45), falloff="linear")

    print(f"\n── Batch queries (N={n_entities}) ────────────────────────────────────")
    _bench(
        f"AttentionCone.query_batch({n_entities} quats)",
        lambda: cone.query_batch(entity_quats),
        reps,
    )
    _bench(
        f"AttentionCone.query_batch_with_attenuation({n_entities} quats)",
        lambda: cone.query_batch_with_attenuation(entity_quats),
        reps,
    )


def bench_attention_manager(n_entities: int, reps: int, rng: np.random.Generator) -> None:
    """Benchmark AttentionManager.update() over N registered entities."""
    entities = _make_entities(n_entities, rng)
    manager = AttentionManager(entities)
    user_quat = _random_quat(rng)

    print(f"\n── AttentionManager.update() (N={n_entities}) ────────────────────────")
    _bench(
        f"AttentionManager.update({n_entities} entities)",
        lambda: manager.update(user_quat, cone_half_angle=np.radians(45)),
        reps,
    )


def bench_multihead(n_entities: int, reps: int, rng: np.random.Generator) -> None:
    """Benchmark MultiHeadAttention.update() with 3 heads over N entities."""
    entities = _make_entities(n_entities, rng)
    user_quat = _random_quat(rng)
    multi = MultiHeadAttention({
        "visual": AttentionCone(user_quat, half_angle_rad=np.radians(45), falloff="linear"),
        "audio":  AttentionCone(user_quat, half_angle_rad=np.radians(90), falloff="cosine"),
        "haptic": AttentionCone(user_quat, half_angle_rad=np.radians(20)),
    })

    print(f"\n── MultiHeadAttention.update() (3 heads, N={n_entities}) ────────────")
    _bench(
        f"MultiHeadAttention.update({n_entities} entities)",
        lambda: multi.update(user_quat, entities),
        reps,
    )
    _bench(
        "MultiHeadAttention.merge_results(union)",
        lambda: multi.merge_results(strategy="union"),
        reps,
    )


def bench_slerp(reps: int, rng: np.random.Generator) -> None:
    """Benchmark SLERP interpolation."""
    from vrspin import slerp

    q1 = _random_quat(rng)
    q2 = _random_quat(rng)

    print("\n── Quaternion interpolation ───────────────────────────────────")
    _bench("slerp(q1, q2, 0.5)", lambda: slerp(q1, q2, 0.5), reps)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VRSpin attention-cone benchmarks")
    parser.add_argument(
        "--n", type=int, default=100,
        help="Number of scene entities for batch/manager benchmarks (default: 100)",
    )
    parser.add_argument(
        "--reps", type=int, default=5000,
        help="Number of repetitions per benchmark (default: 5000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print(f"  VRSpin Attention-Cone Benchmark")
    print(f"  entities (N) = {args.n}   reps = {args.reps:,}   seed = {args.seed}")
    print("=" * 70)

    bench_single_cone(args.reps, rng)
    bench_batch(args.n, args.reps, rng)
    bench_attention_manager(args.n, args.reps, rng)
    bench_multihead(args.n, args.reps, rng)
    bench_slerp(args.reps, rng)

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
