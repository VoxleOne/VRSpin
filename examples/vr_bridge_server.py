#!/usr/bin/env python3
"""VRSpin WebSocket bridge server for Unity / Unreal integration.

Accepts ``frame_update`` messages from a VR engine client, processes
attention queries through VRSpin, and returns ``attention_result``
messages.

Usage::

    python examples/vr_bridge_server.py          # default: localhost:8765
    python examples/vr_bridge_server.py --port 9000

Protocol
--------

**Client → Server** (VR Engine → VRSpin)::

    {
      "type": "frame_update",
      "timestamp": 1616700000.123,
      "user": {
        "head_quaternion": [0.0, 0.1, 0.0, 0.995],
        "position": [1.0, 1.7, 3.0]
      },
      "entities": [
        {
          "id": "fountain",
          "orientation": [0.0, 0.0, 0.0, 1.0],
          "position": [5.0, 0.0, 3.0],
          "type": "object"
        }
      ]
    }

**Server → Client** (VRSpin → VR Engine)::

    {
      "type": "attention_result",
      "timestamp": 1616700000.125,
      "attended_entities": [
        {"id": "fountain", "attention_strength": 0.89, "highlight": true}
      ],
      "npc_updates": [
        {"id": "vendor", "new_orientation": [0.05, 0.1, 0.0, 0.994],
         "state": "aware"}
      ],
      "audio_gains": [
        {"id": "fountain_audio", "gain": 0.72}
      ]
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any, Dict, List

import numpy as np

from vrspin import AttentionCone, NPCAttentionAgent
from vrspin.scene import SceneEntity, AttentionManager


# ---------------------------------------------------------------------------
# Frame processing
# ---------------------------------------------------------------------------


def _build_entities(raw: List[Dict[str, Any]]) -> List[SceneEntity]:
    """Convert raw JSON entity dicts into :class:`SceneEntity` objects."""
    entities: List[SceneEntity] = []
    for item in raw:
        entities.append(
            SceneEntity(
                name=item["id"],
                orientation=item.get("orientation", [0, 0, 0, 1]),
                position=item.get("position", [0, 0, 0]),
                entity_type=item.get("type", "object"),
            )
        )
    return entities


def process_frame(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single ``frame_update`` and return an ``attention_result``.

    This is a pure function: no server state is retained between calls.

    Args:
        msg: Parsed JSON message with ``user`` and ``entities`` keys.

    Returns:
        A dict ready for JSON serialisation.
    """
    user_quat = np.asarray(
        msg["user"]["head_quaternion"], dtype=float,
    )
    entities = _build_entities(msg.get("entities", []))

    # --- Attention queries ------------------------------------------------
    visual_cone = AttentionCone(
        user_quat, half_angle=np.radians(45), falloff="linear",
    )
    audio_cone = AttentionCone(
        user_quat, half_angle=np.radians(90), falloff="cosine",
    )

    attended_entities: List[Dict[str, Any]] = []
    audio_gains: List[Dict[str, Any]] = []
    npc_updates: List[Dict[str, Any]] = []

    for entity in entities:
        # Visual
        vis_strength = visual_cone.attenuation(entity.orientation)
        if vis_strength > 0:
            attended_entities.append({
                "id": entity.name,
                "attention_strength": round(vis_strength, 4),
                "highlight": True,
            })

        # Audio
        gain = audio_cone.attenuation(entity.orientation)
        if entity.entity_type == "audio_source" or gain > 0:
            audio_gains.append({
                "id": entity.name,
                "gain": round(gain, 4),
            })

        # NPC awareness (single-frame; no persistent state kept)
        if entity.entity_type == "npc":
            agent = NPCAttentionAgent(entity, perception_half_angle=np.radians(60))
            aware = agent.is_aware_of(user_quat)
            npc_updates.append({
                "id": entity.name,
                "new_orientation": entity.orientation.tolist(),
                "state": "aware" if aware else "idle",
            })

    return {
        "type": "attention_result",
        "timestamp": time.time(),
        "attended_entities": attended_entities,
        "npc_updates": npc_updates,
        "audio_gains": audio_gains,
    }


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------


async def _handler(websocket: Any) -> None:
    """Handle a single WebSocket client connection."""
    async for raw in websocket:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send(json.dumps({"error": "invalid JSON"}))
            continue

        if msg.get("type") != "frame_update":
            await websocket.send(
                json.dumps({"error": f"unknown message type: {msg.get('type')}"})
            )
            continue

        result = process_frame(msg)
        await websocket.send(json.dumps(result))


async def main(host: str = "localhost", port: int = 8765) -> None:
    """Start the VRSpin WebSocket bridge server.

    Args:
        host: Bind address.
        port: Bind port.
    """
    try:
        import websockets  # noqa: F811
    except ImportError:
        print(
            "ERROR: 'websockets' package is required.\n"
            "       Install it with: pip install websockets"
        )
        return

    print(f"VRSpin bridge listening on ws://{host}:{port}")
    async with websockets.serve(_handler, host, port):
        await asyncio.Future()  # run forever


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VRSpin WebSocket bridge server")
    parser.add_argument("--host", default="localhost", help="Bind address")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(main(args.host, args.port))
