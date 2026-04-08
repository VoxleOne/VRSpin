/**
 * Plaza scene — JavaScript port of vrspin/plaza.py.
 *
 * Constructs the complete plaza scene graph matching the Python VirtualPlaza:
 *   plaza (root)
 *   ├── north_zone   [0°]   — Fountain / Elena NPC / FountainAmbience
 *   ├── nw_zone     [70°]   — VR Art knowledge panel
 *   ├── west_zone   [85°]   — Digital Sculpture knowledge panel
 *   └── east_zone  [-70°]   — MarketStand / Kai NPC / MarketMusic
 *
 * All orientations are quaternion Y-rotations matching plaza.py exactly.
 *
 * @module scene/plaza
 */

import { quatFromYDeg, SpinNode } from "../core/spinstep.js"
import { createPlazaNode } from "../core/node.js"

/**
 * Build the complete plaza scene.
 * Returns the flat array of PlazaNodes and the SpinStep tree root.
 *
 * @returns {{ nodes: object[], root: SpinNode }}
 */
export function buildPlazaScene() {
  // Orientations — matching plaza.py _y_rot() calls
  const north = quatFromYDeg(0)
  const nw = quatFromYDeg(70)
  const west = quatFromYDeg(85)
  const east = quatFromYDeg(-70)

  // ------ Scene entities ------

  const fountain = createPlazaNode({
    id: "Fountain",
    orientation: north,
    entityType: "object",
    metadata: {
      description: "A cascading stone fountain at the plaza centre.",
      color: [0.35, 0.55, 0.85],  // blue-stone
      renderRadius: 5,
      renderYOffset: 0,
    },
  })

  const marketStand = createPlazaNode({
    id: "MarketStand",
    orientation: east,
    entityType: "object",
    metadata: {
      description: "A colourful merchant stall selling VR goods.",
      color: [0.85, 0.55, 0.2],  // warm orange-brown
      renderRadius: 5,
      renderYOffset: 0,
    },
  })

  const elena = createPlazaNode({
    id: "Elena",
    orientation: north,
    entityType: "npc",
    metadata: {
      greeting: "Welcome to the plaza! The fountain has been here for ages.",
      perceptionHalfAngle: 120 * Math.PI / 180,
      slerpSpeed: 0.5,
      color: [0.75, 0.3, 0.5],  // rose pink
      renderRadius: 4,           // closer than objects
      renderYOffset: 0,
    },
  })

  const kai = createPlazaNode({
    id: "Kai",
    orientation: east,
    entityType: "npc",
    metadata: {
      greeting: "Step right up! Best wares in the metaverse!",
      perceptionHalfAngle: 100 * Math.PI / 180,
      slerpSpeed: 0.5,
      color: [0.3, 0.75, 0.4],  // merchant green
      renderRadius: 4,           // closer than objects
      renderYOffset: 0,
    },
  })

  const fountainAmbience = createPlazaNode({
    id: "FountainAmbience",
    orientation: north,
    entityType: "audio",
    metadata: {
      content: "soft water sounds",
      baseVolume: 0.7,
      renderRadius: 5,
      renderYOffset: -0.5,       // ground level
    },
  })

  const marketMusic = createPlazaNode({
    id: "MarketMusic",
    orientation: east,
    entityType: "audio",
    metadata: {
      content: "lively merchant tune",
      baseVolume: 0.9,
      renderRadius: 5,
      renderYOffset: -0.5,       // ground level
    },
  })

  const vrArtPanel = createPlazaNode({
    id: "VR Art",
    orientation: nw,
    entityType: "panel",
    metadata: {
      pages: [
        {
          heading: "Virtual Reality Art",
          body: "Creating immersive art in virtual worlds — orientation shapes what you perceive.",
          triggerAngleDeg: 0,
        },
        {
          heading: "Interaction Design in VR",
          body: "How head orientation drives UX — menus replaced by looking.",
          triggerAngleDeg: 10,
        },
      ],
      currentPage: 0,
      color: [0.15, 0.4, 0.85],  // deep blue
      renderRadius: 6,            // farther — billboard distance
      renderYOffset: 1.0,         // elevated — floating display
    },
  })

  const digitalSculpturePanel = createPlazaNode({
    id: "Digital Sculpture",
    orientation: west,
    entityType: "panel",
    metadata: {
      pages: [
        {
          heading: "Digital Sculpture",
          body: "3-D sculptural works that exist purely in virtual space.",
          triggerAngleDeg: 0,
        },
      ],
      currentPage: 0,
      color: [0.85, 0.35, 0.85],  // purple
      renderRadius: 6,             // farther — billboard distance
      renderYOffset: 1.0,          // elevated — floating display
    },
  })

  // ------ Flat node list ------
  const nodes = [
    fountain, marketStand,
    elena, kai,
    fountainAmbience, marketMusic,
    vrArtPanel, digitalSculpturePanel,
  ]

  // ------ SpinStep Node Tree (for traversal) ------
  const northZone = new SpinNode("north_zone", north, [
    new SpinNode("Fountain", north),
    new SpinNode("Elena", north),
    new SpinNode("FountainAmbience", north),
  ])

  const nwZone = new SpinNode("northwest_zone", nw, [
    new SpinNode("VR Art", nw),
  ])

  const westZone = new SpinNode("west_zone", west, [
    new SpinNode("Digital Sculpture", west),
  ])

  const eastZone = new SpinNode("east_zone", east, [
    new SpinNode("MarketStand", east),
    new SpinNode("Kai", east),
    new SpinNode("MarketMusic", east),
  ])

  const root = new SpinNode("plaza", [0, 0, 0, 1], [
    northZone, nwZone, westZone, eastZone,
  ])

  return { nodes, root }
}
