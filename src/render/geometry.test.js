import { describe, it } from "node:test"
import assert from "node:assert/strict"

import {
  VERTEX_STRIDE, VERTEX_STRIDE_BYTES, POSITION_OFFSET, NORMAL_OFFSET,
  makeBoxGeometry, makeSphereGeometry, makePanelGeometry,
  makeGroundPlaneGeometry, mergeGeometries,
  makeFountainGeometry, makeMarketStandGeometry, makeNPCGeometry,
  makeKnowledgePanelGeometry, makeAudioMarkerGeometry,
} from "./geometry.js"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

describe("geometry — constants", () => {
  it("VERTEX_STRIDE is 6 (position xyz + normal xyz)", () => {
    assert.strictEqual(VERTEX_STRIDE, 6)
  })

  it("VERTEX_STRIDE_BYTES is 24", () => {
    assert.strictEqual(VERTEX_STRIDE_BYTES, 24)
  })

  it("POSITION_OFFSET is 0", () => {
    assert.strictEqual(POSITION_OFFSET, 0)
  })

  it("NORMAL_OFFSET is 12", () => {
    assert.strictEqual(NORMAL_OFFSET, 12)
  })
})

// ---------------------------------------------------------------------------
// Box
// ---------------------------------------------------------------------------

describe("geometry — makeBoxGeometry", () => {
  it("returns Float32Array data", () => {
    const geo = makeBoxGeometry()
    assert.ok(geo.data instanceof Float32Array)
  })

  it("unit box has 36 vertices (12 triangles × 3 verts)", () => {
    const geo = makeBoxGeometry(1, 1, 1)
    assert.strictEqual(geo.vertexCount, 36)
  })

  it("data length matches vertexCount × stride", () => {
    const geo = makeBoxGeometry(2, 3, 4)
    assert.strictEqual(geo.data.length, geo.vertexCount * VERTEX_STRIDE)
  })

  it("box with offset shifts vertex positions", () => {
    const geo = makeBoxGeometry(1, 1, 1, 10, 20, 30)
    // Check first vertex position (should be near offset)
    const x = geo.data[0], y = geo.data[1], z = geo.data[2]
    assert.ok(Math.abs(x - 10) <= 0.5, `x ${x} not near 10`)
    assert.ok(Math.abs(y - 20) <= 0.5, `y ${y} not near 20`)
    assert.ok(Math.abs(z - 30) <= 0.5, `z ${z} not near 30`)
  })

  it("normals are unit length (face normals)", () => {
    const geo = makeBoxGeometry()
    // Check normal of first vertex (offset 3,4,5)
    const nx = geo.data[3], ny = geo.data[4], nz = geo.data[5]
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz)
    assert.ok(Math.abs(len - 1.0) < 1e-5, `normal length ${len}`)
  })

  it("front face normal is [0, 0, 1]", () => {
    const geo = makeBoxGeometry()
    // First triangle is front face
    assert.ok(Math.abs(geo.data[3] - 0) < 1e-5)
    assert.ok(Math.abs(geo.data[4] - 0) < 1e-5)
    assert.ok(Math.abs(geo.data[5] - 1) < 1e-5)
  })
})

// ---------------------------------------------------------------------------
// Sphere
// ---------------------------------------------------------------------------

describe("geometry — makeSphereGeometry", () => {
  it("returns Float32Array data", () => {
    const geo = makeSphereGeometry()
    assert.ok(geo.data instanceof Float32Array)
  })

  it("vertex count = rings × segments × 6 (2 tris × 3 verts each)", () => {
    const rings = 8, segments = 10
    const geo = makeSphereGeometry(0.5, rings, segments)
    assert.strictEqual(geo.vertexCount, rings * segments * 6)
  })

  it("data length matches vertexCount × stride", () => {
    const geo = makeSphereGeometry(1, 6, 8)
    assert.strictEqual(geo.data.length, geo.vertexCount * VERTEX_STRIDE)
  })

  it("normals are approximately unit length", () => {
    const geo = makeSphereGeometry(1, 8, 10)
    // Sample a few normals
    for (let i = 0; i < 5; i++) {
      const base = i * VERTEX_STRIDE * 6 // skip a few vertices
      if (base + 5 >= geo.data.length) break
      const nx = geo.data[base + 3], ny = geo.data[base + 4], nz = geo.data[base + 5]
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz)
      assert.ok(Math.abs(len - 1.0) < 0.05, `normal length ${len} at vertex ${i}`)
    }
  })

  it("sphere with offset shifts positions", () => {
    const geo = makeSphereGeometry(1, 4, 4, 5, 10, 15)
    // All positions should be within radius of offset
    for (let v = 0; v < geo.vertexCount; v++) {
      const base = v * VERTEX_STRIDE
      const x = geo.data[base], y = geo.data[base + 1], z = geo.data[base + 2]
      assert.ok(Math.abs(x - 5) <= 1.1, `x ${x} too far from 5`)
      assert.ok(Math.abs(y - 10) <= 1.1, `y ${y} too far from 10`)
      assert.ok(Math.abs(z - 15) <= 1.1, `z ${z} too far from 15`)
    }
  })
})

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------

describe("geometry — makePanelGeometry", () => {
  it("has 12 vertices (4 triangles: 2 front + 2 back)", () => {
    const geo = makePanelGeometry()
    assert.strictEqual(geo.vertexCount, 12)
  })

  it("front face normals point in +Z", () => {
    const geo = makePanelGeometry()
    // First 6 vertices are front face
    assert.ok(Math.abs(geo.data[5] - 1.0) < 1e-5) // nz of first vertex
  })

  it("back face normals point in -Z", () => {
    const geo = makePanelGeometry()
    // Vertices 6-11 are back face, first back vertex at index 6 * STRIDE
    const backStart = 6 * VERTEX_STRIDE
    assert.ok(Math.abs(geo.data[backStart + 5] - (-1.0)) < 1e-5)
  })
})

// ---------------------------------------------------------------------------
// Ground plane
// ---------------------------------------------------------------------------

describe("geometry — makeGroundPlaneGeometry", () => {
  it("has 6 vertices (2 triangles)", () => {
    const geo = makeGroundPlaneGeometry()
    assert.strictEqual(geo.vertexCount, 6)
  })

  it("all normals point up [0, 1, 0]", () => {
    const geo = makeGroundPlaneGeometry()
    for (let v = 0; v < geo.vertexCount; v++) {
      const base = v * VERTEX_STRIDE
      assert.ok(Math.abs(geo.data[base + 3] - 0) < 1e-5)
      assert.ok(Math.abs(geo.data[base + 4] - 1) < 1e-5)
      assert.ok(Math.abs(geo.data[base + 5] - 0) < 1e-5)
    }
  })

  it("all Y positions are 0", () => {
    const geo = makeGroundPlaneGeometry(10)
    for (let v = 0; v < geo.vertexCount; v++) {
      const y = geo.data[v * VERTEX_STRIDE + 1]
      assert.ok(Math.abs(y) < 1e-5)
    }
  })
})

// ---------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------

describe("geometry — mergeGeometries", () => {
  it("combines vertex counts correctly", () => {
    const a = makeBoxGeometry(1, 1, 1) // 36 verts
    const b = makeBoxGeometry(1, 1, 1) // 36 verts
    const merged = mergeGeometries([a, b])
    assert.strictEqual(merged.vertexCount, 72)
  })

  it("combined data length is sum of parts", () => {
    const a = makeBoxGeometry()
    const b = makePanelGeometry()
    const merged = mergeGeometries([a, b])
    assert.strictEqual(merged.data.length, a.data.length + b.data.length)
  })

  it("preserves data from first geometry", () => {
    const a = makeBoxGeometry()
    const merged = mergeGeometries([a])
    for (let i = 0; i < a.data.length; i++) {
      assert.ok(Math.abs(merged.data[i] - a.data[i]) < 1e-6)
    }
  })
})

// ---------------------------------------------------------------------------
// Entity-specific composites
// ---------------------------------------------------------------------------

describe("geometry — entity composites", () => {
  it("fountain has 3 boxes worth of vertices (108)", () => {
    const geo = makeFountainGeometry()
    assert.strictEqual(geo.vertexCount, 36 * 3)
  })

  it("market stand has 6 boxes (counter + roof + 4 poles)", () => {
    const geo = makeMarketStandGeometry()
    assert.strictEqual(geo.vertexCount, 36 * 6)
  })

  it("NPC has head sphere + body box", () => {
    const geo = makeNPCGeometry()
    const headVerts = 8 * 10 * 6  // sphere: rings × segments × 6
    const bodyVerts = 36            // box
    assert.strictEqual(geo.vertexCount, headVerts + bodyVerts)
  })

  it("knowledge panel has content quad + 4 border boxes", () => {
    const geo = makeKnowledgePanelGeometry()
    const contentVerts = 12  // double-sided panel
    const borderVerts = 36 * 4
    assert.strictEqual(geo.vertexCount, contentVerts + borderVerts)
  })

  it("audio marker is a small sphere", () => {
    const geo = makeAudioMarkerGeometry()
    const expected = 6 * 8 * 6  // rings × segments × 6
    assert.strictEqual(geo.vertexCount, expected)
  })

  it("all composites return valid Float32Array", () => {
    const geos = [
      makeFountainGeometry(),
      makeMarketStandGeometry(),
      makeNPCGeometry(),
      makeKnowledgePanelGeometry(),
      makeAudioMarkerGeometry(),
    ]
    for (const geo of geos) {
      assert.ok(geo.data instanceof Float32Array)
      assert.strictEqual(geo.data.length, geo.vertexCount * VERTEX_STRIDE)
      assert.ok(geo.vertexCount > 0)
    }
  })
})
