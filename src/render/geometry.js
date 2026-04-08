/**
 * Geometry generators — interleaved position + normal vertex data.
 *
 * All generators return { data: Float32Array, vertexCount: number }.
 * Vertex format: [px, py, pz, nx, ny, nz] — 6 floats per vertex.
 * Stride: 24 bytes. Position offset: 0. Normal offset: 12.
 *
 * Composite entity generators (fountain, market stand, NPC, panel, audio)
 * build recognizable shapes from primitive box/sphere building blocks.
 *
 * @module render/geometry
 */

/** Floats per vertex (position xyz + normal xyz). */
export const VERTEX_STRIDE = 6

/** Stride in bytes for vertex attribute setup. */
export const VERTEX_STRIDE_BYTES = VERTEX_STRIDE * 4

/** Byte offset of position attribute. */
export const POSITION_OFFSET = 0

/** Byte offset of normal attribute. */
export const NORMAL_OFFSET = 12

// ---------------------------------------------------------------------------
// Primitive helpers
// ---------------------------------------------------------------------------

function pushTriFlat(arr, p1, p2, p3, nx, ny, nz) {
  arr.push(
    p1[0], p1[1], p1[2], nx, ny, nz,
    p2[0], p2[1], p2[2], nx, ny, nz,
    p3[0], p3[1], p3[2], nx, ny, nz,
  )
}

function pushTriSmooth(arr, p1, n1, p2, n2, p3, n3) {
  arr.push(
    p1[0], p1[1], p1[2], n1[0], n1[1], n1[2],
    p2[0], p2[1], p2[2], n2[0], n2[1], n2[2],
    p3[0], p3[1], p3[2], n3[0], n3[1], n3[2],
  )
}

// ---------------------------------------------------------------------------
// Box
// ---------------------------------------------------------------------------

/**
 * Box geometry with flat-shaded face normals.
 *
 * @param {number} sx - width
 * @param {number} sy - height
 * @param {number} sz - depth
 * @param {number} ox - X offset
 * @param {number} oy - Y offset
 * @param {number} oz - Z offset
 * @returns {{ data: Float32Array, vertexCount: number }}
 */
export function makeBoxGeometry(sx = 1, sy = 1, sz = 1, ox = 0, oy = 0, oz = 0) {
  const hx = sx / 2, hy = sy / 2, hz = sz / 2
  const verts = []
  const p = (x, y, z) => [x + ox, y + oy, z + oz]

  // Front (+Z)
  pushTriFlat(verts, p(-hx, -hy, hz), p(hx, -hy, hz), p(hx, hy, hz), 0, 0, 1)
  pushTriFlat(verts, p(-hx, -hy, hz), p(hx, hy, hz), p(-hx, hy, hz), 0, 0, 1)
  // Back (-Z)
  pushTriFlat(verts, p(-hx, -hy, -hz), p(-hx, hy, -hz), p(hx, hy, -hz), 0, 0, -1)
  pushTriFlat(verts, p(-hx, -hy, -hz), p(hx, hy, -hz), p(hx, -hy, -hz), 0, 0, -1)
  // Top (+Y)
  pushTriFlat(verts, p(-hx, hy, -hz), p(-hx, hy, hz), p(hx, hy, hz), 0, 1, 0)
  pushTriFlat(verts, p(-hx, hy, -hz), p(hx, hy, hz), p(hx, hy, -hz), 0, 1, 0)
  // Bottom (-Y)
  pushTriFlat(verts, p(-hx, -hy, -hz), p(hx, -hy, -hz), p(hx, -hy, hz), 0, -1, 0)
  pushTriFlat(verts, p(-hx, -hy, -hz), p(hx, -hy, hz), p(-hx, -hy, hz), 0, -1, 0)
  // Right (+X)
  pushTriFlat(verts, p(hx, -hy, -hz), p(hx, hy, -hz), p(hx, hy, hz), 1, 0, 0)
  pushTriFlat(verts, p(hx, -hy, -hz), p(hx, hy, hz), p(hx, -hy, hz), 1, 0, 0)
  // Left (-X)
  pushTriFlat(verts, p(-hx, -hy, -hz), p(-hx, -hy, hz), p(-hx, hy, hz), -1, 0, 0)
  pushTriFlat(verts, p(-hx, -hy, -hz), p(-hx, hy, hz), p(-hx, hy, -hz), -1, 0, 0)

  const data = new Float32Array(verts)
  return { data, vertexCount: data.length / VERTEX_STRIDE }
}

// ---------------------------------------------------------------------------
// Sphere
// ---------------------------------------------------------------------------

function spherePos(r, theta, phi, ox, oy, oz) {
  return [
    r * Math.sin(theta) * Math.cos(phi) + ox,
    r * Math.cos(theta) + oy,
    r * Math.sin(theta) * Math.sin(phi) + oz,
  ]
}

function sphereNorm(theta, phi) {
  return [
    Math.sin(theta) * Math.cos(phi),
    Math.cos(theta),
    Math.sin(theta) * Math.sin(phi),
  ]
}

/**
 * UV sphere with smooth-shaded vertex normals.
 *
 * @param {number} radius
 * @param {number} rings - latitude subdivisions
 * @param {number} segments - longitude subdivisions
 * @param {number} ox - X offset
 * @param {number} oy - Y offset
 * @param {number} oz - Z offset
 * @returns {{ data: Float32Array, vertexCount: number }}
 */
export function makeSphereGeometry(radius = 0.5, rings = 12, segments = 16, ox = 0, oy = 0, oz = 0) {
  const verts = []
  for (let r = 0; r < rings; r++) {
    const t1 = (r / rings) * Math.PI
    const t2 = ((r + 1) / rings) * Math.PI
    for (let s = 0; s < segments; s++) {
      const p1a = (s / segments) * 2 * Math.PI
      const p2a = ((s + 1) / segments) * 2 * Math.PI

      const pos1 = spherePos(radius, t1, p1a, ox, oy, oz)
      const pos2 = spherePos(radius, t2, p1a, ox, oy, oz)
      const pos3 = spherePos(radius, t2, p2a, ox, oy, oz)
      const pos4 = spherePos(radius, t1, p2a, ox, oy, oz)

      const n1 = sphereNorm(t1, p1a)
      const n2 = sphereNorm(t2, p1a)
      const n3 = sphereNorm(t2, p2a)
      const n4 = sphereNorm(t1, p2a)

      pushTriSmooth(verts, pos1, n1, pos2, n2, pos3, n3)
      pushTriSmooth(verts, pos1, n1, pos3, n3, pos4, n4)
    }
  }
  const data = new Float32Array(verts)
  return { data, vertexCount: data.length / VERTEX_STRIDE }
}

// ---------------------------------------------------------------------------
// Panel (double-sided quad)
// ---------------------------------------------------------------------------

/**
 * Double-sided panel quad with face normals.
 *
 * @param {number} w - width
 * @param {number} h - height
 * @returns {{ data: Float32Array, vertexCount: number }}
 */
export function makePanelGeometry(w = 1.2, h = 0.8) {
  const hw = w / 2, hh = h / 2
  const verts = []
  // Front (+Z)
  pushTriFlat(verts, [-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], 0, 0, 1)
  pushTriFlat(verts, [-hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0], 0, 0, 1)
  // Back (-Z)
  pushTriFlat(verts, [hw, -hh, 0], [-hw, -hh, 0], [-hw, hh, 0], 0, 0, -1)
  pushTriFlat(verts, [hw, -hh, 0], [-hw, hh, 0], [hw, hh, 0], 0, 0, -1)
  const data = new Float32Array(verts)
  return { data, vertexCount: data.length / VERTEX_STRIDE }
}

// ---------------------------------------------------------------------------
// Ground plane
// ---------------------------------------------------------------------------

/**
 * Large ground plane quad (y=0, in XZ plane, normal pointing up).
 *
 * @param {number} size - half-extent in X and Z
 * @returns {{ data: Float32Array, vertexCount: number }}
 */
export function makeGroundPlaneGeometry(size = 25) {
  const verts = []
  pushTriFlat(verts, [-size, 0, -size], [size, 0, -size], [size, 0, size], 0, 1, 0)
  pushTriFlat(verts, [-size, 0, -size], [size, 0, size], [-size, 0, size], 0, 1, 0)
  const data = new Float32Array(verts)
  return { data, vertexCount: data.length / VERTEX_STRIDE }
}

// ---------------------------------------------------------------------------
// Merge utility
// ---------------------------------------------------------------------------

/**
 * Merge multiple geometries into a single interleaved buffer.
 *
 * @param {{ data: Float32Array, vertexCount: number }[]} geos
 * @returns {{ data: Float32Array, vertexCount: number }}
 */
export function mergeGeometries(geos) {
  let total = 0
  for (const g of geos) total += g.data.length
  const merged = new Float32Array(total)
  let off = 0
  for (const g of geos) {
    merged.set(g.data, off)
    off += g.data.length
  }
  return { data: merged, vertexCount: total / VERTEX_STRIDE }
}

// ---------------------------------------------------------------------------
// Entity-specific composite geometry
// ---------------------------------------------------------------------------

/**
 * Fountain — 3-tier cascading structure (base, middle, top).
 * Recognizable wedding-cake silhouette. ~108 triangles.
 */
export function makeFountainGeometry() {
  return mergeGeometries([
    makeBoxGeometry(0.8, 0.3, 0.8, 0, -0.3, 0),  // Base tier
    makeBoxGeometry(0.5, 0.3, 0.5, 0, 0.0, 0),    // Middle tier
    makeBoxGeometry(0.3, 0.2, 0.3, 0, 0.25, 0),   // Top tier
  ])
}

/**
 * Market stand — counter box + 4 support poles + roof canopy.
 * Table-with-awning silhouette. ~216 triangles.
 */
export function makeMarketStandGeometry() {
  return mergeGeometries([
    makeBoxGeometry(1.0, 0.4, 0.5, 0, 0, 0),              // Counter
    makeBoxGeometry(1.2, 0.06, 0.6, 0, 0.55, 0),           // Roof
    makeBoxGeometry(0.06, 0.5, 0.06, -0.45, 0.25, -0.2),   // Pole FL
    makeBoxGeometry(0.06, 0.5, 0.06, 0.45, 0.25, -0.2),    // Pole FR
    makeBoxGeometry(0.06, 0.5, 0.06, -0.45, 0.25, 0.2),    // Pole BL
    makeBoxGeometry(0.06, 0.5, 0.06, 0.45, 0.25, 0.2),     // Pole BR
  ])
}

/**
 * NPC humanoid — sphere head + box body.
 * Snowman-like silhouette with clearly separated head. ~230 triangles.
 */
export function makeNPCGeometry() {
  return mergeGeometries([
    makeSphereGeometry(0.18, 8, 10, 0, 0.38, 0),   // Head
    makeBoxGeometry(0.28, 0.48, 0.18, 0, -0.02, 0), // Body
  ])
}

/**
 * Knowledge panel — content quad with thin frame border.
 * Floating holographic display silhouette.
 */
export function makeKnowledgePanelGeometry() {
  const w = 1.4, h = 1.0, t = 0.04
  return mergeGeometries([
    makePanelGeometry(w - t * 2, h - t * 2),            // Content area
    makeBoxGeometry(w, t, t, 0, h / 2 - t / 2, 0),     // Top border
    makeBoxGeometry(w, t, t, 0, -h / 2 + t / 2, 0),    // Bottom border
    makeBoxGeometry(t, h, t, -w / 2 + t / 2, 0, 0),    // Left border
    makeBoxGeometry(t, h, t, w / 2 - t / 2, 0, 0),     // Right border
  ])
}

/**
 * Audio marker — small sphere indicating a sound source.
 * Subtle, non-distracting presence.
 */
export function makeAudioMarkerGeometry() {
  return makeSphereGeometry(0.12, 6, 8)
}
