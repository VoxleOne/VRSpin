import { Geometry } from "./Geometry.js"

export class BoxGeometry extends Geometry {
  constructor() {
    super(new Float32Array([
      -0.5, -0.5, 0,
       0.5, -0.5, 0,
       0.0,  0.5, 0
    ]))
  }
}
