import { Matrix } from 'ml-matrix';
/**
 * Performs E = Ax - b and E'E/A.rows
 *
 * @export
 * @param {Matrix} A input data
 * @param {Matrix} x current coefficients
 * @param {Matrix} b output data
 * @returns the mean squared error
 */
export function meanSquaredError(A: Matrix, x: Matrix, b: Matrix) {
  const e = A.mmul(x).sub(b);
  return e.dot(e);
}
