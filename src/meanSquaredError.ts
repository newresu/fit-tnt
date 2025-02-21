import { AnyMatrix } from './types';
/**
 * Performs E = Ax - b and E'E/A.rows
 *
 * @export
 * @param {Matrix} A input data
 * @param {Matrix} x current coefficients
 * @param {Matrix} b output data
 * @returns the mean squared error
 */
export function meanSquaredError(A: AnyMatrix, x: AnyMatrix, b: AnyMatrix) {
  const e = A.mmul(x).sub(b);
  return e.dot(e) / e.rows;
}
