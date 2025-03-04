import { AnyMatrix } from './types';

/**
 * Performs Frobenius Norm or Standard Norm.
 *
 * @export
 * @param A input data
 * @param X current coefficients
 * @param B output data
 * @returns the mean squared error
 */
export function meanSquaredError(A: AnyMatrix, X: AnyMatrix, B: AnyMatrix) {
  const e = A.mmul(X).sub(B);
  return e.dot(e) / (e.rows * B.columns);
}
