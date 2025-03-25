import { AnyMatrix } from './types';

/**
 * Calculate the mean squared error.
 *
 * It works for matrices or vectors.
 *
 * It operates like this:
 * If you pass A only, it squares all and the mean **by column**.
 * If you pass A and B, it does A-B and then same as above.
 * If you pass A, X and B it does `AX` and then same as above.
 * @param A input data
 * @param B output data
 * @param X current coefficients
 * @returns the mean squared error
 */
export function meanSquaredError(A: AnyMatrix, B: AnyMatrix, X: AnyMatrix) {
  const Err: AnyMatrix = A.mmul(X).sub(B);
  const { rows: samples, columns: coeffs } = Err;

  const result: number[] = new Array(X.columns).fill(0) as number[];
  if (Err.isColumnVector()) {
    return [Err.dot(Err) / samples];
  } else {
    // square of each number in the matrix, and add it up.
    for (let i = 0; i < samples; i++) {
      for (let j = 0; j < coeffs; j++) {
        result[j] += Err.get(i, j) ** 2;
      }
    }
    for (let v = 0; v < result.length; v++) {
      result[v] /= samples;
    }
  }
  return result;
}
