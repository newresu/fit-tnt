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

interface SquaredSumOpts {
  /**
   * When "column" is used, the elements added are those in the same column.
   * For `[[1,2],[3,4]]` it adds 1+3 (in this case 1^2 + 3^2 = 10)
   * Then does the same for 2 and 4.
   *
   * When "row" is used, the elements added are those in the same row.
   * For `[[1,2], [3,4]]` it adds 1+2 (1^2 + 2^2) then does the same for
   * 3 and 4.
   * @default "column"
   */
  by: 'column' | 'row';
}
/**
 * 1. Square the matrix
 * 2. Add the column elements.
 * @param A input matrix.
 * @param opts options for performing the sums.
 * @returns array with mean values per column
 */
export function squaredSum(A: AnyMatrix, opts: Partial<SquaredSumOpts> = {}) {
  const { by = 'column' } = opts;
  const result: number[] = new Array(by === 'column' ? A.columns : A.rows).fill(
    0,
  ) as number[];
  if (A.isColumnVector() || A.isRowVector()) {
    return [A.dot(A)];
  } else {
    // square of each number in the matrix, and add it up.
    if (by === 'column') {
      for (let j = 0; j < A.columns; j++) {
        let terms = 0;
        for (let i = 0; i < A.rows; i++) {
          terms += A.get(i, j) ** 2;
        }
        result[j] = terms;
      }
    } else {
      for (let i = 0; i < A.rows; i++) {
        let terms = 0;
        for (let j = 0; j < A.columns; j++) {
          terms += A.get(i, j) ** 2;
        }
        result[i] = terms;
      }
    }
  }
  return result;
}
