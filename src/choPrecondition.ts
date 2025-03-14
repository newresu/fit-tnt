import type { Matrix } from 'ml-matrix';
import { CholeskyDecomposition } from 'ml-matrix';

import { PreconditionError } from './Errors';

/**
 * Do `A^T A + d*I` until AtA is positive definite and `L` is "nice".
 * **Mutates** AtA
 * For info on [MDN:EPSILON](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/EPSILON)
 *
 * @param AtA - Symmetric matrix from the normal equation.
 * @param maxIterations
 * @param ratio target for ratio of diagonal values in L.
 * @returns Cholesky Decomposition of AtA
 */
export function choleskyPreconditionTrick(AtA: Matrix) {
  let choleskyDC = new CholeskyDecomposition(AtA);

  let diag = choleskyDC.lowerTriangularMatrix.diagonal();
  let [min, avg] = arrayMinAndAverage(diag);

  let ratio = min / avg + Number.EPSILON;
  let epsilon = min + Number.EPSILON * 1000;
  let it = 15; // increase epsilon
  let npdIt = 5; //non positive definite iterations
  while (ratio < 1e-3 || !choleskyDC.isPositiveDefinite()) {
    if (!choleskyDC.isPositiveDefinite()) {
      npdIt -= 1;
    }
    if (!Number.isFinite(epsilon) || it === 0 || npdIt === 0) {
      //includes isNaN
      throw new PreconditionError();
    }
    for (let i = 0; i < AtA.rows; i++) {
      AtA.set(i, i, AtA.get(i, i) + epsilon);
    }
    epsilon *= 10;
    choleskyDC = new CholeskyDecomposition(AtA); //again
    diag = choleskyDC.lowerTriangularMatrix.diagonal();
    [min, avg] = arrayMinAndAverage(diag);
    ratio = min / avg;

    it--;
  }
  return choleskyDC;
}

/**
 * Calculate min and average of an array.
 * values all positive | 0 -> don't take `abs(item)`
 * @param arr array of numbers
 * @returns min and average values.
 */
function arrayMinAndAverage(arr: number[]): [number, number] {
  let min = arr[0];
  let avg = 0;
  for (const item of arr) {
    avg += item;
    if (item < min) {
      min = item;
    }
  }
  return [min, avg / arr.length];
}
