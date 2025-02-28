import type { Matrix } from 'ml-matrix';
import { CholeskyDecomposition } from 'ml-matrix';

import { PreconditionError } from './Errors';

/**
 * Do `A^T A + d*I` until AtA is positive definite and `L` is "nice".
 * **Mutates** AtA
 * For info on [MDN:EPSILON](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/EPSILON)
 * @param AtA - Symmetric matrix from the normal equation.
 * @returns Cholesky Decomposition of AtA
 */
export function choleskyPreconditionTrick(AtA: Matrix) {
  let choleskyDC = new CholeskyDecomposition(AtA);

  let diag = choleskyDC.lowerTriangularMatrix.diagonal();
  let [min, avg] = arrayMeanAndAverage(diag);

  let ratio = (min + Number.EPSILON) / (avg + Number.EPSILON);
  let epsilon = min + Number.EPSILON;
  let it = 10;
  while (ratio < 1e-5 || !choleskyDC.isPositiveDefinite()) {
    if (!Number.isFinite(epsilon) || !it) {
      //includes isNaN
      throw new PreconditionError();
    }
    for (let i = 0; i < AtA.rows; i++) {
      AtA.set(i, i, AtA.get(i, i) + epsilon);
    }
    epsilon *= 10;
    choleskyDC = new CholeskyDecomposition(AtA); //again
    diag = choleskyDC.lowerTriangularMatrix.diagonal();
    [min, avg] = arrayMeanAndAverage(diag);
    ratio = (min + Number.EPSILON) / (avg + Number.EPSILON);

    it--;
  }
  return choleskyDC;
}

/**
 * Calculate mean and average of an array.
 * @param arr array of numbers
 * @returns
 */
function arrayMeanAndAverage(arr: number[]) {
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
