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
  const criteria = getCriteria(diag);

  const minValue = criteria.min;
  let ratio = criteria.ratio;
  let epsilon = minValue + Number.EPSILON * 1000;

  let it = 15; // increase epsilon
  let npdIt = 5; //non-positive-definite iterations
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
    choleskyDC = new CholeskyDecomposition(AtA); //again
    diag = choleskyDC.lowerTriangularMatrix.diagonal();
    ratio = getCriteria(diag).ratio;
    epsilon *= 10;

    it--;
  }
  return choleskyDC;
}

interface Criteria {
  /**
   * min value in array.
   */
  min: number;
  /**
   * min / avg
   */
  ratio: number;
}
/**
 * Calculate min, ratio (min/avg)
 * values all positive | 0 -> don't take `abs(item)`
 * @param arr array of numbers
 * @returns {@link Criteria}
 */
function getCriteria(arr: number[]): Criteria {
  let min = arr[0];
  let sum = 0;
  for (const item of arr) {
    sum += item;
    if (item < min) {
      min = item;
    }
  }
  return { min, ratio: min / (sum / arr.length) };
}
