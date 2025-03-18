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
  let criteria = getCriteria(diag);

  let it = 15; // increase epsilon
  let npdIt = 5; //non-positive-definite iterations

  while (criteria.ratio < 1e-4 || !choleskyDC.isPositiveDefinite()) {
    if (!choleskyDC.isPositiveDefinite()) {
      npdIt--;
    }
    if (!Number.isFinite(criteria.eps) || !it || !npdIt) {
      //includes isNaN
      throw new PreconditionError();
    }
    for (let i = 0; i < AtA.rows; i++) {
      AtA.set(i, i, AtA.get(i, i) + criteria.eps);
    }
    choleskyDC = new CholeskyDecomposition(AtA); //again
    diag = choleskyDC.lowerTriangularMatrix.diagonal();
    criteria = getCriteria(diag, 15 - (it - 1));
    it--;
  }

  return choleskyDC;
}

interface Criteria {
  /**
   * epsilon
   */
  eps: number;
  /**
   * min / avg
   */
  ratio: number;
}
/**
 * Calculate epsilon and ratio (min/avg)
 * values all positive | 0 -> don't take `abs(item)`
 * @param arr array of numbers
 * @returns {@link Criteria}
 */
function getCriteria(arr: number[], power = 0): Criteria {
  let min = Infinity;
  let avg = 0;
  for (const item of arr) {
    if (Number.isFinite(item)) {
      avg += item;
      if (item < min) {
        min = item;
      }
    }
  }
  min += Number.EPSILON * 1000;
  avg = avg / arr.length + Number.EPSILON * 1000;
  return {
    eps: min * 10 ** power,
    ratio: min / avg,
  };
}
