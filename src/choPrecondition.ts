import type { Matrix } from 'ml-matrix';
import { CholeskyDecomposition } from 'ml-matrix';

import { PreconditionError } from './Errors';

/**
 * Do `A^T A += d*I` until AtA is positive definite and `L` is "nice".
 * **Mutates** AtA
 *
 * @param AtA - Symmetric matrix from the normal equation.
 * @returns Cholesky Decomposition of AtA
 */
export function choleskyPrecondition(AtA: Matrix) {
  let choleskyDC = new CholeskyDecomposition(AtA);

  let diag = choleskyDC.lowerTriangularMatrix.diagonal();

  let it = 5; // increase epsilon
  let criteria = getCriteria(diag, -it);

  while (criteria.ratio < 1e-4 || !choleskyDC.isPositiveDefinite()) {
    if (!Number.isFinite(criteria.eps) || !it) {
      //includes isNaN
      throw new PreconditionError();
    }
    for (let i = 0; i < AtA.rows; i++) {
      AtA.set(i, i, AtA.get(i, i) + criteria.eps);
    }
    choleskyDC = new CholeskyDecomposition(AtA); //again
    diag = choleskyDC.lowerTriangularMatrix.diagonal();
    criteria = getCriteria(diag, 1 - it);
    it--;
  }

  return choleskyDC;
}

interface Criteria {
  /**
   * epsilon added to the diagonal of L
   */
  eps: number;
  /**
   * min / avg used as stop condition.
   */
  ratio: number;
}
/**
 * Calculate epsilon and ratio (min/avg)
 * values all positive | 0 -> don't take `abs(item)`
 * @param arr array of numbers
 * @param power multiplies the min value by 10**power
 * @returns {@link Criteria}
 */
function getCriteria(arr: number[], power: number): Criteria {
  // below this it's inaccurate.
  const delta = Number.EPSILON * 1000;
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
  min += delta;
  avg = avg / arr.length;
  return {
    eps: avg * Math.pow(10, power),
    ratio: min / avg,
  };
}
