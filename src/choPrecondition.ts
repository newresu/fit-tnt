import type { Matrix } from 'ml-matrix';
import { CholeskyDecomposition } from 'ml-matrix';
import { PreconditionError } from './Errors';

export function choleskyPrecondition(AtA: Matrix) {
  /**
   * Add epsilon to the diagonal until AtA is positive definite.
   * **Note** that we are changing AtA
   * For info on [MDN:EPSILON](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/EPSILON)
   * @param AtA - Symmetric matrix from the normal equation.
   * @returns Cholesky Decomposition of AtA
   */
  const max_avg = Math.abs(Math.max(...AtA.mean('column')));
  let epsilon = Number.EPSILON * 100; // order of magnitude max column
  if (max_avg > 1) {
    epsilon *= max_avg;
  }

  let choleskyDC = new CholeskyDecomposition(AtA);
  let it = 0;
  while (!choleskyDC.isPositiveDefinite()) {
    if (!Number.isFinite(epsilon) || it == 4) {
      //includes isNaN
      throw new PreconditionError();
    }
    for (let i = 0; i < AtA.rows; i++) {
      AtA.set(i, i, AtA.get(i, i) + epsilon);
    }
    epsilon *= 10;
    choleskyDC = new CholeskyDecomposition(AtA); //again
    it++;
  }
  return choleskyDC;
}
