import type { Matrix } from 'ml-matrix';
import { CholeskyDecomposition } from 'ml-matrix';

export function choleskyPrecondition(AtA: Matrix) {
  /**
   * Add epsilon to the diagonal until AtA is positive definite.
   * **Note** that we are changing AtA
   * For info on [MDN:EPSILON](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/EPSILON)
   * @param AtA - Symmetric matrix from the normal equation.
   * @returns Cholesky Decomposition of AtA
   */
  let epsilon = Number.EPSILON * AtA.max() * AtA.columns;
  let RtR = new CholeskyDecomposition(AtA);
  while (!RtR.isPositiveDefinite()) {
    epsilon *= 10;
    for (let i = 0; i < AtA.columns; i++) {
      AtA.set(i, i, AtA.get(i, i) + epsilon);
    }
    RtR = new CholeskyDecomposition(AtA); //again
  }
  return RtR;
}
