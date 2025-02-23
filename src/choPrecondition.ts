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
  let epsilon = Number.EPSILON * AtA.max() * AtA.rows; // order of magnitude max column

  let choleskyDC = new CholeskyDecomposition(AtA);
  let it = 0;
  while (!choleskyDC.isPositiveDefinite()) {
    if (Number.isNaN(epsilon) || !Number.isFinite(epsilon) || it == 4) {
      throw new Error(
        'Preconditioning AtA failed. This may be due to ill-conditioning. Please, raise an issue with the matrix that errors.',
      );
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
