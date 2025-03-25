import type { Matrix } from 'ml-matrix';
import { CholeskyDecomposition } from 'ml-matrix';

import { PreconditionError } from './Errors';
import { getCriteria } from './getCholeskyCriteria';

/**
 * Do `A^T A += d*I` until AtA is positive definite and `L` is "nice".
 * **Mutates** AtA
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
