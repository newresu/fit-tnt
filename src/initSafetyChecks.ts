import type { Matrix } from 'ml-matrix';

import { AsymmetricError } from './Errors';

/**
 * 
 * @param A input data matrix
 * @param y ouput data vector
 */
export function initSafetyChecks(A: Matrix, y: Matrix) {
  if (A.rows !== y.rows) {
    throw new RangeError(
      `Rows of A and y must match. Found dim(A)=(${A.rows}, ${A.columns}) and dim(y)=(${y.rows}, ${y.columns})`,
    );
  }
  if (!y.isColumnVector()) {
    throw new Error(`One Right-Hand-Side is supported. Found ${y.columns}`);
  }
}
