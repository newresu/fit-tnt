import { AnyMatrix } from './types';

/**
 * Check that dimensionality of matrices matches.
 * @param A input data matrix
 * @param X solution matrix
 * @param B results matrix
 */
export function checkMatchingDimensions(A: AnyMatrix, X: AnyMatrix, B: AnyMatrix) {
  if (A.rows !== B.rows) {
    throw new RangeError(
      `Rows of A and y must match. Found dim(A)=(${A.rows}, ${A.columns}) and dim(y)=(${B.rows}, ${B.columns})`,
    );
  }
  if (A.columns !== X.rows) {
    throw new RangeError(
      `Columns of A and rows of X must match. Found dim(A)=(${A.rows}, ${A.columns}) and dim(X)=(${X.rows}, ${X.columns})`,
    );
  }
  if (X.columns !== B.columns) {
    throw new RangeError(
      `Columns of X and y must match. Found dim(X)=(${X.rows}, ${X.columns}) and dim(y)=(${B.rows}, ${B.columns})`,
    );
  }
}
