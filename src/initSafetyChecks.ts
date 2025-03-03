import { AnyMatrix } from './types';

/**
 *
 * @param A input data matrix
 * @param X coefficient matrix
 * @param B known output vector
 */
export function initSafetyChecks(A: AnyMatrix, X: AnyMatrix, B: AnyMatrix) {
  if (A.rows !== B.rows) {
    throw new RangeError(
      `Found A rows = ${A.rows} and B rows = ${B.rows}. They must match.`,
    );
  }
  if (A.columns !== X.rows) {
    throw new RangeError(
      `Found A columns = ${A.columns} and X rows = ${X.rows}. They must match.`,
    );
  }
  if (B.columns !== X.columns) {
    throw new RangeError(
      `Found B columns = ${B.columns} and X columns = ${X.columns}. They must match.`,
    );
  }
}
