import { AnyMatrix } from './types';

interface SquaredSumOpts {
  /**
   * When "column" is used, the elements added are those in the same column.
   * For `[[1,2],[3,4]]` it adds 1+3 (in this case 1^2 + 3^2 = 10)
   * Then does the same for 2 and 4.
   *
   * When "row" is used, the elements added are those in the same row.
   * For `[[1,2], [3,4]]` it adds 1+2 (1^2 + 2^2) then does the same for
   * 3 and 4.
   * @default "column"
   */
  by: 'column' | 'row';
}
/**
 * 1. Square the matrix
 * 2. Add the column elements.
 * @param A input matrix.
 * @param opts options for performing the sums.
 * @returns array with mean values per column
 */
export function squaredSum(A: AnyMatrix, opts: Partial<SquaredSumOpts> = {}) {
  const { by = 'column' } = opts;
  const result: number[] = new Array(by === 'column' ? A.columns : A.rows).fill(
    0,
  ) as number[];
  if (A.isColumnVector() || A.isRowVector()) {
    return [A.dot(A)];
  } else {
    // square of each number in the matrix, and add it up.
    if (by === 'column') {
      for (let j = 0; j < A.columns; j++) {
        let terms = 0;
        for (let i = 0; i < A.rows; i++) {
          terms += A.get(i, j) ** 2;
        }
        result[j] = terms;
      }
    } else {
      for (let i = 0; i < A.rows; i++) {
        let terms = 0;
        for (let j = 0; j < A.columns; j++) {
          terms += A.get(i, j) ** 2;
        }
        result[i] = terms;
      }
    }
  }
  return result;
}
