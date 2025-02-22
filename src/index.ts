import { _tnt } from './tnt';
import { TNTOpts, Array1D, Array2D, TNTResults } from './types';
import { meanSquaredError } from './meanSquaredError';
/**
 * Code uses single (or few) symbols for very-known matrices
 * i.e A x = b, uses those symbols.
 */
import { Matrix, pseudoInverse } from 'ml-matrix';

/**
 * Find the coefficients `x` for `A x = b`; `A` is the data, `b` the known output.
 *
 * Only one right-hand-side supported (i.e `b` can not be a matrix, but must be a column vector passed as array.)
 *
 * tnt is [based off the paper](https://ieeexplore.ieee.org/abstract/document/8425520).
 * @param data - the input or data matrix (2D Array)
 * @param output - the known-output vector (1D Array)
 * @param opts
 * @returns @see {@link TNTResults}
 */
export function tnt(
  data: Array2D | Matrix,
  output: Array1D | Matrix,
  opts: Partial<TNTOpts> = {},
): TNTResults {
  const A = Matrix.isMatrix(data) ? data : new Matrix(data);
  const b = Matrix.isMatrix(output) ? output : Matrix.columnVector(output);

  const { pseudoinverseFallback = false } = opts;

  const mse: number[] = [meanSquaredError(A, x, b)];
  try {
    return {
      solution: _tnt(A, b, mse, opts).to1DArray(),
      mse,
      iterations: mse.length,
    };
  } catch (e) {
    if (pseudoinverseFallback) {
      const svd_sol = pseudoInverse(A).mmul(b);
      mse.push(meanSquaredError(A, svd_sol, b));
      return { solution: svd_sol.to1DArray(), iterations: mse.length, mse };
    } else {
      throw new Error((e as Error).message);
    }
  }
}

export { TNTOpts, TNTResults };
