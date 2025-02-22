import { pseudoInverse, MatrixTransposeView, Matrix } from 'ml-matrix';

import { initSafetyChecks } from './initSafetyChecks';
import { invertLLt } from './triangularSubstitution';
import { choleskyPrecondition } from './choPrecondition';

import { TNTOpts, Array1D, Array2D, TNTResults } from './types';
import { meanSquaredError } from './meanSquaredError';

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

  const mse: number[] = [];
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

function worthContinuing(mse: number[], tolerance: number) {
  /**
   * @returns Whether we should continue or not.
   */
  const last = mse[mse.length - 1];
  if (mse.length > 0 && last < tolerance) {
    // there is already a good result
    return false;
  } else if (mse.length > 1 && last >= mse[mse.length - 2]) {
    // worse than previous (otherwise use smaller tolerance)
    return false;
  }
  return true;
}

/**
 * Private function (main method)
 * @param A
 * @param b
 * @param mse this will be mutated; this allows user to get all MSEs. 
 * @param options
 * @returns best-found coefficients
 */
export function _tnt(
  A: Matrix,
  b: Matrix,
  mse: number[],
  options: Partial<TNTOpts> = {},
) {
  const { maxIterations = A.columns * 3, tolerance = 10e-26 } = options;
  const x = Matrix.zeros(A.columns, 1); // column of coefficients.

  mse.push(meanSquaredError(A, x, b));
  const At = new MatrixTransposeView(A); // copy is ok. it's used a few times.
  const AtA = At.mmul(A); //square m. will be mutated.
  initSafetyChecks(AtA, A, b); //throws custom errors on issues.
  const choleskyDC = choleskyPrecondition(AtA);
  const L = choleskyDC.lowerTriangularMatrix;
  const AtA_inv = invertLLt(L);

  const residual = Matrix.sub(b, A.mmul(x)); // r = b - Ax_0
  let gradient = At.mmul(residual); // r_hat = At * r
  // `z_0 = AtA_inv * r_hat = x_0 - A_inv * b`
  let x_error = AtA_inv.mmul(gradient);
  const p = x_error.clone(); // z_0 clone

  let lowest_mse = Infinity;
  let it = 0;
  let sqe, alpha, beta_denom, beta, x_best; // column of coefficients.
  while (it < maxIterations && worthContinuing(mse, tolerance)) {
    sqe = residual.dot(residual); //.to1DArray, multiply and add.
    alpha = x_error.dot(gradient) / sqe;
    x.add(Matrix.mul(p, alpha));
    if (isNaN(x.get(0, 0)) || !Number.isFinite(x.get(0, 0))) {
      throw new Error(
        'Infinite or NaN coefficients were found. This may be due to a very ill-conditioned matrix. You can try `{use_SVD:true}` and it will fallback to solve using pseudoinverse.',
      );
    }
    residual.sub(residual.clone().mul(alpha)); // update residual
    beta_denom = x_error.dot(gradient); // using old values
    gradient = At.mmul(residual); //new g
    x_error = AtA_inv.mmul(gradient); // new x_error
    beta = x_error.dot(gradient) / beta_denom; // new/old ratio
    p.multiply(beta).add(x_error); // with new x_error
    mse.push(meanSquaredError(A, x, b));
    if (mse[mse.length - 1] < lowest_mse) {
      lowest_mse = mse[mse.length - 1];
      x_best = x.clone();
    }
    it++;
  }
  if (!x_best) {
    throw new Error('Internal Error calculating the solution / coefficients.');
  }
  return x_best;
}
