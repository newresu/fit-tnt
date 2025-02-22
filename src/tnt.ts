/**
 * Code uses single (or few) symbols for very-known matrices
 * i.e A x = b, uses those symbols.
 */

import { Matrix, pseudoInverse, solve } from 'ml-matrix';

import { choleskyPrecondition } from './choPrecondition';
import { initSafetyChecks } from './initSafetyChecks';
import { invertLLt } from './triangularSubstitution';

type Array2D = ArrayLike<ArrayLike<number>>;
type Array1D = ArrayLike<number>;

export interface TNTOpts {
  /**
   * @default `A.columns * 3`
   */
  maxIterations: number;
  /**
   * When current_error < tolerance it stops.
   * @default 10E-20
   */
  tolerance: number;
  /**
   * If the software errors (normally very ill-conditioned matrix) it fallbacks to the slower pseudo-inverse.
   * @default false
   */
  pseudoInverse_fallback: boolean;
}
export interface TNTResults {
  iterations: number;
  /**
   * Mean Squared Error i.e `(Ax-b)*(Ax-b)'/n`
   * At each iteration.
   */
  mse: number[];
  /**
   * The coefficients
   */
  solution: Array1D;
}
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
  const x = Matrix.zeros(A.columns, 1); // column of coefficients.
  let x_best = x.clone(); // column of coefficients.

  const At = A.transpose(); // copy is ok. it's used a few times.
  const AtA = At.mmul(A); //square m. will be mutated.

  initSafetyChecks(AtA, A, b); //throws custom errors on issues.

  // svd-pseudo_inverse for small matrices?
  // mutates AtA until RtR is positive definite.
  const choleskyDC = choleskyPrecondition(AtA);
  const L = choleskyDC.lowerTriangularMatrix;
  const AtA_inv = invertLLt(L);
  const residual = Matrix.sub(b, A.mmul(x)); // r = b - Ax_0
  let gradient = At.mmul(residual); // r_hat = At * r
  // `z_0 = AtA_inv * r_hat = x_0 - A_inv * b`
  let x_error = AtA_inv.mmul(gradient);
  let p = x_error.clone(); // z_0 clone

  // `x_error`,  `residual` updated as it iterates
  const { maxIterations = A.columns * 3, tolerance = 10e-26 } = opts;
  let last_mse = meanSquaredError(A, x, b);
  let lowest_mse = Infinity;
  const mse: number[] = [last_mse];

  let sqe, alpha, beta_denom, beta;
  let it = 0;
  try {
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
      p = p.multiply(beta).add(x_error); // with new x_error
      last_mse = meanSquaredError(A, x, b);
      if (last_mse < lowest_mse) {
        lowest_mse = last_mse;
        x_best = x.clone();
      }
      mse.push(last_mse);
      it++;
    }
  } catch (e) {
    if (opts.pseudoInverse_fallback) {
      const svd_sol = pseudoInverse(A).mmul(b);
      mse.push(meanSquaredError(A, svd_sol, b));
      return { solution: svd_sol.to1DArray(), iterations: it, mse };
    } else {
      throw new Error(e);
    }
  }

  return { solution: x_best.to1DArray(), iterations: it, mse };
}

function meanSquaredError(A: Matrix, x: Matrix, b: Matrix) {
  return A.mmul(x).sub(b).pow(2).mean();
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
