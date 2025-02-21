/**
 * Code uses single (or few) symbols for very-known matrices
 * i.e A x = b, uses those symbols.
 */

import { Matrix, MatrixTransposeView } from "ml-matrix";

import {
  upperTriangularSubstitution,
  lowerTriangularSubstitution,
} from "./triangularSubstitution";
import { choleskyPrecondition } from "./choPrecondition";
import { initSafetyChecks } from "./initSafetyChecks";

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
 * Find the coefficients `x` for `Ax=b`; `A` is the data, `b` the known output.
 *
 * Only one right hand side supported (i.e `b` can not be a matrix, but must be a column vector passed as array.)
 *
 * tnt [based off the paper](https://ieeexplore.ieee.org/abstract/document/8425520).
 *
 * @param data the input or data matrix (2D Array)
 * @param result the output vector (1D Array)
 * @returns @see {@link TNTResults}
 */
export function tnt(
  data: Array2D | Matrix,
  result: Array1D | Matrix,
  opts: Partial<TNTOpts> = {}
): TNTResults {
  const A = Matrix.isMatrix(data) ? data : new Matrix(data);
  const b = Matrix.isMatrix(result) ? result : Matrix.columnVector(result);
  const x = Matrix.zeros(A.columns, 1); // column of coefficients.

  const At = A.transpose(); // copy is ok. it's used a few times.
  const AtA = At.mmul(A); //square m. will be mutated.

  initSafetyChecks(AtA, A, b); //throws custom errors on issues.

  // svd-pseudo_inverse for small matrices?
  // mutates AtA until RtR is positive definite.
  const LLt = choleskyPrecondition(AtA);
  const L = LLt.lowerTriangularMatrix;
  const Lt = new MatrixTransposeView(L);
  const AtA_inv = upperTriangularSubstitution(
    Lt,
    lowerTriangularSubstitution(L, Matrix.eye(AtA.rows))
  );

  const residual = Matrix.sub(b, A.mmul(x)); // r = b - Ax_0
  let gradient = At.mmul(residual); // r_hat = At * r
  // `z_0 = AtA_inv * r_hat = x_0 - A_inv * b`
  let x_error = AtA_inv.mmul(gradient);
  let p = x_error.clone(); // z_0 clone

  // `x_error`,  `residual` updated as it iterates
  const { maxIterations = A.columns * 3, tolerance = 10e-26 } = opts;
  const mse: number[] = [];

  let sqe, alpha, beta_denom, beta;
  let it = 0;
  while (it < maxIterations && worthContinuing(mse, tolerance)) {
    sqe = residual.dot(residual); //.to1DArray, multiply and add.
    alpha = x_error.dot(gradient) / sqe;
    x.add(Matrix.mul(p, alpha));
    residual.sub(residual.clone().mul(alpha)); // update residual
    beta_denom = x_error.dot(gradient); // using old values
    gradient = At.mmul(residual); //new g
    x_error = AtA_inv.mmul(gradient); // new x_error
    beta = x_error.dot(gradient) / beta_denom; // new/old ratio
    p = p.multiply(beta).add(x_error); // with new x_error
    mse.push(meanSquaredError(A, x, b));
    it++
  }

  return { solution: x.to1DArray(), iterations: it, mse: mse };
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
