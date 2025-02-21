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



type Array2D = ArrayLike<ArrayLike<number>>
type Array1D = ArrayLike<number>

/**
 * Find the coefficients (x) for `Ax=b`, where `x` and `b` are vectors.
 *
 * tnt [based off the paper](https://ieeexplore.ieee.org/abstract/document/8425520).
 *
 * @param data the input or data matrix (2D Array)
 * @param result the output vector (1D Array)
 * @returns the best-found coefficients (1D Array)
 */
export function tnt(data: Array2D| Matrix, result: Array1D|Matrix) {

  const A = Matrix.isMatrix(data)? data : new Matrix(data)
  const b = Matrix.isMatrix(result)?result: Matrix.columnVector(result)
  const x = Matrix.zeros(A.columns, 1); // column of coefficients.

  const At = A.transpose(); // copy is ok. it's used a few times.
  let AtA = At.mmul(A); //square m. will be mutated.

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

  
  let residual = Matrix.sub(b, A.mmul(x));// r = b - Ax_0
  let gradient = At.mmul(residual);// r_hat = At * r
  // `z_0 = AtA_inv * r_hat = x_0 - A_inv * b`
  let x_error = AtA_inv.mmul(gradient);
  let p = x_error.clone(); // z_0 clone

  // `x_error`,  `residual` updated as it iterates
  for (let it = 0; it < A.columns * 3; it++) {
    const sqe = residual.dot(residual); //.to1DArray, multiply and add.
    const alpha = x_error.dot(gradient) / sqe;
    x.add(Matrix.mul(p, alpha));
    residual.sub(residual.clone().mul(alpha)); // update residual
    const beta_denom = x_error.dot(gradient); // using old values
    gradient = At.mmul(residual); //new g
    x_error = AtA_inv.mmul(gradient); // new x_error
    const beta = x_error.dot(gradient) / beta_denom; // new/old ratio
    p = p.multiply(beta).add(x_error); // with new x_error
  }
  return x.to1DArray();
}

