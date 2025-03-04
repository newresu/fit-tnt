import { Matrix } from 'ml-matrix';

import { choleskyPreconditionTrick } from './choPrecondition';
import { initSafetyChecks } from './initSafetyChecks';
import { invertLLt } from './invertLLt';
import { meanSquaredError } from './meanSquaredError';
import { symmetricMul } from './symmetricMul';
import { AnyMatrix, Array1D, Array2D, EarlyStopping, TNTOpts } from './types';

/**
 * Find the coefficients `x` for `A x = b`; `A` is the data, `b` the known output.
 *
 * Only one right-hand-side supported (i.e `b` can not be a matrix, but must be a column vector passed as array.)
 *
 * tnt is [based off the paper](https://ieeexplore.ieee.org/abstract/document/8425520).
 * @param data the input or data matrix (2D Array)
 * @param output the known-output vector (1D Array)
 * @param opts {@link TNTOpts}
 */
export class TNT {
  xBest: AnyMatrix;
  /**
   * {@link TNTOpts["maxIterations"]}
   */
  maxIterations: number;
  /**
   * {@link TNTOpts["earlyStopping"]}
   */
  earlyStopping: EarlyStopping;

  /**
   * Mean Squared Error for each iteration plus the initial guess (`mse[0]`)
   */
  mse: number[];
  /**
   * Minimum Mean Squared Error in all the iterations.
   */
  mseMin: number;
  /**
   * Last MSE of all iterations.
   */
  mseLast: number;

  constructor(
    data: Array2D | AnyMatrix,
    output: Array2D | Array1D | AnyMatrix,
    opts: Partial<TNTOpts> = {},
  ) {
    const A = Matrix.checkMatrix(data);
    const b = Matrix.isMatrix(output)
      ? output
      : Array.isArray(output[0])
        ? new Matrix(output as number[][])
        : Matrix.columnVector(output as number[]);
    this.xBest = new Matrix(A.columns, 1);

    // unpack options
    const {
      maxIterations = 3 * A.columns,
      earlyStopping: { minMSE = 1e-20 } = {},
    } = opts;

    this.maxIterations = maxIterations;
    this.earlyStopping = { minMSE };

    this.mse = [b.dot(b) / b.columns];
    this.mseLast = this.mseMin = this.mse[0];

    if (this.mseLast === 0) return;

    this.#tnt(A, b);
  }

  get iterations() {
    return this.mse.length - 1;
  }

  /**
   * 1. Calculate `mseLast`
   * 2. Updates `mse[]`
   * 3. Sets `mseMin` if improved, and `xBest` in that case.
   * @param A input data matrix
   * @param b known output vector
   * @param x current coefficients
   * @return void
   */
  #updateMSEAndX(A: AnyMatrix, b: AnyMatrix, x: AnyMatrix, cloneX = true) {
    this.mseLast = meanSquaredError(A, x, b);
    this.mse.push(this.mseLast);
    if (this.mseLast < this.mseMin) {
      this.mseMin = this.mseLast;
      this.xBest = cloneX ? x.clone() : x;
    }
  }

  /**
   * Private function (main method)
   * @param A
   * @param b
   * @param mse this will be mutated; this allows user to get all MSEs.
   * @param options
   * @returns best-found coefficients
   */
  #tnt(A: AnyMatrix, b: AnyMatrix) {
    const x = Matrix.zeros(A.columns, 1); // column of coefficients.
    const At = A.transpose(); // copy is ok. it's used a few times.
    const AtA = symmetricMul(At);
    initSafetyChecks(A, b); //throws custom errors on issues.
    const choleskyDC = choleskyPreconditionTrick(AtA);
    const L = choleskyDC.lowerTriangularMatrix;
    const AtA_inv = invertLLt(L);

    const residual = b.clone(); // r = b - Ax_0 (but Ax_0 is 0)
    let gradient = At.mmul(residual); // r_hat = At * r
    // `z_0 = AtA_inv * r_hat = x_0 - A_inv * b`
    let xError = AtA_inv.mmul(gradient);
    const p = xError.clone(); // z_0 clone

    let w, alpha, betaDenom, beta;
    for (let it = 0; it < this.maxIterations; it++) {
      w = A.mmul(p);
      alpha = xError.dot(gradient) / w.dot(w);
      x.add(Matrix.mul(p, alpha)); //update x

      if (
        !Number.isFinite(alpha) ||
        x.to1DArray().some((v) => !Number.isFinite(v))
      ) {
        break;
      }

      this.#updateMSEAndX(A, b, x); //updates: mse and counter and xBest

      if (this.mseLast !== this.mseMin) {
        break;
      }

      betaDenom = xError.dot(gradient); // old CG (maybe)
      residual.sub(residual.clone().mul(alpha)); // update residual

      gradient = At.mmul(residual); // new g
      xError = AtA_inv.mmul(gradient); // new x_error
      beta = xError.dot(gradient) / betaDenom; // new_CG/old_CG
      p.multiply(beta).add(xError); // update p
    }
  }
}
