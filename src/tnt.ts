import { pseudoInverse, MatrixTransposeView, Matrix } from 'ml-matrix';

import { initSafetyChecks } from './initSafetyChecks';
import { invertLLt } from './triangularSubstitution';
import { choleskyPrecondition } from './choPrecondition';

import { TNTOpts, Array1D, Array2D, EarlyStopping } from './types';
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
export class TNT {
  A: Matrix;
  b: Matrix;

  pseudoInverseFallback: boolean;
  maxIterations: number;
  earlyStopping: EarlyStopping;

  mse: number[];
  iterations: number;

  constructor(
    data: Array2D | Matrix,
    output: Array1D | Matrix,
    opts: Partial<TNTOpts> = {},
  ) {
    this.A = Matrix.isMatrix(data) ? data : new Matrix(data);
    this.b = Matrix.isMatrix(output) ? output : Matrix.columnVector(output);

    // unpack options
    const {
      pseudoInverseFallback = false,
      maxIterations = 3 * this.A.columns,
      earlyStopping: { minError = 10e-20, patience = 3 } = {},
    } = opts;

    this.pseudoInverseFallback = pseudoInverseFallback;
    this.maxIterations = maxIterations;
    this.earlyStopping = { minError, patience };

    this.mse = [];
    this.iterations = 0;
  }

  solve() {
    try {
      const solution = this._tnt();
      this.iterations = this.mse.length;
      return solution;
    } catch (e) {
      if (this.pseudoInverseFallback) {
        return this._pseudoInverseFallback();
      } else {
        throw new Error((e as Error).message);
      }
    }
  }
  _pseudoInverseFallback() {
    const svd_sol = pseudoInverse(this.A).mmul(this.b);
    this.mse.push(meanSquaredError(this.A, svd_sol, this.b));
    this.iterations = this.mse.length;
    return svd_sol;
  }
  /**
   * Private function (main method)
   * @param A
   * @param b
   * @param mse this will be mutated; this allows user to get all MSEs.
   * @param options
   * @returns best-found coefficients
   */
  _tnt() {
    const x = Matrix.zeros(this.A.columns, 1); // column of coefficients.
    const At = new MatrixTransposeView(this.A); // copy is ok. it's used a few times.
    const AtA = At.mmul(this.A); //square m. will be mutated.
    initSafetyChecks(AtA, this.A, this.b); //throws custom errors on issues.
    const choleskyDC = choleskyPrecondition(AtA);
    const L = choleskyDC.lowerTriangularMatrix;
    const AtA_inv = invertLLt(L);

    const residual = this.b.clone(); // r = b - Ax_0 (but Ax_0 is 0)
    this.mse.push(this.b.dot(this.b) / this.b.columns);
    let gradient = At.mmul(residual); // r_hat = At * r
    // `z_0 = AtA_inv * r_hat = x_0 - A_inv * b`
    let x_error = AtA_inv.mmul(gradient);
    const p = x_error.clone(); // z_0 clone

    let lowest_mse = Infinity;
    let it = 0;
    let sqe, alpha, beta_denom, beta, x_best; // column of coefficients.
    while (it < this.maxIterations && this._worthContinuing()) {
      sqe = residual.dot(residual); //.to1DArray, multiply and add.
      alpha = x_error.dot(gradient) / sqe;
      x.add(Matrix.mul(p, alpha));
      if (!Number.isFinite(x.get(0, 0))) {
        //throws on NaN as well.
        throw new Error(
          'Infinite or NaN coefficients were found. This may be due to a very ill-conditioned matrix. Try `{pseudoInverseFallback:true}`.',
        );
      }
      residual.sub(residual.clone().mul(alpha)); // update residual
      beta_denom = x_error.dot(gradient); // using old values
      gradient = At.mmul(residual); //new g
      x_error = AtA_inv.mmul(gradient); // new x_error
      beta = x_error.dot(gradient) / beta_denom; // new/old ratio
      p.multiply(beta).add(x_error); // with new x_error
      this.mse.push(meanSquaredError(this.A, x, this.b));
      if (this.mse[this.mse.length - 1] < lowest_mse) {
        lowest_mse = this.mse[this.mse.length - 1];
        x_best = x.clone();
      }
      it++;
    }
    if (!x_best) {
      throw new Error(
        'Internal Error calculating the solution / coefficients.',
      );
    }
    return x_best;
  }

  _worthContinuing() {
    /**
     * @returns Whether we should continue or not.
     */
    const last = this.mse[this.mse.length - 1];
    let patience = this.earlyStopping.patience;
    if (this.mse.length > 0 && last < this.earlyStopping.minError) {
      // there is already a good result
      return false;
    } else if (this.mse.length > 1) {
      for (let i = this.mse.length - 2; i >= 0; i--) {
        if (last < this.mse[i] && patience > 0) {
          return true;
        }
        patience--;
      }
      // worse than previous (otherwise use smaller tolerance)
      return false;
    }
    return true;
  }
}
