import { pseudoInverse, MatrixTransposeView, Matrix } from 'ml-matrix';

import { initSafetyChecks } from './initSafetyChecks';
import { invertLLt } from './triangularSubstitution';
import { choleskyPrecondition } from './choPrecondition';

import { TNTOpts, Array1D, Array2D, EarlyStopping } from './types';
import { meanSquaredError } from './meanSquaredError';
import { NaNOrNonFiniteError } from './Errors';

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
  xBest: Matrix;

  pseudoInverseFallback: boolean;
  maxIterations: number;
  earlyStopping: EarlyStopping;

  mse: number[];
  mseMin: number;
  mseLast: number;
  _noImprovementCounter: number;

  constructor(
    data: Array2D | Matrix,
    output: Array1D | Matrix,
    opts: Partial<TNTOpts> = {},
  ) {
    const A = Matrix.isMatrix(data) ? data : new Matrix(data);
    const b = Matrix.isMatrix(output) ? output : Matrix.columnVector(output);
    this.xBest = new Matrix(A.columns, 1);

    // unpack options
    const {
      pseudoInverseFallback = false,
      maxIterations = 3 * A.columns,
      earlyStopping: { minError = 10e-20, patience = 3 } = {},
    } = opts;

    this.pseudoInverseFallback = pseudoInverseFallback;
    this.maxIterations = maxIterations;
    this.earlyStopping = { minError, patience };

    this.mse = [];
    this.mseMin = Infinity;
    this.mseLast = Infinity;
    this._noImprovementCounter = 0;
    this._solve(A, b);
  }

  get iterations() {
    return this.mse.length;
  }

  _solve(A: Matrix, b: Matrix) {
    try {
      this._tnt(A, b);
    } catch (e) {
      if (this.pseudoInverseFallback) {
        const x = pseudoInverse(A).mmul(b);
        this._update(A, b, x, false); //false==no cloning of X
        return;
      }
      throw new Error((e as Error).message);
    }
  }

  /**
   * Updates mse[], mse values, counter and xBest
   * @param x current coefficients
   */
  _update(A: Matrix, b: Matrix, x: Matrix, cloneX: boolean = true) {
    this.mseLast = meanSquaredError(A, x, b);
    this.mse.push(this.mseLast);
    if (this.mseLast < this.mseMin) {
      this.mseMin = this.mseLast;
      this._noImprovementCounter = 0;
      this.xBest = cloneX ? x.clone() : x;
    } else {
      this._noImprovementCounter += 1;
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
  _tnt(A: Matrix, b: Matrix) {
    const x = Matrix.zeros(A.columns, 1); // column of coefficients.
    const At = new MatrixTransposeView(A); // copy is ok. it's used a few times.
    const AtA = At.mmul(A); //square m. will be mutated.
    initSafetyChecks(A, b); //throws custom errors on issues.
    const choleskyDC = choleskyPrecondition(AtA);
    const L = choleskyDC.lowerTriangularMatrix;
    const AtA_inv = invertLLt(L);

    const residual = b.clone(); // r = b - Ax_0 (but Ax_0 is 0)
    let gradient = At.mmul(residual); // r_hat = At * r
    // `z_0 = AtA_inv * r_hat = x_0 - A_inv * b`
    let xError = AtA_inv.mmul(gradient);
    const p = xError.clone(); // z_0 clone

    let sqe = b.dot(b);
    this.mse.push(sqe / b.columns);
    this.mseLast = this.mseMin = this.mse[0];

    let alpha, betaDenom, beta; // column of coefficients.
    for (let it = 0; it < this.maxIterations; it++) {
      alpha = xError.dot(gradient) / sqe;
      x.add(Matrix.mul(p, alpha)); //update x
      if (!Number.isFinite(x.get(0, 0)) || !Number.isFinite(alpha)) {
        throw new NaNOrNonFiniteError();
      }
      this._update(A, b, x); //updates: mse and counter and xBest
      if (this._noImprovementCounter > this.earlyStopping.patience) {
        break;
      }
      if (this.mseMin < this.earlyStopping.minError) {
        break;
      }
      betaDenom = xError.dot(gradient); // old CG (maybe)
      residual.sub(residual.clone().mul(alpha)); // update residual
      gradient = At.mmul(residual); // new g
      xError = AtA_inv.mmul(gradient); // new x_error
      beta = xError.dot(gradient) / betaDenom; // new_CG/old_CG
      p.multiply(beta).add(xError); // update p

      sqe = residual.dot(residual); //.to1DArray, multiply and add.
    }
  }
}
