import { pseudoInverse, Matrix } from 'ml-matrix';

import { initSafetyChecks } from './initSafetyChecks';
import { invertLLt } from './triangularSubstitution';
import { choleskyPrecondition } from './choPrecondition';

import { TNTOpts, Array1D, Array2D, EarlyStopping, AnyMatrix } from './types';
import { meanSquaredError } from './meanSquaredError';
import { NaNOrNonFiniteError } from './Errors';
import { fastAtA } from './fastAtA';

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
  xBest: AnyMatrix;

  /**
   * @see {@link TNTOpts["unacceptableError"]}
   */
  unacceptableError: number;
  /**
   * @see {@link TNTOpts["pseudoInverseFallback"]}
   */
  pseudoInverseFallback: boolean;
  /**
   * @see {@link TNTOpts["criticalRatio"]}
   */
  criticalRatio: number;
  /**
   * @see {@link TNTOpts["maxIterations"]}
   */
  maxIterations: number;
  /**
   * @see {@link TNTOpts["earlyStopping"]}
   */
  earlyStopping: EarlyStopping;

  /**
   * Mean Squared Error for each iteration plus the initial guess (`mse[0]`)
   */
  /**
   * method
   * @default TNT
   */
  method: 'TNT' | 'pseudoInverse';
  mse: number[];

  /**
   * Minimum Mean Squared Error in all the iterations.
   * @default $||b||_2^2$ since x_0 = zero-vector initially.
   */
  mseMin: number;
  /**
   * Last MSE of all iterations.
   * @default $||b||_2^2$ since x_0 = zero-vector initially.
   */
  mseLast: number;
  /**
   * Keeps track of the patience at each iteration.
   */
  _noImprovementCounter: number;

  constructor(
    data: Array2D | AnyMatrix,
    output: Array2D | Array1D | AnyMatrix,
    opts: Partial<TNTOpts> = {},
  ) {
    const A = Matrix.isMatrix(data) ? data : new Matrix(data);
    const b = Matrix.isMatrix(output)
      ? output
      : Array.isArray(output[0])
        ? new Matrix(output as number[][])
        : Matrix.columnVector(output as number[]);
    this.xBest = new Matrix(A.columns, 1);

    // unpack options
    const {
      pseudoInverseFallback = false,
      maxIterations = 3 * A.columns,
      unacceptableError = 1e-2,
      criticalRatio = 1e-2,
      earlyStopping: { minError = 1e-20, patience = 2 } = {},
    } = opts;

    this.pseudoInverseFallback = pseudoInverseFallback;
    this.maxIterations = maxIterations;
    this.earlyStopping = { minError, patience };
    this.unacceptableError = unacceptableError;
    this.criticalRatio = criticalRatio;
    this.method = 'TNT';

    this.mse = [b.dot(b) / b.columns];
    this.mseLast = this.mseMin = this.mse[0];

    this._noImprovementCounter = 0;
    if (
      A.rows / A.columns <= this.criticalRatio &&
      this.pseudoInverseFallback
    ) {
      try {
        this._pseudoInverse(A, b);
      } catch (y) {
        if (y instanceof Error) {
          throw new Error(y.message);
        }
      }
    } else if (this.mseLast !== 0) {
      this._solve(A, b);
    }
  }

  get iterations() {
    return this.mse.length - 1;
  }

  _solve(A: AnyMatrix, b: AnyMatrix) {
    try {
      this._tnt(A, b);
    } catch (e) {
      // tnt fails
      if (this.pseudoInverseFallback) {
        // fallback
        try {
          this._pseudoInverse(A, b);
        } catch (y) {
          // fallback fails
          if (e instanceof Error && y instanceof Error) {
            throw new Error(`${e.message},\n ${y.message}`);
          }
        }
      } else {
        // no fallback try
        if (e instanceof Error) {
          throw new Error(e.message);
        }
      }
    }
  }

  /**
   * Updates mse[], mse values, counter and xBest
   * @param x current coefficients
   * @return 1 means stop, 0 means success.
   */
  _update(A: AnyMatrix, b: AnyMatrix, x: AnyMatrix, cloneX: boolean = true) {
    this.mseLast = meanSquaredError(A, x, b);
    this.mse.push(this.mseLast);
    if (this.mseLast < this.mseMin) {
      this.mseMin = this.mseLast;
      this._noImprovementCounter = 0;
      this.xBest = cloneX ? x.clone() : x;
    } else {
      this._noImprovementCounter += 1;
    }
    if (this._noImprovementCounter > this.earlyStopping.patience) {
      return 1;
    }
    if (this.mseMin < this.earlyStopping.minError) {
      return 1;
    }
  }
  /**
   * Updates best result if reacher.
   * @param A
   * @param b
   */
  _pseudoInverse(A: AnyMatrix, b: AnyMatrix) {
    const x = pseudoInverse(A).mmul(b);
    this._update(A, b, x, false); //false==no cloning of X
    if (this.mseMin === this.mseLast) {
      this.method = 'pseudoInverse';
    }
    return;
  }

  /**
   * Private function (main method)
   * @param A
   * @param b
   * @param mse this will be mutated; this allows user to get all MSEs.
   * @param options
   * @returns best-found coefficients
   */
  _tnt(A: AnyMatrix, b: AnyMatrix) {
    const x = Matrix.zeros(A.columns, 1); // column of coefficients.
    const At = A.transpose(); // copy is ok. it's used a few times.
    // const AtA = At.mmul(A); //square m. will be mutated.
    const AtA = fastAtA(At);
    initSafetyChecks(A, b); //throws custom errors on issues.
    const choleskyDC = choleskyPrecondition(AtA);
    const L = choleskyDC.lowerTriangularMatrix;
    const AtA_inv = invertLLt(L);

    const residual = b.clone(); // r = b - Ax_0 (but Ax_0 is 0)
    let gradient = At.mmul(residual); // r_hat = At * r
    // `z_0 = AtA_inv * r_hat = x_0 - A_inv * b`
    let xError = AtA_inv.mmul(gradient);
    const p = xError.clone(); // z_0 clone

    let sqe, alpha, betaDenom, beta, stop; // column of coefficients.
    for (let it = 0; it < this.maxIterations; it++) {
      sqe = residual.dot(residual); //.to1DArray, multiply and add.
      alpha = xError.dot(gradient) / sqe;
      x.add(Matrix.mul(p, alpha)); //update x
      if (!Number.isFinite(x.get(0, 0)) || !Number.isFinite(alpha)) {
        throw new NaNOrNonFiniteError();
      }
      stop = this._update(A, b, x); //updates: mse and counter and xBest
      if (stop == 1) {
        break;
      }

      betaDenom = xError.dot(gradient); // old CG (maybe)
      residual.sub(residual.clone().mul(alpha)); // update residual

      gradient = At.mmul(residual); // new g
      xError = AtA_inv.mmul(gradient); // new x_error
      beta = xError.dot(gradient) / betaDenom; // new_CG/old_CG
      p.multiply(beta).add(xError); // update p
    }
    if (this.mseMin > this.unacceptableError) {
      throw new Error(
        `TNT could not converge (its mseMin=${this.mseMin} is unacceptable)`,
      );
    }
  }
}
