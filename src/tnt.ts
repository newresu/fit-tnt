import { Matrix, MatrixColumnSelectionView } from 'ml-matrix';

import { choleskyPreconditionTrick } from './choPrecondition';
import { initSafetyChecks } from './initSafetyChecks';
import { invertLLt } from './invertLLt';
import { meanSquaredError } from './meanSquaredError';
import { symmetricMul } from './symmetricMul';
import { AnyMatrix, Array1D, Array2D, EarlyStopping, TNTOpts } from './types';

interface ColumnInfo {
  mse: number[];
  mseMin: number;
  mseLast: number;
  iterations: number;
}
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
  XBest: AnyMatrix;
  /**
   * {@link TNTOpts["maxIterations"]}
   */
  maxIterations: number;
  /**
   * {@link TNTOpts["earlyStopping"]}
   */
  earlyStopping: EarlyStopping;

  metadata: ColumnInfo[];

  constructor(
    data: Array2D | AnyMatrix,
    output: Array2D | Array1D | AnyMatrix,
    opts: Partial<TNTOpts> = {},
  ) {
    const A = Matrix.checkMatrix(data);
    const B = Matrix.isMatrix(output)
      ? output
      : Array.isArray(output[0])
        ? new Matrix(output as number[][])
        : Matrix.columnVector(output as number[]);
    this.XBest = new Matrix(A.columns, B.columns);

    // unpack options
    const {
      maxIterations = 3 * A.columns,
      earlyStopping: { minMSE = 1e-20 } = {},
    } = opts;

    this.maxIterations = maxIterations;
    this.earlyStopping = { minMSE };

    this.metadata = B.pow(2)
      .mean('column')
      .map((x) => {
        return {
          mse: [x],
          mseMin: x,
          mseLast: x,
          iterations: 0,
        };
      });

    this.#tnt(A, B);
  }

  /**
   * 1. Calculate `mseLast`
   * 2. Updates `mse[]`
   * 3. Sets `mseMin` if improved, and `xBest` in that case.
   * When some columns have been left out, both X and B are sub column views.
   * @param A input data matrix
   * @param B known output vector
   * @param X current coefficients
   * @param indices to set the results to.
   * @return void
   */
  #updateMSEAndX(
    A: AnyMatrix,
    B: AnyMatrix,
    X: AnyMatrix,
    cols2solve: number[],
  ) {
    const indices = cols2solveToColIndices(cols2solve);
    const X_sel = new MatrixColumnSelectionView(X, indices);
    const B_sel = new MatrixColumnSelectionView(B, indices);
    const mseLast = meanSquaredError(A, X_sel, B_sel);
    for (let i = 0; i < indices.length; i++) {
      const column = this.metadata[indices[i]];
      column.mse.push(mseLast[i]);
      column.mseLast = mseLast[i];
      column.iterations++;
      if (column.mseLast < column.mseMin) {
        column.mseMin = column.mseLast;
        this.XBest.setColumn(i, X.getColumn(i));
      } else {
        cols2solve[indices[i]] = 0;
      }
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
  #tnt(A: AnyMatrix, B: AnyMatrix) {
    const X = Matrix.zeros(A.columns, B.columns);

    initSafetyChecks(A, X, B);

    // binary array to keep track of which columns to solve
    const cols2solve = new Array(X.columns).fill(1);

    const At = A.transpose(); // copy is ok
    const AtA = symmetricMul(At);

    const choleskyDC = choleskyPreconditionTrick(AtA);
    const L = choleskyDC.lowerTriangularMatrix;
    const AtA_inv = invertLLt(L);

    const Residual = B.clone(); // r = b - Ax_0 (but Ax_0 is 0)
    let Gradient = At.mmul(Residual); // r_hat = At * r
    // `z_0 = AtA_inv * r_hat = x_0 - A_inv * b`
    let XError = AtA_inv.mmul(Gradient);
    const P = XError.clone(); // z_0 clone

    let W: Matrix;
    let WW: number[];
    let alpha: number[];
    let betaDenom: number[];
    let beta: number[];

    for (let it = 0; it < this.maxIterations; it++) {
      W = A.mmul(P);
      WW = W.pow(2).sum('column');
      alpha = Matrix.multiply(XError, Gradient)
        .sum('column')
        .map((x, i) => x / WW[i]);

      X.add(P.clone().mulRowVector(alpha)); //update x

      for (let i = 0; i < X.columns; i++) {
        if (!Number.isFinite(alpha[i])) {
          cols2solve[i] = 0;
        }
      }
      this.#updateMSEAndX(A, B, X, cols2solve); //updates: mse and counter and xBest

      if (cols2solve.every((x) => x === 0)) break;

      betaDenom = Matrix.multiply(XError, Gradient).sum('column'); // old CG (maybe)
      Residual.sub(Residual.clone().mulRowVector(alpha)); // update residual

      Gradient = At.mmul(Residual); // new g
      XError = AtA_inv.mmul(Gradient); // new x_error
      beta = Matrix.multiply(XError, Gradient)
        .sum('column')
        .map((x, i) => x / betaDenom[i]);

      P.mulRowVector(beta).add(XError); // update p
    }
  }
}

function cols2solveToColIndices(arr: number[]) {
  const indices: number[] = [];
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] === 1) {
      indices.push(i);
    }
  }
  return indices;
}
