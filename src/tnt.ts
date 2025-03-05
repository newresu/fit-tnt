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
    const B = ensureMatrix(output);
    this.XBest = new Matrix(A.columns, B.columns);

    // unpack options
    const {
      maxIterations = 3 * A.columns,
      earlyStopping: { minMSE = 1e-20 } = {},
    } = opts;

    this.maxIterations = maxIterations;
    this.earlyStopping = { minMSE };

    this.metadata = Matrix.pow(B, 2)
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
    const mseLast = meanSquaredError(A, X, B);
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
    let indices: number[];
    let X_View: Matrix | MatrixColumnSelectionView = X;
    let B_View: Matrix | MatrixColumnSelectionView = B;
    const X_ErrorView: Matrix | MatrixColumnSelectionView = XError;
    let GradientView: Matrix | MatrixColumnSelectionView = Gradient;
    const P_ErrorView = P;
    let ResidualView = Residual;

    for (let it = 0; it < this.maxIterations; it++) {
      W = A.mmul(P_ErrorView);
      WW = W.pow(2).sum('column');
      alpha = Matrix.multiply(X_ErrorView, GradientView)
        .sum('column')
        .map((x, i) => x / WW[i]);

      for (let i = 0; i < X_View.columns; i++) {
        if (!Number.isFinite(alpha[i])) {
          cols2solve[i] = 0;
        }
      }

      indices = cols2solveToColIndices(cols2solve);
      if (indices.length < cols2solve.length) {
        alpha = indices.map((x) => alpha[x]);
        X_View = new MatrixColumnSelectionView(X, indices);
        B_View = new MatrixColumnSelectionView(B, indices);
        GradientView = new MatrixColumnSelectionView(Gradient, indices);
        ResidualView = new MatrixColumnSelectionView(Residual, indices);
      }
      X_View.add(P_ErrorView.clone().mulRowVector(alpha)); //update x

      this.#updateMSEAndX(A, B_View, X_View, cols2solve); //updates: mse and counter and xBest

      if (cols2solve.every((x) => x === 0)) break;

      betaDenom = Matrix.multiply(X_ErrorView, GradientView).sum('column'); // old CG (maybe)
      ResidualView.sub(ResidualView.clone().mulRowVector(alpha)); // update residual

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

function ensureMatrix(output: Array1D | Array2D | AnyMatrix): AnyMatrix {
  return Matrix.isMatrix(output)
    ? output
    : Array.isArray(output[0])
      ? new Matrix(output as number[][])
      : Matrix.columnVector(output as number[]);
}
