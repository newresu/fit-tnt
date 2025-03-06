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
  #updateMSEAndX(A: AnyMatrix, B: AnyMatrix, X: AnyMatrix, indices: number[]) {
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
        indices[i] = NaN;
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
    const X: AnyMatrix = Matrix.zeros(A.columns, B.columns);
    initSafetyChecks(A, X, B);

    // binary array to keep track of which columns to solve
    let indices = new Array(X.columns).fill(0).map((_, i) => i);
    let newIndices: number[] = [];
    let subsetIndices: number[] = [];
    let subsetNewIndices: number[] = [];

    const At = A.transpose(); // copy is ok
    const AtA = symmetricMul(At);

    const choleskyDC = choleskyPreconditionTrick(AtA);
    const L = choleskyDC.lowerTriangularMatrix;
    const AtA_inv = invertLLt(L);

    const Residual = B.clone(); // r = b - Ax_0 (but Ax_0 is 0)
    let Gradient: AnyMatrix = At.mmul(Residual); // r_hat = At * r
    // `z_0 = AtA_inv * r_hat = x_0 - A_inv * b`
    let XError = AtA_inv.mmul(Gradient);
    const P = XError.clone(); // z_0 clone

    let W: Matrix;
    let WW: number[];
    let alpha: number[];
    let betaDenom: number[];
    let beta: number[];
    let X_View = X;
    let B_View = B;
    let GradientView = Gradient;
    let P_View: AnyMatrix = P;
    let ResidualView = Residual;
    let X_ErrorView: AnyMatrix = XError;

    for (let it = 0; it < this.maxIterations; it++) {
      W = A.mmul(P_View);
      WW = W.pow(2).sum('column');
      alpha = Matrix.multiply(XError, Gradient)
        .sum('column')
        .map((x, i) => x / WW[i]);

      // first filter through alpha
      newIndices = [];
      subsetIndices = [];
      for (let i = 0; i < alpha.length; i++) {
        if (Number.isFinite(alpha[i])) {
          newIndices.push(indices[i]);
          subsetIndices.push(i);
        }
      }
      if (newIndices.length === 0) break;
      alpha = alpha.filter(Number.isFinite);

      // get the indices of the columns to solve
      if (newIndices.length < X_View.columns) {
        X_View = new MatrixColumnSelectionView(X, newIndices);
        B_View = new MatrixColumnSelectionView(B, newIndices);
        P_View = new MatrixColumnSelectionView(P, newIndices);
      }
      X_View.add(P_View.clone().mulRowVector(alpha)); //update x

      // narrow down again, for the ones that didn't improve
      this.#updateMSEAndX(A, B_View, X_View, newIndices);

      indices = newIndices;
      newIndices = [];
      subsetNewIndices = [];
      const new_alpha: number[] = [];
      for (let i = 0; i < indices.length; i++) {
        if (Number.isFinite(indices[i])) {
          newIndices.push(indices[i]);
          subsetNewIndices.push(subsetIndices[i]);
          new_alpha.push(alpha[i]);
        }
      }
      alpha = new_alpha;
      // get the indices of the columns to solve
      if (newIndices.length === 0) break;
      if (newIndices.length < X_View.columns) {
        X_View = new MatrixColumnSelectionView(X, newIndices);
        B_View = new MatrixColumnSelectionView(B, newIndices);
      }

      GradientView = new MatrixColumnSelectionView(Gradient, subsetNewIndices);
      X_ErrorView = new MatrixColumnSelectionView(XError, subsetNewIndices);
      ResidualView = new MatrixColumnSelectionView(Residual, subsetNewIndices);

      betaDenom = Matrix.multiply(X_ErrorView, GradientView).sum('column'); // old CG (maybe)
      // console.log(ResidualView)
      ResidualView.sub(ResidualView.clone().mulRowVector(alpha)); // update residual

      Gradient = At.mmul(ResidualView); // new g
      XError = AtA_inv.mmul(Gradient); // new x_error
      beta = Matrix.multiply(XError, Gradient)
        .sum('column')
        .map((x, i) => x / betaDenom[i]);

      P_View = new MatrixColumnSelectionView(P, newIndices);
      P_View.mulRowVector(beta).add(XError); // update p
      indices = newIndices;
    }
  }
}

function ensureMatrix(output: Array1D | Array2D | AnyMatrix): AnyMatrix {
  return Matrix.isMatrix(output)
    ? output
    : Array.isArray(output[0])
      ? new Matrix(output as number[][])
      : Matrix.columnVector(output as number[]);
}
