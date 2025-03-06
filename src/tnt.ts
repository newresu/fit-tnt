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

  /**
   * Information regarding the solution coefficients, their error and iterations.
   * {@link ColumnInfo}
   */
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
    // same but for matrices that are recalculated as it runs.
    let subsetIndices;

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
    let [alpha, betaDenom, beta]: number[][] = []

    // We will use views
    let [X_View, B_View, P_View]: AnyMatrix[] = [X, B, P];
    let [GradientView, ResidualView, XErrorView]: AnyMatrix[] = [
      Gradient,
      Residual,
      XError,
    ];

    for (let it = 0; it < this.maxIterations; it++) {
      W = A.mmul(P_View);
      WW = W.pow(2).sum('column');
      alpha = Matrix.multiply(XError, Gradient)
        .sum('column')
        .map((x, i) => x / WW[i]);

      [indices, alpha, subsetIndices] = updateIndices(indices, alpha);

      // get the indices of the columns to solve
      if (indices.length < X_View.columns) {
        [X_View, B_View, P_View] = getColumnViews(indices, X, B, P);
      }
      X_View.add(P_View.clone().mulRowVector(alpha)); //update x

      // With X updated, we need to narrow down again.
      this.#updateMSEAndX(A, B_View, X_View, indices);

      [indices, alpha, subsetIndices] = secondUpdateIndices(
        indices,
        subsetIndices,
        alpha,
      );

      // get the indices of the columns to solve
      if (indices.length === 0) break;
      if (indices.length < X_View.columns) {
        [X_View, B_View] = getColumnViews(indices, X, B);
      }

      [GradientView, XErrorView, ResidualView] = getColumnViews(
        subsetIndices,
        Gradient,
        XErrorView,
        ResidualView,
      );

      betaDenom = Matrix.multiply(XErrorView, GradientView).sum('column'); // old CG (maybe)
      // console.log(ResidualView)
      ResidualView.sub(ResidualView.clone().mulRowVector(alpha)); // update residual

      Gradient = At.mmul(ResidualView); // new g
      XError = AtA_inv.mmul(Gradient); // new x_error
      beta = Matrix.multiply(XError, Gradient)
        .sum('column')
        .map((x, i) => x / betaDenom[i]);

      P_View = new MatrixColumnSelectionView(P, indices);
      P_View.mulRowVector(beta).add(XError); // update p
    }
  }
}

function ensureMatrix(output: Array1D | Array2D | AnyMatrix): AnyMatrix {
  if (Matrix.isMatrix(output)) {
    return output;
  } else if (Array.isArray(output[0])) {
    return new Matrix(output as number[][]);
  }
  return Matrix.columnVector(output as number[]);
}

function updateIndices(indices: number[], alpha: number[]) {
  // first filter through alpha
  const tmpIndices = [];
  const subsetIndices = [];
  for (let i = 0; i < alpha.length; i++) {
    if (Number.isFinite(alpha[i])) {
      tmpIndices.push(indices[i]);
      subsetIndices.push(i);
    }
  }
  indices = tmpIndices;
  alpha = alpha.filter(Number.isFinite);
  return [indices, alpha, subsetIndices];
}

function secondUpdateIndices(
  indices: number[],
  subsetIndices: number[],
  alpha: number[],
) {
  const tmpIndices = [];
  const tmpSubsetIndices = [];
  const tmpAlpha = [];
  for (let i = 0; i < indices.length; i++) {
    if (Number.isFinite(indices[i])) {
      tmpIndices.push(indices[i]);
      tmpSubsetIndices.push(subsetIndices[i]);
      tmpAlpha.push(alpha[i]);
    }
  }
  indices = tmpIndices;
  subsetIndices = tmpSubsetIndices;
  alpha = tmpAlpha;
  return [indices, alpha, subsetIndices];
}

function getColumnViews(indices: number[], ...matrices: AnyMatrix[]) {
  return matrices.map((m) => new MatrixColumnSelectionView(m, indices));
}
