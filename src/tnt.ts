import { Matrix, MatrixColumnSelectionView } from 'ml-matrix';

import { choleskyPreconditionTrick } from './choPrecondition';
import { initSafetyChecks } from './initSafetyChecks';
import { invertLLt } from './invertLLt';
import { meanSquaredError } from './meanSquaredError';
import { symmetricMul } from './symmetricMul';
import { AnyMatrix, Array1D, Array2D, EarlyStopping, TNTOpts } from './types';

/**
 * Each column of X has its own life and its metadata is stored separately.
 */
interface ColumnInfo {
  /**
   * Mean Squared Error.
   */
  mse: number[];
  /**
   * Minimum Mean Squared Error in all iterations
   * It is also the Mean Squared Error of the returned coefficients.
   */
  mseMin: number;
  /**
   * Mean Squared Error in the last iteration.
   */
  mseLast: number;
  iterations: number;
}

/**
 * Find the best $X$ in $A X = B$; where $A$ and $B$ are known.
 * By 'best' it refers to the least-squares (least error) solution.
 *
 * Multiple RHS are supported (i.e $B$ can be a vector or matrix)
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
   * 3. Sets `mseMin` if improved, and `XBest` in that case.
   * When some columns have been left out, both X and B are sub column views.
   * @param A input data matrix
   * @param B known output. Note that this will be a View.
   * @param X coefficients. Note that this will be a View.
   * @param indices track which columns of initial X are optimized.
   */
  #updateMSEAndX(mseLast: number[], XView: AnyMatrix, indices: number[]) {
    for (let i = 0; i < indices.length; i++) {
      const columnInfo = this.metadata[indices[i]];
      columnInfo.mse.push(mseLast[i]);
      columnInfo.mseLast = mseLast[i];
      columnInfo.iterations++;
      if (columnInfo.mseLast < columnInfo.mseMin) {
        columnInfo.mseMin = columnInfo.mseLast;
        this.XBest.setColumn(i, XView.getColumn(i));
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
    let [alpha, betaDenom, beta]: number[][] = [[], [], []];

    // These are updated with `indices`
    let [X_View, B_View, P_View]: AnyMatrix[] = [X, B, P];
    // These are updated with `subsetIndices`
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

      if (alpha.length === 0) break;
      // indices of the columns to solve
      [indices, alpha, subsetIndices] = updateIndices(indices, alpha);

      // view of columns to solve if needed
      if (indices.length < X_View.columns) {
        [X_View, B_View, P_View] = getColumnViews(indices, X, B, P);
      }
      X_View.add(P_View.clone().mulRowVector(alpha)); //update x

      // With X updated, we need to narrow down again.
      const mseLast = meanSquaredError(A, X_View, B_View);
      this.#updateMSEAndX(mseLast, X_View, indices);

      [indices, alpha, subsetIndices] = updateIndices(
        indices,
        alpha,
        subsetIndices,
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

/**
 * The output $B$ can be passed as flat array, nested array or matrix.
 * This function ensures that it is correctly converted to matrix in those
 * cases.
 *
 * @param B as input
 * @returns B as a matrix.
 */
function ensureMatrix(input: Array1D | Array2D | AnyMatrix): AnyMatrix {
  if (Matrix.isMatrix(input)) {
    return input;
  } else if (Array.isArray(input[0])) {
    return new Matrix(input as number[][]);
  }
  return Matrix.columnVector(input as number[]);
}

function updateIndices(
  indices: number[],
  alpha: number[],
  subsetIndices?: number[],
) {
  const tmpIndices = [];
  const tmpAlpha = [];
  const tmpSubsetIndices = [];
  if (subsetIndices) {
    for (let i = 0; i < indices.length; i++) {
      if (Number.isFinite(indices[i])) {
        tmpIndices.push(indices[i]);
        tmpSubsetIndices.push(subsetIndices[i]);
        tmpAlpha.push(alpha[i]);
      }
    }
  } else {
    for (let i = 0; i < alpha.length; i++) {
      if (Number.isFinite(alpha[i])) {
        tmpIndices.push(indices[i]);
        tmpAlpha.push(alpha[i]);
        tmpSubsetIndices.push(i);
      }
    }
  }
  return [tmpIndices, tmpAlpha, tmpSubsetIndices];
}

/**
 * Generate a matrix column view for the matrices from the indices.
 * @param indices
 * @param matrices
 * @returns views.
 */
function getColumnViews(indices: number[], ...matrices: AnyMatrix[]) {
  return matrices.map((m) => new MatrixColumnSelectionView(m, indices));
}
