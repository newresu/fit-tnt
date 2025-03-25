import { Matrix, MatrixColumnSelectionView } from 'ml-matrix';

import { choleskyPrecondition } from './choPrecondition';
import { checkMatchingDimensions } from './initSafetyChecks';
import { invertLLt } from './invertLLt';
import { meanSquaredError, squaredSum } from './squaredSum';
import { symmetricMul } from './symmetricMul';
import {
  AnyMatrix,
  Array1D,
  Array2D,
  ColumnInfo,
  EarlyStopping,
  TNTOpts,
} from './types';
import { ensureMatrixB, filterIndices, getColumnViews } from './utils';

/**
 * Find the best `X` in `A X = B`; where `A` and `B` are known.
 * By 'best' it refers to the least-squares (least error) solution.
 *
 * Multiple RHS are supported (`B` can be a vector or matrix)
 * @param data the input or data matrix (2D Array)
 * @param output the known-output
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
    const B = ensureMatrixB(output);
    this.XBest = new Matrix(A.columns, B.columns);

    // unpack options
    const { maxIterations = 4, earlyStopping: { minMSE = 1e-20 } = {} } = opts;

    this.maxIterations = maxIterations;
    this.earlyStopping = { minMSE };

    this.metadata = squaredSum(B).map((x) => {
      x = x / B.rows;
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
   * 1. Append last mse for each column
   * 2. Set `XBest` and `mseMin` **iff** it improved
   * 3. Sets `indices[i]=NaN` if it didn't improve.
   * @param mseLast list of mean squared errors for each column
   * @param XView columns of X currently available.
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
        this.XBest.setColumn(indices[i], XView.getColumn(i));
      } else {
        indices[i] = NaN;
      }
    }
  }

  /**
   * Find the XBest and set it in the class.
   * @param A data matrix
   * @param B solution matrix
   */
  #tnt(A: AnyMatrix, B: AnyMatrix) {
    const X: AnyMatrix = Matrix.zeros(A.columns, B.columns);
    checkMatchingDimensions(A, X, B);

    // indices of current "on" columns of X.
    let indices = new Array(X.columns).fill(0).map((_, i) => i);
    // same but for matrices that are recalculated as it runs.
    let shiftedIndices;

    const At = A.transpose(); // copy is ok
    const AtA = symmetricMul(At);

    const choleskyDC = choleskyPrecondition(AtA);
    const L = choleskyDC.lowerTriangularMatrix;
    const AtA_inv = invertLLt(L);

    const Residual = B.clone(); // r = b - Ax_0 (Ax_0 = 0)
    let Gradient = At.mmul(Residual); // r_hat = At * r
    // z_0 = AtA_inv * r_hat = AtA_inv * At b - x_0
    let XError = AtA_inv.mmul(Gradient);
    const P = XError.clone(); // z_0 clone

    let W: Matrix;
    let ww: number[];
    let [alpha, betaDenom, beta, mseLast]: number[][] = [[], [], []];

    // These are updated with `indices`
    let [X_View, B_View, P_View]: AnyMatrix[] = [X, B, P];
    // These are updated with `shiftedIndices`
    let [GradientView, ResidualView, XErrorView]: AnyMatrix[] = [
      Gradient,
      Residual,
      XError,
    ];

    for (let it = 0; it < this.maxIterations; it++) {
      W = A.mmul(P_View);
      ww = squaredSum(W);
      alpha = Matrix.multiply(XError, Gradient)
        .sum('column')
        .map((x, i) => x / ww[i]);
      // indices of the columns to solve
      [indices, alpha, shiftedIndices] = filterIndices(indices, alpha);
      // after removing NaNs alpha may be empty.
      if (alpha.length === 0) break;

      // view of columns to solve if needed
      if (indices.length < X_View.columns) {
        [X_View, B_View, P_View] = getColumnViews(indices, X, B, P);
      }
      X_View.add(P_View.clone().mulRowVector(alpha)); //update x

      // With X updated, we need to narrow down again.
      mseLast = meanSquaredError(A, B_View, X_View);
      this.#updateMSEAndX(mseLast, X_View, indices);

      [indices, alpha, shiftedIndices] = filterIndices(
        indices,
        alpha,
        shiftedIndices,
      );
      // after removing NaNs indices may be empty
      if (indices.length === 0) break;

      if (indices.length < X_View.columns) {
        [X_View, B_View] = getColumnViews(indices, X, B);
      }

      [GradientView, XErrorView, ResidualView] = getColumnViews(
        shiftedIndices,
        Gradient,
        XErrorView,
        ResidualView,
      );

      // using old error and gradient
      betaDenom = Matrix.multiply(XErrorView, GradientView).sum('column');
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
