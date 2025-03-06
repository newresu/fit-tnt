import { AbstractMatrix, Matrix, MatrixColumnSelectionView } from 'ml-matrix';

export type Array2D = ArrayLike<ArrayLike<number>>;
export type Array1D = ArrayLike<number>;

export interface EarlyStopping {
  /**
   * If it gets below this error, it stops
   * @default 10E-20
   * Note: sufficient but not necessary condition to stop.
   * If the error is below this value, it stops.
   * It does not apply otherwise.
   */
  minMSE: number;
}
export interface TNTOpts {
  /**
   * @default `A.columns * 3`
   */
  maxIterations: number;
  /**
   * Stops the optimization on conditions.
   */
  earlyStopping: EarlyStopping;
}
/**
 * Each column of X has its own life and its metadata is stored separately.
 */
export interface ColumnInfo {
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


export type AnyMatrix = Matrix | AbstractMatrix | MatrixColumnSelectionView;
