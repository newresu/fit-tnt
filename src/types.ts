import { AbstractMatrix, Matrix } from 'ml-matrix';

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

export type AnyMatrix = Matrix | AbstractMatrix;
