import { AbstractMatrix, Matrix } from 'ml-matrix';

export type Array2D = ArrayLike<ArrayLike<number>>;
export type Array1D = ArrayLike<number>;

export interface EarlyStopping {
  /**
   * If it gets below this error, it stops
   * @default 10E-20
   * Note: In many cases, it will still return a larger error,
   * because afterNRounds was reached
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
