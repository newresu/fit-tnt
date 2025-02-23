export type Array2D = ArrayLike<ArrayLike<number>>;
export type Array1D = ArrayLike<number>;
export interface EarlyStopping {
  /**
   * If it gets below this error, it stops
   * @default 10E-20
   * Note: In many cases, it will still return a larger error,
   * because afterNRounds was reached
   */
  minError: number;
  /**
   * As in Keras, number of iterations to wait without improvement
   * 3 is normally enough
   * @default 3
   */
  patience: number;
}
export interface TNTOpts {
  /**
   * @default `A.columns * 3`
   */
  maxIterations: number;
  /**
   * Combinations are additive: one is sufficient to stop.
   */
  earlyStopping: EarlyStopping /**
   * If the software errors (normally very ill-conditioned matrix) it fallbacks to the slower pseudo-inverse.
   * In this case, the last `mse` is the pseudo inverse's, `iterations` includes both.
   * @default false
   */;
  pseudoInverseFallback: boolean;
}