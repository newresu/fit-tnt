import { Matrix, AbstractMatrix } from 'ml-matrix';

export type Array2D = ArrayLike<ArrayLike<number>>;
export type Array1D = ArrayLike<number>;

/**
 * @export
 */
export interface EarlyStopping {
  /**
   * If it gets below this error, it stops
   * @default 10E-20
   * Note: In many cases, it will still return a larger error,
   * because afterNRounds was reached
   */
  minError: number;
}
/**
 * @export
 */
export interface TNTOpts {
  /**
   * We are not interested in errors above this number.
   * It throws an error or passes to pseudoInverse if that is `true`.
   * @default 1E-2
   */
  unacceptableError: number;
  /**
   * @default `A.columns * 3`
   */
  maxIterations: number;
  /**
   * This uses a custom precondition that has empirically been
   * shown to work in our tests.
   *
   * @default true
   */
  usePreconditionTrick: boolean;
  /**
   * Stops the optimization on conditions.
   */
  earlyStopping: EarlyStopping /**
   * If the software errors (normally very ill-conditioned matrix) it fallbacks to the slower pseudo-inverse.
   * In this case, the last `mse` is the pseudo inverse's, `iterations` includes both.
   * @default false
   */;
  pseudoInverseFallback: boolean;
  /**
   * If you have a matrix of 2 x 1000
   * the AtA is 1000 x 1000.
   * In this case TNT will be slower than pseudoInverse (`pI`),
   * As a rule of thumb when ratio is < 1/100 one should use `pI`
   * This setting only has meaning if `pseudoInverse==true`
   * @default 0.1
   */
  criticalRatio: number;
}

export type AnyMatrix = Matrix | AbstractMatrix;
