import { Matrix, AbstractMatrix } from 'ml-matrix';

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
}
export interface TNTOpts {
  /**
   * It throws an error or passes to pseudoInverse if enabled with `pseudoInverseFallback:true`.
   * @default 1E-2
   */
  unacceptableError: number;
  /**
   * @default `A.columns * 3`
   */
  maxIterations: number;
  /**
   * Use a custom precondition that has empirically been
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
   * @default true
   */;
  pseudoInverseFallback: boolean;
  /**
   * `rows/cols` ratio below which to use the pseudo-inverse.
   * This setting only has effect if `pseudoInverseFallback: true`
   *
   * A few examples (using default of 0.1):
   *    * Matrix of 100 x 2000: uses pseudo inverse.
   *    * Matrix of 200 x 2000: uses pseudo inverse.
   *    * Matrix of 201 x 2000: uses TNT
   *    * Matrix of 2000 x 2000: uses TNT

   * Why this setting? Because $A^T\,A$ becomes expensive to calculate.
   * @default 0.1
   */
  criticalRatio: number;
}

export type AnyMatrix = Matrix | AbstractMatrix;
