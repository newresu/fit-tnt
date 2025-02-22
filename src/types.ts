export type Array2D = ArrayLike<ArrayLike<number>>;
export type Array1D = ArrayLike<number>;
export interface TNTOpts {
  /**
   * @default `A.columns * 3`
   */
  maxIterations: number;
  /**
   * When current_error < tolerance it stops.
   * @default 10E-20
   */
  tolerance: number;
  /**
   * If the software errors (normally very ill-conditioned matrix) it fallbacks to the slower pseudo-inverse.
   * In this case, the last `mse` is the pseudo inverse's, `iterations` includes both.
   * @default false
   */
  pseudoinverseFallback: boolean;
}
export interface TNTResults {
  iterations: number;
  /**
   * Mean Squared Error i.e `(Ax-b)*(Ax-b)'/n`
   * At each iteration.
   */
  mse: number[];
  /**
   * The coefficients
   */
  solution: Array1D;
}
