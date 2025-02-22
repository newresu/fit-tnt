import { MatrixTransposeView, Matrix } from 'ml-matrix';

import { initSafetyChecks } from './initSafetyChecks';
import { invertLLt } from './triangularSubstitution';
import { choleskyPrecondition } from './choPrecondition';
import { meanSquaredError } from './meanSquaredError';
import { TNTOpts } from './types';

function worthContinuing(mse: number[], tolerance: number) {
  /**
   * @returns Whether we should continue or not.
   */
  const last = mse[mse.length - 1];
  if (mse.length > 0 && last < tolerance) {
    // there is already a good result
    return false;
  } else if (mse.length > 1 && last >= mse[mse.length - 2]) {
    // worse than previous (otherwise use smaller tolerance)
    return false;
  }
  return true;
}

export function _tnt(
  A: Matrix,
  b: Matrix,
  mse: number[],
  options: Partial<TNTOpts>={},
) {
  const { maxIterations = A.columns * 3, tolerance = 10e-26 } = options;

  const x = Matrix.zeros(A.columns, 1); // column of coefficients.
  const At = new MatrixTransposeView(A); // copy is ok. it's used a few times.
  const AtA = At.mmul(A); //square m. will be mutated.
  initSafetyChecks(AtA, A, b); //throws custom errors on issues.
  const choleskyDC = choleskyPrecondition(AtA);
  const L = choleskyDC.lowerTriangularMatrix;
  const AtA_inv = invertLLt(L);

  const residual = Matrix.sub(b, A.mmul(x)); // r = b - Ax_0
  let gradient = At.mmul(residual); // r_hat = At * r
  // `z_0 = AtA_inv * r_hat = x_0 - A_inv * b`
  let x_error = AtA_inv.mmul(gradient);
  const p = x_error.clone(); // z_0 clone

  let lowest_mse = Infinity;
  let it = 0;
  let sqe, alpha, beta_denom, beta, x_best; // column of coefficients.
  while (it < maxIterations && worthContinuing(mse, tolerance)) {
    sqe = residual.dot(residual); //.to1DArray, multiply and add.
    alpha = x_error.dot(gradient) / sqe;
    x.add(Matrix.mul(p, alpha));
    if (isNaN(x.get(0, 0)) || !Number.isFinite(x.get(0, 0))) {
      throw new Error(
        'Infinite or NaN coefficients were found. This may be due to a very ill-conditioned matrix. You can try `{use_SVD:true}` and it will fallback to solve using pseudoinverse.',
      );
    }
    residual.sub(residual.clone().mul(alpha)); // update residual
    beta_denom = x_error.dot(gradient); // using old values
    gradient = At.mmul(residual); //new g
    x_error = AtA_inv.mmul(gradient); // new x_error
    beta = x_error.dot(gradient) / beta_denom; // new/old ratio
    p.multiply(beta).add(x_error); // with new x_error
    mse.push(meanSquaredError(A, x, b));
    if (mse[mse.length - 1] < lowest_mse) {
      lowest_mse = mse[mse.length - 1];
      x_best = x.clone();
    }
    it++;
  }
  if (!x_best) {
    throw new Error('Internal Error calculating the solution / coefficients.');
  }
  return x_best;
}
