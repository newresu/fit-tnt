import type { Matrix } from 'ml-matrix';
import { CholeskyDecomposition } from 'ml-matrix';
import { PreconditionError } from './Errors';

export function choleskyPrecondition(AtA: Matrix) {
  /**
   * Add epsilon to the diagonal until AtA is positive definite.
   * **Note** that we are changing AtA
   * For info on [MDN:EPSILON](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/EPSILON)
   * @param AtA - Symmetric matrix from the normal equation.
   * @returns Cholesky Decomposition of AtA
   */
  const max_avg = Math.max(...AtA.abs().mean('column')) / 100;
  let epsilon = Number.EPSILON * 100; // order of magnitude max column
  if (max_avg > 1) {
    epsilon *= max_avg;
  }

  let choleskyDC = new CholeskyDecomposition(AtA);
  let it = 5; //max 2 iterations
  while (!choleskyDC.isPositiveDefinite()) {
    if (!Number.isFinite(epsilon) || !it) {
      //includes isNaN
      throw new PreconditionError();
    }
    for (let i = 0; i < AtA.rows; i++) {
      AtA.set(i, i, AtA.get(i, i) + epsilon);
    }
    epsilon *= 10;
    choleskyDC = new CholeskyDecomposition(AtA); //again
    it--;
  }
  return choleskyDC;
}

/**
 * Use diagonal of L and PD to improve the matrix.
 * @param AtA
 * @returns
 */
export function choleskyPreconditionTrick(AtA: Matrix) {
  let choleskyDC = new CholeskyDecomposition(AtA);

  let diag = choleskyDC.lowerTriangularMatrix.diagonal();
  positiveArray(diag);
  let [min, avg] = newMinAvg(diag, 0, 0);

  let ratio = (min + Number.EPSILON) / (avg + Number.EPSILON);
  let epsilon = min + Number.EPSILON;
  let it = 2000; //max 2 iterations
  while (ratio < 1e-5 || !choleskyDC.isPositiveDefinite()) {
    if (!Number.isFinite(epsilon) || !it) {
      //includes isNaN
      throw new PreconditionError();
    }
    for (let i = 0; i < AtA.rows; i++) {
      AtA.set(i, i, AtA.get(i, i) + epsilon);
    }
    epsilon *= 10;
    choleskyDC = new CholeskyDecomposition(AtA); //again
    diag = choleskyDC.lowerTriangularMatrix.diagonal();
    positiveArray(diag);
    [min, avg] = newMinAvg(diag, min, avg);
    ratio = (min + Number.EPSILON) / (avg + Number.EPSILON);

    it--;
  }
  return choleskyDC;
}

function positiveArray(arr: number[]) {
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < 0) {
      arr[i] = -arr[i];
    }
  }
}
function newMinAvg(diag: number[], min: number, avg: number) {
  min = diag[0];
  avg = 0;
  for (let i = 0; i < diag.length; i++) {
    avg += diag[i];
    if (diag[i] < min) {
      min = diag[i];
    }
  }
  return [min, avg / diag.length];
}
