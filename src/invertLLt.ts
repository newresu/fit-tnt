import { Matrix } from 'ml-matrix';

import { fastAtA } from './fastAtA';

/**
 * Performs the lower triangular substitution (starts from the top-left.)
 * @param lowerTriangular
 * @param rhs supports multiple right hand sides.
 * @returns solution to the system of equations
 */
export function lowerTriangularInverse(lowerTriangular: Matrix) {
  let terms;
  const { rows } = lowerTriangular;
  const V = new Matrix(rows, rows);
  for (let i = 0; i < rows; i++) {
    for (let k = 0; k < rows; k++) {
      terms = i === k ? 1 : 0;
      for (let j = 0; j <= i - 1; j++) {
        terms -= lowerTriangular.get(i, j) * V.get(j, k);
      }
      V.set(i, k, terms / lowerTriangular.get(i, i));
    }
  }
  return V;
}
/**
 * Solve the system  `LL^T X = I`
 * To obtain the inverse. This involves solving two systems of equations.
 *
 * However, calculating $L^{-1}$, we can find the other one.
 * So it's really just one system and one matmul.
 
 * @param L lower triangular.
 * @returns inverse
 */
export function invertLLt(L: Matrix) {
  return fastAtA(lowerTriangularInverse(L).transpose());
}
