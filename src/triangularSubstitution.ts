import { MatrixTransposeView, Matrix } from 'ml-matrix';

/**
 * Performs the lower triangular substitution (starts from the top-left.)
 * @param lowerTriangular
 * @param rhs supports multiple right hand sides.
 * @returns solution to the system of equations
 */
export function lowerTriangularSubstitution(
  lowerTriangular: Matrix,
  rhs: Matrix,
) {
  const V = new Matrix(rhs.rows, rhs.columns);
  for (let i = 0; i < lowerTriangular.rows; i++) {
    for (let k = 0; k < rhs.columns; k++) {
      let terms = rhs.get(i, k);
      for (let j = 0; j <= i - 1; j++) {
        terms -= lowerTriangular.get(i, j) * V.get(j, k);
      }
      V.set(i, k, terms / lowerTriangular.get(i, i));
    }
  }
  return V;
}

/**
 * Performs the upper triangular substitution (starts from the bottom-right.)
 * @param upperTriangular
 * @param rhs supports multiple right hand sides.
 * @returns solution to the system of equations
 */
export function upperTriangularSubstitution(
  upperTriangular: Matrix | MatrixTransposeView,
  rhs: Matrix,
) {
  const V = new Matrix(rhs.rows, rhs.columns);
  for (let i = upperTriangular.rows - 1; i >= 0; i--) {
    for (let k = 0; k < rhs.columns; k++) {
      let terms = rhs.get(i, k);
      for (let j = upperTriangular.columns - 1; j >= i + 1; j--) {
        terms -= upperTriangular.get(i, j) * V.get(j, k);
      }
      V.set(i, k, terms / upperTriangular.get(i, i));
    }
  }
  return V;
}

/**
 * Invert LLt by using identity i.e `LLt x = I`
 * @param L lower triangular.
 * @returns inverse
 */
export function invertLLt(L: Matrix) {
  return upperTriangularSubstitution(
    new MatrixTransposeView(L),
    lowerTriangularSubstitution(L, Matrix.eye(L.rows)),
  );
}
