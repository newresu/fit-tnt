import { Matrix, MatrixTransposeView } from "ml-matrix";

// back substitution for A X = B
// where X and B can be vectors or matrices
// A can be either lower or upper triangular.
function lowerTriangularSubstitution(lowerTriangular: Matrix, rhs: Matrix) {
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

function upperTriangularSubstitution(upperTriangular: Matrix|MatrixTransposeView, rhs: Matrix) {
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

export {upperTriangularSubstitution, lowerTriangularSubstitution}