import { MatrixTransposeView, Matrix } from 'ml-matrix';

export function lowerTriangularSubstitution(
  lowerTriangular: Matrix,
  rhs: Matrix,
) {
  const V = new Matrix(rhs.rows, rhs.columns);
  // console.log(lowerTriangular, rhs);
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
  // console.log('V from Upper: ', V);
  return V;
}

export function invertLLt(L: Matrix) {
  return upperTriangularSubstitution(
    new MatrixTransposeView(L),
    lowerTriangularSubstitution(L, Matrix.eye(L.rows)),
  );
}
