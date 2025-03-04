import { Matrix } from 'ml-matrix';

/**
 * Multiply a matrix by its transpose.
 * The result is a square matrix.
 * We take advantage of the symmetry to reduce the number of operations.
 * @param {Matrix} B - The matrix to multiply by its transpose.
 * @returns {Matrix} The result of `B * B^t`.

 * If `B = At` then this calculates `AtA`.
 */
export function symmetricMul(B: Matrix) {
  const BBt = new Matrix(B.rows, B.rows);
  const diagonal: number[] = [];

  let d;
  for (let i = 0; i < B.rows; i++) {
    d = 0;
    // calculate diagonal value
    for (let t = 0; t < B.columns; t++) {
      d += B.get(i, t) ** 2;
    }
    diagonal.push(d);

    // go over cols of A, using rows of At
    for (let j = 0; j < i; j++) {
      // go over other rows of A
      let terms = 0;
      for (let k = 0; k < B.columns; k++) {
        terms += B.get(i, k) * B.get(j, k);
      }
      BBt.set(i, j, terms);
    }
  }
  BBt.add(BBt.transpose());
  for (let i = 0; i < B.rows; i++) {
    BBt.set(i, i, diagonal[i]);
  }
  return BBt;
}
