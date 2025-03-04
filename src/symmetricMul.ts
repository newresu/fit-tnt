import { Matrix } from 'ml-matrix';

/**
 * Multiply a matrix by its transpose.
 * The result is a square matrix.
 * To calculate `AtA` pass `At`, to calculate `AAt` pass `A`.
 *
 * We take advantage of the symmetry to reduce the number of operations.
 * @param {Matrix} B - The matrix to multiply by its transpose.
 * @returns {Matrix} The result of `B * B^t`.
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
      // another opts is to transpose add outside the loop
      // but same time and less memory to just set both
      BBt.set(j, i, terms);
      BBt.set(i, j, terms);
    }
  }
  // BBt.add(BBt.transpose());
  for (let i = 0; i < B.rows; i++) {
    BBt.set(i, i, diagonal[i]);
  }
  return BBt;
}

/**
 * Multiply a upper triangular by its transpose.
 * If you need `Lt=U` and `U * Lt`, then use this function, passing `U`
 * Take advantage of symmetry.
 * @param {Matrix} L - The matrix to multiply by its transpose.
 * @returns {Matrix} The result of `B * B^t`.

 * If `B = At` then this calculates `AtA`.
 */
export function symmetricMulUpperLower(U: Matrix) {
  const R = new Matrix(U.rows, U.rows);
  const diagonal: number[] = [];

  let d;
  for (let i = 0; i < U.rows; i++) {
    d = 0;
    // calculate diagonal value (by itself)
    for (let t = 0; t < U.columns; t++) {
      d += U.get(i, t) ** 2;
    }
    diagonal.push(d);

    // go over cols of A, using rows of At
    for (let j = 0; j < i; j++) {
      // go over other rows of A
      let terms = 0;
      for (let k = i; k < U.columns; k++) {
        terms += U.get(i, k) * U.get(j, k);
      }
      R.set(j, i, terms);
      R.set(i, j, terms);
    }
  }
  for (let i = 0; i < U.rows; i++) {
    R.set(i, i, diagonal[i]);
  }
  return R;
}
