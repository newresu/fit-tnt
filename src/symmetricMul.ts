import { Matrix } from 'ml-matrix';

/**
 * Multiply a matrix by its transpose.
 * To calculate `AtA` pass `At`, to calculate `AAt` pass `A`.
 *
 * Uses symmetry and contiguity to increase speed.
 * @param B - The matrix to multiply by its transpose.
 * @returns Square matrix, result of `B * B^t`.
 */
export function symmetricMul(B: Matrix) {
  const BBt = new Matrix(B.rows, B.rows);

  let d, terms;
  for (let i = 0; i < B.rows; i++) {
    //row i
    d = 0;
    // set diagonal value
    for (let t = 0; t < B.columns; t++) {
      d += B.get(i, t) ** 2;
    }
    BBt.set(i, i, d);

    // row_i x all_prev_rows; filling up an L
    // but we set both: L and L'
    for (let j = 0; j < i; j++) {
      terms = 0;
      for (let k = 0; k < B.columns; k++) {
        // dot prod
        terms += B.get(i, k) * B.get(j, k);
      }
      BBt.set(j, i, terms);
      BBt.set(i, j, terms);
    }
  }
  // BBt.add(BBt.transpose()); // no speed up, more memory.
  return BBt;
}

/**
 * Compute `U * L`.
 *
 * Take advantage of symmetry.
 * @param U - The upper triangular matrix to multiply by its transpose.
 * @returns `U * L`
 */
export function symmetricMulUpperLower(U: Matrix) {
  const R = new Matrix(U.rows, U.rows);

  let d, terms;
  for (let i = 0; i < U.rows; i++) {
    d = 0;
    for (let t = 0; t < U.columns; t++) {
      d += U.get(i, t) ** 2;
    }
    R.set(i, i, d);

    for (let j = 0; j < i; j++) {
      terms = 0;
      for (let k = i; k < U.columns; k++) {
        terms += U.get(i, k) * U.get(j, k);
      }
      R.set(j, i, terms);
      R.set(i, j, terms);
    }
  }
  return R;
}
