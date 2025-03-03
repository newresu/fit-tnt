import { Matrix } from 'ml-matrix';

/**
 * We know that At * A is symmetric,
 * We calculate half of it and add the transpose.
 * The diagonal is added afterwards.
 *
 * The reason to use the transpose is that selecting rows may be faster
 * due to contiguity of the data.
 */
export function fastAtA(At: Matrix) {
  const AtA = new Matrix(At.rows, At.rows);
  const diagonal: number[] = [];

  let d;
  for (let i = 0; i < At.rows; i++) {
    d = 0;
    // calculate diagonal value
    for (let t = 0; t < At.columns; t++) {
      d += At.get(i, t) ** 2;
    }
    diagonal.push(d);

    // go over cols of A, using rows of At
    for (let j = 0; j < i; j++) {
      // go over other rows of A
      let terms = 0;
      for (let k = 0; k < At.columns; k++) {
        terms += At.get(i, k) * At.get(j, k);
      }
      AtA.set(i, j, terms);
    }
  }
  AtA.add(AtA.transpose());
  for (let i = 0; i < At.rows; i++) {
    AtA.set(i, i, diagonal[i]);
  }
  return AtA;
}
