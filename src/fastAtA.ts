import { Matrix } from 'ml-matrix';
export function fastAtA(At: Matrix) {
  /**
   * When multiplying At * A, since we know the result is symmetric,
   * we can just calculate half of it, and we only need the transpose.
   *
   * The reason to use the transpose is that selecting rows may be faster
   * due to contiguity of the data.
   */
  const AtA = new Matrix(At.rows, At.rows);
  const diagonal: number[] = [];
  // let row_i;
  let d;
  for (let i = 0; i < At.rows; i++) {
    d = 0;
    // calculate diagonal value
    // row_i = At.getRow(i);
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
