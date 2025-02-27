import { Matrix } from 'ml-matrix';

interface MakeDataOpts {
  useBias: boolean; // false
  outputColumns: number; //default 1
  addNoise: boolean; // true. Add noise to A using another random.
}
export function makeData(
  samples: number,
  coefficients: number, // do not include bias here.
  opts: Partial<MakeDataOpts> = {},
) {
  /**
   * Make some random samples with a number of "coefficients".
   */
  const { useBias = false, outputColumns = 1, addNoise = true } = opts;
  // design matrix / input data
  const A = Matrix.random(samples, coefficients, {
    random: myRandom,
  });
  // coefficients matrix
  const X = Matrix.random(coefficients, outputColumns);
  // output matrix
  const B = A.mmul(X);
  if (useBias) {
    // row vector
    const b = Matrix.random(1, outputColumns);
    B.addRowVector(b);
    return { inputs: A, outputs: B, coefficients: X, bias: b };
  }
  if (addNoise) {
    A.add(Matrix.random(samples, coefficients, { random: myRandom }).div(100));
  }
  return { inputs: A, outputs: B, coefficients: X };
}

function myRandom() {
  const randomSign = Math.random() > 0.5 ? -1 : 1;
  return Math.random() * randomSign;
}
