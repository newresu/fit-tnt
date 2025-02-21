import { Matrix } from "ml-matrix";

interface MakeDataOpts {
  useBias: boolean;
  outputColumns: number; //default 1
}
export function makeData(
  samples: number,
  coefficients: number, // do not include bias here.
  opts: Partial<MakeDataOpts> = {}
) {
  /**
   * Make some random samples with a number of "coefficients".
   */
  const { useBias = false, outputColumns = 1 } = opts;
  // design matrix / input data
  const A = Matrix.random(samples, coefficients);
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

  return { inputs: A, outputs: B, coefficients: X };
}

// console.log(makeData(5, 2, { outputColumns: 1 }));
