import { Matrix } from 'ml-matrix';

interface MakeDataOpts {
  useBias: boolean; // false
  outputColumns: number; //default 1
  addNoise: boolean; // true. Add noise to `b` using 1/100 of scaling of X
  scaleX: number; // Scale the X by this number.
  scaleA: number; // Scale the A by this number.
}
export function makeData(
  samples: number,
  coefficients: number, // do not include bias here.
  opts: Partial<MakeDataOpts> = {},
) {
  /**
   * Make some random samples with a number of "coefficients".
   */
  const {
    useBias = false,
    scaleX = 1,
    scaleA = 1,
    outputColumns = 1,
    addNoise = true,
  } = opts;
  // design matrix / input data
  const A = Matrix.random(samples, coefficients, {
    random: myRandom,
  });

  if (scaleA !== 1) {
    // before adding bias
    A.mul(scaleA);
  }

  if (useBias) {
    A.addColumn(Matrix.ones(samples, 1));
  }

  const X = Matrix.random(
    useBias ? coefficients + 1 : coefficients,
    outputColumns,
  );
  if (scaleX !== 1) {
    X.mul(scaleX);
  }

  const B = A.mmul(X);
  if (addNoise) {
    B.add(
      Matrix.random(samples, outputColumns, { random: myRandom }).mul(
        scaleX / 100,
      ),
    );
  }
  return { inputs: A, outputs: B, coefficients: X };
}

function myRandom() {
  const randomSign = Math.random() > 0.5 ? -1 : 1;
  return Math.random() * randomSign;
}
