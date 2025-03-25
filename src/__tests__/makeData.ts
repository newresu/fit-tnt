import { Matrix } from 'ml-matrix';

/** Options for generating data.  */
interface MakeDataOpts {
  /**
   * Whether to include a bias term in the generated data.
   * @default false
   */
  useBias: boolean;

  /**
   * Number of output/coefficient columns to generate.
   * @default 1
   */
  outputColumns: number;

  /**
   * Whether to add noise to the output data `B`.
   * If true, noise is added using 1/100 of the scaling factor of `X`.
   * @default true
   */
  addNoise: boolean;

  /**
   * Scale the input data (`X`) by this number.
   * @default 1 (no scaling)
   */
  scaleX: number;

  /**
   * Scale the coefficients (`A`) by this number.
   * @default 1 (no scaling)
   */
  scaleA: number;
}
/**
 * Make `A`, `B` and `X` matrices to use for testing purposes.
 * @param samples how many samples to use
 * @param coefficients how many `X` or `B` columns to use.
 * @param opts {@link MakeDataOpts}
 * @returns `A`, `B`, and `X`
 */
export function makeData(
  samples: number,
  coefficients: number, // do not include bias here.
  opts: Partial<MakeDataOpts> = {},
) {
  const {
    scaleX = 1,
    scaleA = 1,
    outputColumns = 1,
    useBias = false,
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

/**
 * Random value with random sign.
 * @returns random value with random sign.
 */
function myRandom() {
  const randomSign = Math.random() > 0.5 ? -1 : 1;
  return Math.random() * randomSign;
}
