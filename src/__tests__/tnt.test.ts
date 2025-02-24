import { makeData } from './makeData';
import { TNT } from '../tnt';
import { expect, test } from 'vitest';
import { Matrix } from 'ml-matrix';
import { TNTOpts } from '../types';

const illConditioned = new Matrix([
  [
    0.06355853189791905, 0.7799679257920251, 0.0019664575485265345,
    0.49785241128125124, 0.9488955201112512,
  ],
  [
    0.06819327032425537, 0.037153233827454946, 0.9247986023698862,
    0.705334939844535, 0.13307672470945064,
  ],
  [
    0.13026353337270136, 0.24163034491879132, 0.9227731156740526,
    0.2830279588620952, 0.0012315083853995379,
  ],
  [
    0.9254405073838763, 0.9132081295563979, 0.29893902393620997,
    0.27094620118832036, 0.06554637642053063,
  ],
]);
const b = Matrix.ones(illConditioned.rows, 1);
test('Many random matrices between 0 and 1', () => {
  for (let i = 0; i < 1e2; i++) {
    const m = Math.ceil(Math.random() * 12) + 2;
    const n = Math.ceil(Math.random() * 12) + 2;
    const { inputs: A, outputs: b } = makeData(m, n);
    const tnt = new TNT(A, b, {
      pseudoInverseFallback: true,
      maxIterations: 4,
      earlyStopping: { minError: 1e-8 },
    });
    expect(Number.isFinite(tnt.xBest.get(0, 0))).toBeTruthy();
    expect(tnt.mseMin).not.toBeNaN();
    expect(tnt.iterations).toBeLessThanOrEqual(tnt.maxIterations); //should be equal, but is +1 when fallbacks to pseudoInverse.
    expect(tnt.mse.length).toBeLessThanOrEqual(tnt.maxIterations + 1); // same
    console.log(tnt.mse, tnt.method);
  }
});

test('Many runs without error', () => {
  for (let i = 0; i < 1e2; i++) {
    const m = Math.ceil(Math.random() * 12) + 2;
    const n = Math.ceil(Math.random() * 12) + 2;
    const { inputs: A, outputs: b } = makeData(m, n);
    const randomRowVector = Matrix.random(1, n).multiply(100);
    const randomColumnVector = Matrix.random(m, 1).multiply(35);
    const bigA = A.mulRowVector(randomRowVector);
    const bigB = b.mulColumnVector(randomColumnVector);
    // console.log(bigA,bigB)
    const tnt = new TNT(bigA, bigB, {
      pseudoInverseFallback: true,
      maxIterations: 4,
      earlyStopping: { minError: 1e-8 },
    });
    expect(Number.isFinite(tnt.xBest.get(0, 0))).toBeTruthy();
    expect(tnt.mseMin).not.toBeNaN();
    expect(tnt.iterations).toBeLessThanOrEqual(tnt.maxIterations + 1); //should be equal, but is +1 when fallbacks to pseudoInverse.
    expect(tnt.mse.length).toBeLessThanOrEqual(tnt.maxIterations + 2); // same
    console.log(tnt.mse, tnt.method);
  }
});

test('example in the readme', () => {
  const A = new Matrix([
    [1, 2, 3],
    [4.01, 7.8, 12.2],
  ]); // 2x3
  const b = Matrix.columnVector([6, 24]); // or [[6],[7]]
  // const b2 = [[8], [3]];
  const opts: Partial<TNTOpts> = {
    maxIterations: 4,
    unacceptableError: 1e-2,
    earlyStopping: { minError: 1e-8 },
    // pseudoInverseFallback: true,
  };
  const r = new TNT(A, b, opts);
  console.log(r);
  expect(r).toBeDefined();
  // expect(new TNT(A, b2, opts)).toBeDefined();
});

test('Ill Conditioned', () => {
  expect(new TNT(illConditioned, b)).toBeDefined();
});

test('fails to optimize enough without PseudoInverse - 2', () => {
  expect(
    new TNT(Matrix.ones(5, 500), Matrix.ones(5, 1), {
      pseudoInverseFallback: false,
    }),
  ).toBeDefined();
});

test('runs fine with pseudoinverse', () => {
  const r = new TNT(Matrix.ones(5, 500), Matrix.ones(5, 1), {
    pseudoInverseFallback: true,
  });
  expect(r.method).toBe('pseudoInverse');
});

test('optimizes with Pseudo Inverse', () => {
  const result = new TNT(illConditioned, b, {
    pseudoInverseFallback: true,
  });
  expect(Number.isFinite(result.xBest.get(0, 0))).toBeTruthy();
  expect(result.mseMin).toBeLessThanOrEqual(result.mseLast);
});
