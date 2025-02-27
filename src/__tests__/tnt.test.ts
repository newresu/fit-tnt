import { Matrix } from 'ml-matrix';
import { expect, test } from 'vitest';

import { TNT } from '../tnt';
import { TNTOpts } from '../types';
import { makeData } from './makeData';

test('Simple Linear Regression', () => {
  const A = new Matrix([
    //col of data
    [0.9242517859627595],
    [0.0818550256160202],
    [-0.624983105290315],
    [-0.82626649326827],
    [-0.7675865514669384],
    [-0.34794526363377654],
    [-0.6488463103599258],
    [0.6235196593971422],
    [0.9077898598099983],
    [0.34626125969154886],
  ]);
  const b = [
    //col of results
    0.23773114859044794, 0.02105431609994324, -0.16075484378482774,
    -0.21252789063518465, -0.19743454683480463, -0.08949663763332648,
    -0.16689281098857903, 0.16037842398411126, 0.23349689914484856,
    0.08906348705953603,
  ];
  const { xBest, mseMin, iterations, mse, maxIterations } = new TNT(A, b, {
    maxIterations: 4,
    usePreconditionTrick: true,
    earlyStopping: { minError: 1e-15 },
  });
  expect(Number.isFinite(xBest.get(0, 0))).toBeTruthy();
  expect(mseMin).not.toBeNaN();
  expect(iterations).toBeLessThanOrEqual(maxIterations);
  expect(mse.length).toBeLessThanOrEqual(maxIterations + 1);
  expect(xBest.get(0, 0)).toBeCloseTo(0.257214702964);
});
test('Many random matrices between 0 and 1', () => {
  for (let i = 0; i < 1e2; i++) {
    const m = Math.ceil(Math.random() * 12) + 2;
    const n = Math.ceil(Math.random() * 12) + 2;
    const { inputs: A, outputs: b } = makeData(m, n);
    const { xBest, mseMin, iterations, mse, maxIterations } = new TNT(A, b, {
      maxIterations: 4,
      usePreconditionTrick: false,
      earlyStopping: { minError: 1e-15 },
    });
    expect(Number.isFinite(xBest.get(0, 0))).toBeTruthy();
    expect(mseMin).not.toBeNaN();
    expect(iterations).toBeLessThanOrEqual(maxIterations);
    expect(mse.length).toBeLessThanOrEqual(maxIterations + 1);
  }
});

test('Many runs without error, use pseudo inverse', () => {
  for (let i = 0; i < 1e2; i++) {
    const m = Math.ceil(Math.random() * 12) + 2;
    const n = Math.ceil(Math.random() * 12) + 2;
    const { inputs: A, outputs: b } = makeData(m, n);
    const randomRowVector = Matrix.random(1, n).multiply(100);
    const randomColumnVector = Matrix.random(m, 1).multiply(35);
    const bigA = A.mulRowVector(randomRowVector);
    const bigB = b.mulColumnVector(randomColumnVector);
    const tnt = new TNT(bigA, bigB, {
      maxIterations: 4,
      earlyStopping: { minError: 1e-6 },
    });
    expect(Number.isFinite(tnt.xBest.get(0, 0))).toBeTruthy();
    expect(tnt.mseMin).not.toBeNaN();
    expect(tnt.iterations).toBeLessThanOrEqual(tnt.maxIterations);
    expect(tnt.mse.length).toBeLessThanOrEqual(tnt.maxIterations + 1);
  }
});

test('example in the readme', () => {
  const A = new Matrix([
    [1, 2, 3],
    [4, 5, 6],
  ]); // 2x3
  const b = [6, 12];
  const b2 = [[6], [12]];
  const opts: Partial<TNTOpts> = {
    maxIterations: 4,
    unacceptableError: 1e-2,
    earlyStopping: { minError: 1e-8 },
  };
  const r = new TNT(A, b, opts);
  expect(r.mseMin).toBeLessThanOrEqual(0.02);
  const r2 = new TNT(A, b2, opts);
  expect(r2.mseMin).toBeLessThanOrEqual(0.02);
});

test('Ill Conditioned', () => {
  // has large condition number.
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
  expect(new TNT(illConditioned, b)).toBeDefined();
});

test('Another Test', () => {
  expect(
    new TNT(Matrix.ones(5, 500), Matrix.ones(5, 1), {
      pseudoInverseFallback: false,
    }),
  ).toBeDefined();
});
