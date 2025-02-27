import { Matrix } from 'ml-matrix';
import { expect, test } from 'vitest';

import { TNT } from '../tnt';
import { TNTOpts } from '../types';

test('example in the readme through both methods', () => {
  const A = new Matrix([
    [1, 2, 3],
    [4, 5, 6],
  ]); // 2x3
  const b = [6, 12];
  const b2 = [[6], [12]];
  const opts: Partial<TNTOpts> = {
    maxIterations: 4,
    maxAllowedMSE: 1,
    earlyStopping: { minError: 1e-8 },
  };
  let r = new TNT(A, b, opts);
  expect(r.mseMin).toBeLessThanOrEqual(0.02);
  expect(r.method).toBe('TNT');

  let r2 = new TNT(A, b2, opts);
  expect(r2.mseMin).toBeLessThanOrEqual(0.02);
  expect(r2.method).toBe('TNT');

  // this forces method 2
  opts.maxIterations = 0;
  r = new TNT(A, b, opts);
  expect(r.mseMin).toBeLessThanOrEqual(0.02);
  expect(r.method).toBe('pseudoInverse');

  r2 = new TNT(A, b2, opts);
  expect(r2.mseMin).toBeLessThanOrEqual(0.02);
  expect(r2.method).toBe('pseudoInverse');
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
  const r = new TNT(illConditioned, b, { maxAllowedMSE: 0.01 });
  expect(r).toBeDefined();
  expect(r.method).toBe('TNT');

  const r2 = new TNT(illConditioned, b, {
    usePreconditionTrick: false,
    pseudoInverseFallback: true,
    maxAllowedMSE: 0.01,
  });
  expect(r2).toBeDefined();
  expect(r2.method).toBe('pseudoInverse');

  expect(
    () =>
      new TNT(illConditioned, b, {
        usePreconditionTrick: false,
        pseudoInverseFallback: false,
      }),
  ).toThrowError();
});

test('Another Test', () => {
  const result = new TNT(Matrix.ones(5, 500), Matrix.ones(5, 1), {
    pseudoInverseFallback: false,
    usePreconditionTrick: false,
  });
  expect(result).toBeDefined();
  expect(result.method).toBe('TNT');
  expect(result.mseMin).toBeLessThanOrEqual(1e-2);
});
