import { makeData } from './makeData';
import { TNT } from '../tnt';
import { expect, test } from 'vitest';
import { Matrix } from 'ml-matrix';

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
test('Many runs without error', () => {
  for (let i = 0; i < 1e4; i++) {
    const m = Math.ceil(Math.random() * 12) + 2;
    const n = Math.ceil(Math.random() * 12) + 2;
    const { inputs: A, outputs: b } = makeData(m, n);
    const tnt = new TNT(A, b, {
      pseudoInverseFallback: true,
      maxIterations: 2,
      earlyStopping: { patience: 2, minError: 1e-8 },
    });
    expect(Number.isFinite(tnt.xBest.get(0, 0))).toBeTruthy();
    expect(tnt.mseMin).toBeLessThanOrEqual(tnt.mseLast);
    expect(tnt.iterations).toBeLessThanOrEqual(tnt.maxIterations + 1); //should be equal, but is +1 when fallbacks to pseudoInverse.
    expect(tnt.mse.length).toBeLessThanOrEqual(tnt.maxIterations + 2); // same
  }
});

test('fails to optimize enough without PseudoInverse', () => {
  expect(() => new TNT(illConditioned, b)).toThrowError();
});

test('optimizes with Pseudo Inverse', () => {
  const result = new TNT(illConditioned, b, {
    pseudoInverseFallback: true,
  });
  expect(Number.isFinite(result.xBest.get(0, 0))).toBeTruthy();
  expect(result.mseMin).toBeLessThanOrEqual(result.mseLast);
  expect(result.mseMin).toBeLessThanOrEqual(1e-2);
  expect(result.method).toBe('pseudoInverse');
});
