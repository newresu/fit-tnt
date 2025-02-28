import { expect, test } from 'vitest';

import { TNT } from '../tnt';
import { makeData } from './makeData';

test('Many random matrices between 0 and 1', () => {
  for (let i = 0; i < 1e2; i++) {
    const m = Math.ceil(Math.random() * 12) + 2;
    const n = Math.ceil(Math.random() * 12) + 2;
    const { inputs: A, outputs: b } = makeData(m, n);
    const { xBest, mseMin } = new TNT(A, b, {
      maxIterations: 4,
      usePreconditionTrick: false,
      earlyStopping: { minMSE: 1e-15 },
    });
    expect(Number.isFinite(xBest.get(0, 0))).toBeTruthy();
    expect(mseMin).not.toBeNaN();
  }
});

test('Scaled Up A', () => {
  for (let i = 0; i < 1e2; i++) {
    const m = Math.ceil(Math.random() * 12) + 2;
    const n = Math.ceil(Math.random() * 12) + 2;
    const { inputs: bigA, outputs: b } = makeData(m, n, { scaleA: 100 });
    const { mse, mseMin, iterations, maxIterations, xBest } = new TNT(bigA, b, {
      maxIterations: 8,
      earlyStopping: { minMSE: 1e-3 },
    });
    expect(Number.isFinite(xBest.get(0, 0))).toBeTruthy();
    expect(mseMin).not.toBeNaN();
    expect(iterations).toBeLessThanOrEqual(maxIterations);
    expect(mse.length).toBeLessThanOrEqual(maxIterations + 2);
  }
});
test('Scaled Up X runs without error', () => {
  for (let i = 0; i < 1e2; i++) {
    const m = Math.ceil(Math.random() * 12) + 2;
    const n = Math.ceil(Math.random() * 12) + 2;
    const { inputs: A, outputs: bigB } = makeData(m, n, {
      scaleX: 1,
      addNoise: true,
    });
    const { mse, mseMin, iterations, maxIterations, xBest } = new TNT(A, bigB, {
      maxIterations: 4,
      earlyStopping: { minMSE: 1e-3 },
    });
    expect(Number.isFinite(xBest.get(0, 0))).toBeTruthy();
    expect(mseMin).not.toBeNaN();
    expect(iterations).toBeLessThanOrEqual(maxIterations);
    expect(mse.length).toBeLessThanOrEqual(maxIterations + 2);
  }
});
