import { makeData } from './makeData';
import { TNT } from '../tnt';
import { expect, test } from 'vitest';
// import { pseudoInverse } from 'ml-matrix';

test('Many runs without error', () => {
  for (let i = 0; i < 1e5; i++) {
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
    expect(tnt.mse.length).toBeLessThanOrEqual(3);
  }
});
