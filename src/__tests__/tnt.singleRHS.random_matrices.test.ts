import { describe, expect, it } from 'vitest';

import { TNT } from '../tnt';
import { makeData } from './makeData';

describe('Single RHS, random values', () => {
  it('Values between 0 and 1', () => {
    for (let i = 0; i < 1e2; i++) {
      const m = Math.ceil(Math.random() * 12) + 2;
      const n = Math.ceil(Math.random() * 12) + 2;
      const { inputs: A, outputs: b } = makeData(m, n);
      const { XBest: xBest, metadata: e } = new TNT(A, b, {
        earlyStopping: { minMSE: 1e-6 },
      });
      expect(e[0].mseMin).toBeLessThan(1e-3);
      expect(xBest.to1DArray().every(Number.isFinite)).toBeTruthy();
    }
  });

  it('Scaled Up A', () => {
    for (let i = 0; i < 1e2; i++) {
      const m = Math.ceil(Math.random() * 12) + 2;
      const n = Math.ceil(Math.random() * 12) + 2;
      const { inputs: bigA, outputs: b } = makeData(m, n, { scaleA: 100 });
      const {
        metadata,
        maxIterations,
        XBest: xBest,
      } = new TNT(bigA, b, {
        maxIterations: 5,
        earlyStopping: { minMSE: 1e-3 },
      });
      expect(xBest.to1DArray().every(Number.isFinite)).toBeTruthy();
      const { mse, mseMin, iterations } = metadata[0];

      expect(mseMin).toBeLessThan(1e-3);
      expect(iterations).toBeLessThanOrEqual(maxIterations);
      expect(mse.length).toBeLessThanOrEqual(maxIterations + 1);
    }
  });
  it('Scaled Up X (on AX=>B) to make large B', () => {
    for (let i = 0; i < 1e2; i++) {
      const m = Math.ceil(Math.random() * 12) + 2;
      const n = Math.ceil(Math.random() * 12) + 2;
      const { inputs: A, outputs: bigB } = makeData(m, n, {
        scaleX: 1,
        addNoise: true,
      });
      const {
        metadata,
        maxIterations,
        XBest: xBest,
      } = new TNT(A, bigB, {
        maxIterations: 4,
        earlyStopping: { minMSE: 1e-3 },
      });

      expect(xBest.to1DArray().every(Number.isFinite)).toBeTruthy();

      const { mse, mseMin, iterations } = metadata[0];
      expect(mseMin).toBeLessThan(1e-3);
      expect(iterations).toBeLessThanOrEqual(maxIterations);
      expect(mse.length).toBeLessThanOrEqual(maxIterations + 1);
    }
  });
});
