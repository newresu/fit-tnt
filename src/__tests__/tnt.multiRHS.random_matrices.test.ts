import { describe, expect, it } from 'vitest';

import { TNT } from '../tnt';
import { makeData } from './makeData';

describe('Multi RHS, random values', () => {
  it('Values between 0 and 1', () => {
    for (let i = 0; i < 1e2; i++) {
      const m = Math.ceil(Math.random() * 12) + 2;
      const n = Math.ceil(Math.random() * 12) + 2;
      const { inputs: A, outputs: b } = makeData(m, n, { outputColumns: n });
      const { XBest, metadata: e } = new TNT(A, b, {
        maxIterations: 4,
        earlyStopping: { minMSE: 1e-6 },
      });
      expect(e[0].mseMin).toBeLessThan(1e-4);
      expect(XBest.to1DArray().every(Number.isFinite)).toBeTruthy();
    }
  });

  it('Scaled Up A', () => {
    for (let i = 0; i < 1e2; i++) {
      const m = Math.ceil(Math.random() * 12) + 2;
      const n = Math.ceil(Math.random() * 12) + 2;
      const { inputs: A, outputs: B } = makeData(m, n, {
        scaleA: 100,
        outputColumns: m,
      });
      const { metadata, maxIterations, XBest } = new TNT(A, B, {
        maxIterations: 15,
        earlyStopping: { minMSE: 1e-3 },
      });
      expect(XBest.to1DArray().every(Number.isFinite)).toBeTruthy();
      const { mse, mseMin, iterations } = metadata[0];
      expect(mseMin).toBeLessThan(1e-4);
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
        outputColumns: m,
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
      expect(mseMin).toBeLessThan(1e-4);
      expect(iterations).toBeLessThanOrEqual(maxIterations);
      expect(mse.length).toBeLessThanOrEqual(maxIterations + 1);
    }
  });
});
