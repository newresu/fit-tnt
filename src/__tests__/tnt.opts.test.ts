import { Matrix } from 'ml-matrix';
import { describe, expect, test } from 'vitest';

import { TNT } from '../index';
import { TNTOpts } from '../types';

const A = new Matrix([
  [1, 2, 3],
  [4, 5, 6],
]); // 2x3
const b = [6, 12];
const b2 = [[6], [12]];

describe('Test TNT Options', () => {
  test('Row and Column inputs return the same.', () => {
    const opts: Partial<TNTOpts> = {
      maxIterations: 4,
      maxAllowedMSE: 0.02,
      earlyStopping: { minMSE: 1e-8 },
      pseudoInverseFallback: true,
    };
    const r = new TNT(A, b, opts);
    expect(r.mseMin).toBeLessThanOrEqual(0.02);
    expect(r.method).toBe('TNT');

    const r2 = new TNT(A, b2, opts);
    expect(r2.mseMin).toEqual(r.mseMin);
    expect(r2.method).toBe(r.method);
  });

  test('Fallback to pseudo inverse when mse is too high.', () => {
    const opts: Partial<TNTOpts> = {
      maxIterations: 0,
      maxAllowedMSE: 0.02,
      earlyStopping: { minMSE: 1e-8 },
    };
    // this forces method 2
    const r = new TNT(A, b, opts);
    expect(r.mseMin).toBeLessThanOrEqual(0.02);
    expect(r.method).toBe('pseudoInverse');
  });
  test('Error when mse is too high and pseudo-inverse is false', () => {
    const opts: Partial<TNTOpts> = {
      maxIterations: 0,
      maxAllowedMSE: 0.0001,
      earlyStopping: { minMSE: 1e-8 },
      pseudoInverseFallback: false,
    };
    expect(() => new TNT(A, b, opts)).toThrowError();
  });
  test('PseudoInverse when ratio is too high', () => {
    const opts: Partial<TNTOpts> = {
      maxIterations: 0,
      maxAllowedMSE: 0.0001,
      earlyStopping: { minMSE: 1e-8 },
      pseudoInverseFallback: true,
    };
    const r = new TNT(Matrix.random(2, 25), b, opts);
    expect(r.method).toBe('pseudoInverse');
  });
});
