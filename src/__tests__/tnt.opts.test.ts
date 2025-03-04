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
      earlyStopping: { minMSE: 1e-8 },
    };
    const r = new TNT(A, b, opts);
    expect(r.mseMin).toBeLessThan(0.02);

    const r2 = new TNT(A, b2, opts);
    expect(r2.mseMin).toEqual(r.mseMin);
  });

  test('Should have large error without iterations.', () => {
    const opts: Partial<TNTOpts> = {
      maxIterations: 0,
      earlyStopping: { minMSE: 1e-8 },
    };
    // this forces method 2
    const r = new TNT(A, b, opts);
    expect(r.mseMin).toBeGreaterThan(179);
  });
});
