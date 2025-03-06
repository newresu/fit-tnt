import { Matrix } from 'ml-matrix';
import { describe, expect, it } from 'vitest';

import { meanSquaredError } from '../meanSquaredError';

describe('meanSquaredError', () => {
  const A = new Matrix([
    [1, 2],
    [3, 4],
  ]);
  it('Returns the mean squared error single RHS', () => {
    const x = new Matrix([[1], [2]]);
    const b = new Matrix([[5], [11]]);
    expect(meanSquaredError(A, x, b)[0]).toBeCloseTo(0, 10);
  });
  it('Returns the mean squared error multi RHS', () => {
    const X = new Matrix([
      [1, 4],
      [2, 4],
    ]);
    const B = new Matrix([
      [5, 14],
      [11, 32],
    ]);
    expect(meanSquaredError(A, X, B)[0]).toBeCloseTo(0, 10);
    expect(meanSquaredError(A, X, B)[1]).toBeCloseTo(10, 10);
  });
});
