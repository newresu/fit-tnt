import { Matrix } from 'ml-matrix';
import { describe, expect, it } from 'vitest';

import { meanSquaredError } from '../squaredSum';

describe('meanSquaredError', () => {
  const A = new Matrix([
    [1, 2],
    [3, 4],
  ]);
  it('Returns the mean squared error single RHS', () => {
    const X = new Matrix([[1], [2]]);
    const B = new Matrix([[5], [11]]);
    expect(meanSquaredError(A, B, X)[0]).toBeCloseTo(0, 10);
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
    expect(meanSquaredError(A, B, X)[0]).toBeCloseTo(0, 10);
    expect(meanSquaredError(A, B, X)[1]).toBeCloseTo(10, 10);
  });
});
