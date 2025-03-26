import { Matrix } from 'ml-matrix';
import { describe, expect, it } from 'vitest';

import { PreconditionError } from '../Errors';
import { choleskyPrecondition } from '../choPrecondition';

describe('choleskyPrecondition', () => {
  it('should return a Cholesky Decomposition for a positive definite matrix', () => {
    const matrix = new Matrix([
      [4, 1],

      [1, 3],
    ]);

    const cholesky = choleskyPrecondition(matrix);

    expect(cholesky.isPositiveDefinite()).toBe(true);
  });

  it('Can not improve this 3x3 matrix', () => {
    const matrix = new Matrix([
      [1, 4, 5],
      [4, 2, 6],
      [5, 6, 3],
    ]);
    expect(() => choleskyPrecondition(matrix)).toThrow();
  });
  it('Can not improve this binary 2x2 matrix.', () => {
    const matrix = new Matrix([
      [0, 1],

      [1, 0],
    ]);

    expect(() => choleskyPrecondition(matrix)).toThrow(PreconditionError);
  });

  it('Can not improve this 2x2 matrix.', () => {
    const matrix = new Matrix([
      [1, 2],
      [2, 1],
    ]);
    expect(() => choleskyPrecondition(matrix)).toThrow(PreconditionError);
  });
});
