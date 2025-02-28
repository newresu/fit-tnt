import { Matrix } from 'ml-matrix';
import { describe, expect, it } from 'vitest';

import { PreconditionError } from '../Errors';
import { choleskyPreconditionTrick } from '../choPrecondition';

describe('choleskyPreconditionTrick', () => {
  it('should return a Cholesky Decomposition for a positive definite matrix', () => {
    const matrix = new Matrix([
      [4, 1],

      [1, 3],
    ]);

    const cholesky = choleskyPreconditionTrick(matrix);

    expect(cholesky.isPositiveDefinite()).toBe(true);
  });

  it('should throw PreconditionError for a non-positive definite matrix', () => {
    const matrix = new Matrix([
      [0, 1],

      [1, 0],
    ]);

    expect(() => choleskyPreconditionTrick(matrix)).toThrow(PreconditionError);
  });

  it('should mutate the matrix to make it positive definite', () => {
    const matrix = new Matrix([
      [1, 2],

      [2, 1],
    ]);

    expect(() => choleskyPreconditionTrick(matrix)).toThrow(PreconditionError);
  });
});
