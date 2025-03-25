import { Matrix } from 'ml-matrix';
import { describe, expect, it } from 'vitest';

import { squaredSum } from '../squaredSum';

describe('squaredSum', () => {
  it('should calculate squared sum by column', () => {
    const matrix = new Matrix([
      [1, 2],
      [3, 4],
    ]);
    const result = squaredSum(matrix, { by: 'column' });
    expect(result).toEqual([10, 20]);
  });

  it('should calculate squared sum by row', () => {
    const matrix = new Matrix([
      [1, 2],
      [3, 4],
    ]);
    const result = squaredSum(matrix, { by: 'row' });
    expect(result).toEqual([5, 25]);
  });

  it('should calculate squared sum for column vector', () => {
    const matrix = new Matrix([[1], [2], [3]]);
    const result = squaredSum(matrix);
    expect(result).toEqual([14]);
  });

  it('should calculate squared sum for row vector', () => {
    const matrix = new Matrix([[1, 2, 3]]);
    const result = squaredSum(matrix);
    expect(result).toEqual([14]);
  });

  it('should use default option (by column) when no option is provided', () => {
    const matrix = new Matrix([
      [1, 2],
      [3, 4],
    ]);
    const result = squaredSum(matrix);
    expect(result).toEqual([10, 20]);
  });
});
