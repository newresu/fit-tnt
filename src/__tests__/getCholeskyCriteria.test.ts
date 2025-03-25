import { describe, expect, it } from 'vitest';

import { getCriteria } from '../getCholeskyCriteria';

describe('getCriteria', () => {
  it('should throw an error if the array is empty', () => {
    expect(() => getCriteria([], 2)).toThrow('Array cannot be empty');
  });

  it('should calculate eps and ratio correctly for a given array and power', () => {
    const arr = [1, 2, 3, 4, 5];
    const power = 2;
    const result = getCriteria(arr, power);
    expect(result.eps).toBeCloseTo(300, 5); // average (3) * 10^2
    expect(result.ratio).toBeCloseTo(1 / 3, 5); // min (1) / average (3)
  });

  it('should handle arrays with non-finite numbers correctly', () => {
    const arr = [1, 2, Infinity, 4, NaN];
    const power = 2;
    const result = getCriteria(arr, power);
    expect(result.eps).toBeCloseTo((7 / 3) * 100, 5);
    expect(result.ratio).toBeCloseTo(1 / (7 / 3), 5);
  });
});
