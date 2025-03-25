import { Matrix } from 'ml-matrix';
import { describe, expect, it, vi } from 'vitest';

import { checkMatchingDimensions } from '../initSafetyChecks';

const m = 3;
const n = 4;
const p = 2;

describe('checkMatchingDimensions', () => {
  it('should return undefined', () => {
    const A = Matrix.random(m, n);
    const B = Matrix.random(m, p);
    const X = Matrix.zeros(n, p);

    const fn = vi.fn().mockImplementation(checkMatchingDimensions);

    expect(fn(A, X, B)).toBeUndefined();
  });

  it('should throw an error fot A, X', () => {
    const A = Matrix.random(m, n);
    const B = Matrix.random(m, p);
    const X = Matrix.zeros(n - 1, p);
    expect(() => {
      checkMatchingDimensions(A, X, B);
    }).toThrow();
  });
  it('should throw an error fot A, X', () => {
    const A = Matrix.random(m, n);
    const B = Matrix.random(m, p - 1);
    const X = Matrix.zeros(n, p);
    expect(() => {
      checkMatchingDimensions(A, X, B);
    }).toThrow();
  });

  it('should throw an error fot A, B', () => {
    const A = Matrix.random(m, n);
    const B = Matrix.random(m - 1, p);
    const X = Matrix.zeros(n, p);
    expect(() => {
      checkMatchingDimensions(A, X, B);
    }).toThrow();
  });
});
