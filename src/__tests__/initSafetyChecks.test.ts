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

  it('Should throw an error for A, X mismatch', () => {
    const A = Matrix.random(m, n);
    const B = Matrix.random(m, p);
    const X = Matrix.zeros(n - 1, p);
    expect(() => {
      checkMatchingDimensions(A, X, B);
    }).toThrow();
  });
  it('Should throw an error for B,X p mismatch.', () => {
    const A = Matrix.random(m, n);
    const B = Matrix.random(m, p - 1);
    const X = Matrix.zeros(n, p);
    expect(() => {
      checkMatchingDimensions(A, X, B);
    }).toThrow();
  });

  it('Should throw an error for A,B n mismatch.', () => {
    const A = Matrix.random(m, n);
    const B = Matrix.random(m - 1, p);
    const X = Matrix.zeros(n, p);
    expect(() => {
      checkMatchingDimensions(A, X, B);
    }).toThrow();
  });
});
