import { Matrix } from 'ml-matrix';
import { describe, expect, it } from 'vitest';

import { initSafetyChecks } from '../initSafetyChecks';

const m = 3;
const n = 4;
const p = 2;

describe('initSafetyChecks', () => {
  it('should return undefined', () => {
    const A = Matrix.random(m, n);
    const B = Matrix.random(m, p);
    const X = Matrix.zeros(n, p);
    const metadata = initSafetyChecks(A, X, B);

    expect(metadata).toBeUndefined();
  });

  it('should throw an error fot A, X', () => {
    const A = Matrix.random(m, n);
    const B = Matrix.random(m, p);
    const X = Matrix.zeros(n - 1, p);
    expect(() => initSafetyChecks(A, X, B)).toThrow();
  });

  it('should throw an error fot A, B', () => {
    const A = Matrix.random(m, n);
    const B = Matrix.random(m - 1, p);
    const X = Matrix.zeros(n, p);
    expect(() => initSafetyChecks(A, X, B)).toThrow();
  });
});
