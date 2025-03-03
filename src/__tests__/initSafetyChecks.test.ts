import { Matrix } from 'ml-matrix';
import { describe, expect, it } from 'vitest';

import { initSafetyChecks } from '../initSafetyChecks';

const m = 3; // "data points"
const n = 4; // "coefficients per project"
const p = 2; // "projects"

const A = Matrix.random(m, n);
const X = Matrix.random(n, p);
const B = Matrix.random(m, p);

describe('initSafetyChecks', () => {
  it('Correct Dimensions sends Undefined', () => {
    expect(initSafetyChecks(A, X, B)).toBeUndefined();
  });

  it('A and B: m must match', () => {
    const B = Matrix.random(4, 1);
    expect(() => initSafetyChecks(A, X, B)).toThrowError(
      'Found A rows = 3 and B rows = 4. They must match.',
    );
  });

  it('A and X: n must match', () => {
    const X = Matrix.random(5, 2);
    expect(() => initSafetyChecks(A, X, B)).toThrowError(
      'Found A columns = 4 and X rows = 5. They must match.',
    );
  });

  it('B and X: p must match', () => {
    const B = Matrix.random(3, 1);
    expect(() => initSafetyChecks(A, X, B)).toThrowError(
      'Found B columns = 1 and X columns = 2. They must match.',
    );
  });
});
