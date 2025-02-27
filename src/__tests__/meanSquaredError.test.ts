import { Matrix } from 'ml-matrix';
import { expect, test } from 'vitest';

import { meanSquaredError } from '../meanSquaredError';

test('Returns the mean squared error', () => {
    const A = new Matrix([
        [1, 2],
        [3, 4],
    ]);
    const x = new Matrix([[1], [2]]);
    const b = new Matrix([[5], [11]]);
    expect(meanSquaredError(A, x, b)).toBeCloseTo(0);
});
