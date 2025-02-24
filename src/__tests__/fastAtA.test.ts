import { expect, test } from 'vitest';
import { Matrix } from 'ml-matrix';

import { fastAtA } from '../fastAtA';

test('Returns the fast AtA matrix', () => {
  const At = new Matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
  ]);
  const AtA = fastAtA(At);
  expect(AtA.isSymmetric()).toBe(true);
});
