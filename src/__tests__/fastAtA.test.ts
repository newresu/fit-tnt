import { Matrix } from 'ml-matrix';
import { expect, test } from 'vitest';

import { fastAtA } from '../fastAtA';

test('Returns the AtA matrix', () => {
  const At = new Matrix([
    [1, 2, 3, 3],
    [4, 5, 6, 6],
    [7, 8, 9, 7],
  ]);
  const AtA = fastAtA(At);
  expect(AtA.isSymmetric()).toBe(true);
  expect(AtA.get(0, 0)).toEqual(23);
  expect(AtA.get(0, 1)).toEqual(50);
  expect(AtA.get(2, 2)).toEqual(243);
});
