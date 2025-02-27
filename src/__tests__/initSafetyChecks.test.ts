import { Matrix } from 'ml-matrix';
import { expect, test } from 'vitest';

import { initSafetyChecks } from '../initSafetyChecks';

const A = Matrix.random(3, 4);
const b = Matrix.random(3, 1);

test('Correct Dimensions sends Undefined', () => {
  expect(initSafetyChecks(A, b)).toBeUndefined();
});

test('A and b: Row dimension must match', () => {
  const b = Matrix.random(4, 1);
  expect(() => initSafetyChecks(A, b)).toThrowError();
});

test('B Column Vector', () => {
  const b = A;
  expect(() => initSafetyChecks(A, b)).toThrowError();
});
