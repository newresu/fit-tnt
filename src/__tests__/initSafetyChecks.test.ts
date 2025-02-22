import { initSafetyChecks } from '../initSafetyChecks';
import { Matrix } from 'ml-matrix';
import { expect, test } from 'vitest';

const A = Matrix.random(3, 4);
const AtA = A.mmul(A.transpose());
const b = Matrix.random(3, 1);

test('Correct Dimensions sends Undefined', () => {
  expect(initSafetyChecks(AtA, A, b)).toBeUndefined();
});

test('A and b: Row dimension must match', () => {
  const b = Matrix.random(4, 1);
  expect(() => initSafetyChecks(AtA, A, b)).toThrowError();
});

test('B Column Vector', () => {
  const b = A
  expect(() => initSafetyChecks(AtA, A, b)).toThrowError();
});
test('Not Symmetric', () => {
  const AtA = Matrix.random(4, 1);
  expect(() => initSafetyChecks(AtA, A, b)).toThrowError();
});
