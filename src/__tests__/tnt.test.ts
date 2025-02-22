import { makeData } from './makeData';
import { tnt } from '../tnt';
import { expect, test } from 'vitest';
import { pseudoInverse } from 'ml-matrix';

test('Many runs without error', () => {
  for (let i = 0; i < 1; i++) {
    const { inputs: A, outputs: b } = makeData(1000, 1000);
    // console.log(pseudoInverse(A).mmul(b));
    const { mse, solution } = tnt(A, b, { maxIterations: 8, tolerance: 1e-10 });
    // console.log(solution, mse);
    // expect(Number.isFinite(solution[0])).toBeTruthy();
    // if (Math.max(...mse) > 1e-4) {
    //   console.log(mse);
    // }
  }
});
