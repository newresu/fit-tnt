import { makeData } from './makeData';
import { _tnt } from '../tnt';
import { expect, test } from 'vitest';
// import { pseudoInverse } from 'ml-matrix';

test('Many runs without error', () => {
  for (let i = 0; i < 1e5; i++) {
    const { inputs: A, outputs: b } = makeData(5, 4);
    // console.log(pseudoInverse(A).mmul(b));
    const solution = _tnt(A, b, [], { maxIterations: 8, tolerance: 1e-20 });
    // console.log(solution, mse);
    expect(Number.isFinite(solution.get(0, 0))).toBeTruthy();
    // if (Math.max(...mse) > 1e-4) {
    //   console.log(mse);
    // }
  }
});
