import { makeData } from './makeData';
import { TNT } from '../tnt';
import { expect, test } from 'vitest';
// import { pseudoInverse } from 'ml-matrix';

test('Many runs without error', () => {
  for (let i = 0; i < 1e6; i++) {
    const m = Math.ceil(Math.random() * 12) + 2;
    const n = Math.ceil(Math.random() * 12) + 2;
    const { inputs: A, outputs: b } = makeData(m, n);
    // console.log(pseudoInverse(A).mmul(b));
    const tnt = new TNT(A, b, {
      maxIterations: 4,
      earlyStopping: { patience: 3, minError: 1e-8 },
    });
    const result = tnt.solve();
    // console.log(solution, mse);
    expect(Number.isFinite(result.get(0, 0))).toBeTruthy();
    // console.log(mse);
  }
});
