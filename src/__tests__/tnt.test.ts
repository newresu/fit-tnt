import { Matrix } from 'ml-matrix';
import { expect, test } from 'vitest';

import { TNT } from '../tnt';

test('Ill Conditioned', () => {
  // has large condition number.
  const illConditioned = new Matrix([
    [
      0.06355853189791905, 0.7799679257920251, 0.0019664575485265345,
      0.49785241128125124, 0.9488955201112512,
    ],
    [
      0.06819327032425537, 0.037153233827454946, 0.9247986023698862,
      0.705334939844535, 0.13307672470945064,
    ],
    [
      0.13026353337270136, 0.24163034491879132, 0.9227731156740526,
      0.2830279588620952, 0.0012315083853995379,
    ],
    [
      0.9254405073838763, 0.9132081295563979, 0.29893902393620997,
      0.27094620118832036, 0.06554637642053063,
    ],
  ]);
  const b = Matrix.ones(illConditioned.rows, 1);
  const r = new TNT(illConditioned, b);
  expect(r).toBeDefined();

  const r2 = new TNT(illConditioned, b, {
    maxIterations: 0,
  });
  expect(r2).toBeDefined();
});

test('Another Test', () => {
  const result = new TNT(Matrix.ones(5, 500), Matrix.ones(5, 1));
  expect(result).toBeDefined();
  expect(result.mseMin).toBeLessThanOrEqual(1e-2);
});
