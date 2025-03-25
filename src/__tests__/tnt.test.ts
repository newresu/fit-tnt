import { Matrix, pseudoInverse } from 'ml-matrix';
import { describe, expect, it } from 'vitest';

import { meanSquaredError } from '../meanSquaredError';
import { TNT } from '../tnt';

describe('Test some ill conditioned matrices', () => {
  it('Ill Conditioned', () => {
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

  it('Another Test', () => {
    const result = new TNT(Matrix.ones(5, 500), Matrix.ones(5, 1));
    expect(result).toBeDefined();
    expect(result.metadata[0].mseMin).toBeLessThanOrEqual(5e-31);
  });

  it('hard matrix multi-RHS', () => {
    const A = new Matrix([
      [-90.2419, -84.2534, -12.7045, -46.9038, -13.755, 95.1238, -61.5441],
      [82.8602, -47.9659, 6.05526, -80.739, 99.4879, -65.3956, -22.0943],
      [-22.5724, 17.741, -85.0359, -71.1472, -7.13554, -58.379, 63.7102],
      [13.4092, 10.267, 25.2449, -88.8331, 32.4427, 87.0503, 54.2427],
      [-5.82175, 71.7253, -34.1971, 94.1174, 88.9854, 94.2196, 7.58638],
      [82.8733, 59.1238, 97.5522, 9.94099, -40.2938, -18.3154, -22.6148],
    ]);

    const B = new Matrix([
      [-121.217, -50.0461, -112.467, -198.172, -66.483, -89.1029],
      [-30.8537, -104.756, -103.17, -55.9705, -25.6904, 10.3326],
      [-136.922, -69.2575, -88.9156, -121.656, -148.303, -26.298],
      [107.924, 135.331, 46.6062, 55.9771, 103.079, 63.6329],
      [232.601, 145.752, 156.415, 182.478, 177.856, 168.421],
      [114.676, 101.811, 82.1189, 177.616, 114.161, 8.06781],
    ]);

    // tnt
    const r = new TNT(A, B);
    const { mseMin } = r.metadata[0];
    expect(mseMin).toBeLessThan(5e-11);

    // pseudoInverse
    const X = pseudoInverse(A).mmul(B);
    const minErrorPseudoInverse = meanSquaredError(A, B, X);
    expect(minErrorPseudoInverse[0]).toBeLessThan(1e-20);
  });
});
