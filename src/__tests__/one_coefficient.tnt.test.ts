import { Matrix } from 'ml-matrix';
import { expect, test } from 'vitest';

import { TNT } from '../tnt';
import { TNTOpts } from '../types';

test('Simple Linear Fit from non-noisy data', () => {
  const A = new Matrix([
    //col of data
    [0.9242517859627595],
    [0.0818550256160202],
    [-0.624983105290315],
    [-0.82626649326827],
    [-0.7675865514669384],
    [-0.34794526363377654],
    [-0.6488463103599258],
    [0.6235196593971422],
    [0.9077898598099983],
    [0.34626125969154886],
  ]);
  const b = [
    //col of results
    0.23773114859044794, 0.02105431609994324, -0.16075484378482774,
    -0.21252789063518465, -0.19743454683480463, -0.08949663763332648,
    -0.16689281098857903, 0.16037842398411126, 0.23349689914484856,
    0.08906348705953603,
  ];
  const { xBest, mseMin, iterations, method, mse, maxIterations } = new TNT(
    A,
    b,
    {
      maxIterations: 4,
      usePreconditionTrick: true,
      earlyStopping: { minError: 1e-15 },
    },
  );
  expect(Number.isFinite(xBest.get(0, 0))).toBeTruthy();
  expect(mseMin).not.toBeNaN();
  expect(iterations).toBeLessThanOrEqual(maxIterations);
  expect(mse.length).toBeLessThanOrEqual(iterations + 1);
  expect(mseMin).toBeCloseTo(0, 10);
  expect(method).toBe('TNT');
  expect(xBest.get(0, 0)).toBeCloseTo(0.257214702964);
});

test('Simple Linear Fit to noisy data', () => {
  const A = new Matrix([
    [-0.008284110337955319],
    [0.5897720744120512],
    [0.15217826587090927],
    [0.25978149066548833],
    [-0.2987909107335514],
    [-0.5341164458709763],
    [0.5196655664209802],
    [-0.9114099314910604],
    [-0.38975386686619523],
    [0.5684580385973504],
  ]);
  const b = [
    -0.004230803938062248, 0.2533694661111869, 0.06908929893243614,
    0.11442237305348127, -0.12912524549758192, -0.23263959668058015,
    0.22835097473417648, -0.3926549809396137, -0.16958217669996367,
    0.24359542640624454,
  ];
  const x = 0.4350441345216933;
  const opts: Partial<TNTOpts> = {
    maxIterations: 4,
    usePreconditionTrick: true,
    earlyStopping: { minError: 1e-15 },
    maxAllowedMSE: 0.00001,
  };
  const { xBest, mseMin, mse, iterations, maxIterations, method } = new TNT(
    A,
    b,
    opts,
  );
  expect(Number.isFinite(xBest.get(0, 0))).toBeTruthy();
  expect(mseMin).not.toBeNaN();
  expect(iterations).toBeLessThanOrEqual(maxIterations);
  expect(mse.length).toBeLessThanOrEqual(iterations + 1);
  expect(xBest.get(0, 0)).toBeCloseTo(x, 2);
  expect(method).toBe('TNT');

  opts.maxAllowedMSE = 0.000000000000001;
  expect(() => new TNT(A, b, opts)).toThrowError('Minimum MSE');

  opts.maxIterations = 0;
  opts.maxAllowedMSE = 0.0001;
  const resultInverse = new TNT(A, b, opts);
  expect(resultInverse.method).toBe('pseudoInverse');
  expect(resultInverse.xBest.get(0, 0)).toBeCloseTo(x, 2);
});
