import { Matrix } from 'ml-matrix';
import { describe, expect, it } from 'vitest';

import { TNT } from '../tnt';
import { TNTOpts } from '../types';

describe('TNT test 1 coefficient', () => {
  it('non-noisy data', () => {
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
    const B = [
      //col of results
      0.23773114859044794, 0.02105431609994324, -0.16075484378482774,
      -0.21252789063518465, -0.19743454683480463, -0.08949663763332648,
      -0.16689281098857903, 0.16037842398411126, 0.23349689914484856,
      0.08906348705953603,
    ];
    const { XBest, metadata, maxIterations } = new TNT(A, B);
    const e = metadata[0];
    expect(e.mseMin).not.toBeNaN();
    expect(e.iterations).toBeLessThanOrEqual(maxIterations);
    expect(e.mse.length).toBeLessThanOrEqual(e.iterations + 1);
    expect(e.mseMin).toBeCloseTo(0, 10);
    expect(XBest.get(0, 0)).toBeCloseTo(0.257214702964);
  });

  it('Simple Linear Fit to noisy data', () => {
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
    const B = [
      -0.004230803938062248, 0.2533694661111869, 0.06908929893243614,
      0.11442237305348127, -0.12912524549758192, -0.23263959668058015,
      0.22835097473417648, -0.3926549809396137, -0.16958217669996367,
      0.24359542640624454,
    ];
    const x = 0.4350441345216933;
    const opts: Partial<TNTOpts> = {
      maxIterations: 4,
      earlyStopping: { minMSE: 1e-15 },
    };
    const { XBest, metadata, maxIterations } = new TNT(A, B);
    const e = metadata[0];
    expect(e.mseMin).not.toBeNaN();
    expect(e.iterations).toBeLessThanOrEqual(maxIterations);
    expect(e.mse.length).toBeLessThanOrEqual(e.iterations + 1);
    expect(XBest.get(0, 0)).toBeCloseTo(x, 2);

    opts.maxIterations = 0;

    const result = new TNT(A, B, opts);
    expect(result.metadata[0].mseMin).toBeLessThan(0.45);
  });
});
