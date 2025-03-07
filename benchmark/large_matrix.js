import { Matrix, pseudoInverse } from 'ml-matrix';
import { performance } from 'perf_hooks';

import { TNT } from '../lib/index.js';

/* first */
const tntTime = [];
const tntErr = [];
let t;
let [s, e] = [0, 0];
const cycles = 100;

const m = 1e4;
const n = 1e2;

for (let i = 0; i < cycles; i++) {
  const A = Matrix.random(m, n).mul(100);
  const b = Matrix.random(m, 1);
  s = performance.now();
  t = new TNT(A, b, {
    maxIterations: 4,
  });
  e = performance.now();
  // push values TNT
  tntTime.push((e - s) / 1000);
  tntErr.push(t.metadata[0].mseMin);
}

const tntavgt = avg(tntTime);

// console.log('\n Matrix Shape: ', m, n, '\n');
console.table({
  TNT: {
    'Avg Exec Time': tntavgt,
    'Avg Error': avg(tntErr),
  },
});

function avg(a) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i];
  }
  return total / a.length;
}
