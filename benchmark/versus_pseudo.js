import { Matrix, pseudoInverse } from 'ml-matrix';
import { performance } from 'perf_hooks';

import { TNT } from '../lib/index.js';

const m = 500; // use 100 to see TNT using pseudo inverse by default
const n = 200;
const p = 4;

/* first */
const tntTime = [];
const tntErr = [];
const piTime = [];
const piErr = [];
let t;
let r;
let [s, e] = [0, 0];
const cycles = 10;

for (let i = 0; i < cycles; i++) {
  const A = Matrix.random(m, n).mul(100000);
  const B = Matrix.random(m, p);
  s = performance.now();
  t = new TNT(A, B, {
    maxIterations: 8,
  });
  e = performance.now();
  // push values TNT
  tntTime.push((e - s) / 1000);
  tntErr.push(avg(t.metadata.map((i) => i.mseMin)));

  /*pseudo inverse*/
  s = performance.now();
  r = pseudoInverse(A).mmul(B);
  e = performance.now();
  r = A.mmul(r).sub(B);

  // push values pseudo inverse
  piTime.push((e - s) / 1000);
  piErr.push(
    avg(
      r
        .pow(2)
        .sum('column')
        .map((i) => i / A.rows),
    ),
  );
}

const tntavgt = avg(tntTime);
const piavgt = avg(piTime);

console.log('\n Matrix Shape: ', m, n, '\n');
console.table({
  TNT: {
    'Avg Exec Time': tntavgt,
    'Avg Error': avg(tntErr),
  },
  PseudoInverse: {
    'Avg Exec Time': piavgt,
    'Avg Error': avg(piErr),
  },
});
console.log('\n ----> Speed Up: ', piavgt / tntavgt, '\n');

function avg(a) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i];
  }
  return total / a.length;
}
