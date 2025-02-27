import { Matrix, pseudoInverse } from 'ml-matrix';
import { performance } from 'perf_hooks';

import { TNT } from '../lib/index.js';

const m = 500; // use 100 to see TNT using pseudo inverse by default
const n = 200;

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
  const A = Matrix.random(m, n).mul(100);
  const b = Matrix.random(m, 1);
  s = performance.now();
  t = new TNT(A, b, { pseudoInverseFallback: true });
  e = performance.now();
  // push values TNT
  tntTime.push((e - s) / 1000);
  tntErr.push(t.mseMin);

  /*pseudo inverse*/
  s = performance.now();
  r = pseudoInverse(A).mmul(b);
  e = performance.now();
  r = A.mmul(r).sub(b);

  // push values pseudo inverse
  piTime.push((e - s) / 1000);
  piErr.push(r.dot(r) / A.rows);
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
