import { performance } from 'perf_hooks';
import { Matrix, pseudoInverse } from 'ml-matrix';
import { TNT } from '../lib/index.js';

const m = 500; // use 100 to see TNT using pseudo inverse by default
const n = 200;

console.log('DIMENSIONS: ', m, n);

/* first */
let tntp = [];
const pi = [];
let t;
let r;
let [s, e] = [0, 0];
const cycles = 10;

for (let i = 0; i < cycles; i++) {
  const A = Matrix.random(m, n);
  const b = Matrix.random(m, 1);
  s = performance.now();
  t = new TNT(A, b, { pseudoInverseFallback: true });
  e = performance.now();
  tntp.push((e - s) / 1000);
  console.log(`${t.method} ${i} error: `, t.mseMin);

  /*pseudo inverse*/
  s = performance.now();
  r = pseudoInverse(A).mmul(b);
  e = performance.now();
  pi.push((e - s) / 1000);
  r = A.mmul(r).sub(b);
  console.log(`PI ${i} error: `, r.dot(r) / A.rows);
}

const tntavgt = avg(tntp)
const piavgt = avg(pi)
console.log('TNT AVG EX TIME: ', tntavgt);
console.log('PI AVG EXEC TIME: ',piavgt );
console.log('RATIO (tnt/pi) AVG TIME: ', tntavgt/piavgt);

function avg(a) {
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    total += a[i];
  }
  return total / a.length;
}
