// fisrt compile with tsc
const { performance } = require('perf_hooks');
const { Matrix, pseudoInverse } = require('ml-matrix');
const { TNT } = require('../lib/index');

const m = 10; // use 100 to see TNT using pseudo inverse by default
const n = 1000;
const A = Matrix.random(m, n);
const b = Matrix.random(m, 1);

const cr = m / n <= 0.01;

let s = performance.now();
let t;
for (let i = 0; i < 1; i++) {
  t = new TNT(A, b, { pseudoInverseFallback: true }).method;
}
let e = performance.now();
console.log(t, (e - s) / 1000);

if (cr) {
  let s = performance.now();
  let t;
  for (let i = 0; i < 1; i++) {
    t = new TNT(A, b, { pseudoInverseFallback: false }).method;
  }
  let e = performance.now();
  console.log(t, (e - s) / 1000);
}

s = performance.now();
for (let i = 0; i < 1; i++) {
  pseudoInverse(A).mmul(b);
}
e = performance.now();

console.log('PseudoInverse: ', (e - s) / 1000);
