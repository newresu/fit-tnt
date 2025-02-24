// fisrt compile with tsc
const { performance } = require('perf_hooks');
const { Matrix, pseudoInverse } = require('ml-matrix');
const { TNT } = require('../lib/index');

const A = Matrix.random(1000, 1000);
const b = Matrix.random(1000, 1);

let s = performance.now();
for (let i = 0; i < 1; i++) {
  new TNT(A, b, { pseudoInverseFallback: true });
}
let e = performance.now();

console.log('TNT: ', (e - s) / 1000);

s = performance.now();
for (let i = 0; i < 1; i++) {
  pseudoInverse(A).mmul(b);
}
e = performance.now();

console.log('PseudoInverse: ', (e - s) / 1000);
