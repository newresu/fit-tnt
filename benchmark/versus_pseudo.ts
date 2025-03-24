import { Matrix, pseudoInverse } from 'ml-matrix';
import { performance } from 'perf_hooks';

import { TNT } from '../lib/index.js';

// rows, columns, projects
const [m, n, p] = [500, 200, 4];

const [tntTime, tntErr]: [number[], number[]] = [[], []];
const [piTime, piErr]: [number[], number[]] = [[], []];

let temp;

let [s, e] = [0, 0];
const cycles = 10;

for (let i = 0; i < cycles; i++) {
  const A = Matrix.random(m, n).mul(100000);
  const B = Matrix.random(m, p);
  s = performance.now();
  temp = new TNT(A, B, {
    maxIterations: 8,
  });
  e = performance.now();
  // push values TNT
  tntTime.push(e - s);
  tntErr.push(avg(temp.metadata.map((i) => i.mseMin)));

  /*pseudo inverse*/
  s = performance.now();
  temp = pseudoInverse(A).mmul(B);
  e = performance.now();
  temp = A.mmul(temp).sub(B);

  // push values pseudo inverse
  piTime.push(e - s);
  piErr.push(
    avg(
      temp
        .pow(2)
        .sum('column')
        .map((i) => i / A.rows),
    ),
  );
}

// milliseconds to seconds
const tntavgt = avg(tntTime) / 1000;
const piavgt = avg(piTime) / 1000;

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

function avg(arr: number[]) {
  let total = 0;
  for (const item of arr) {
    total += item;
  }
  return total / arr.length;
}
