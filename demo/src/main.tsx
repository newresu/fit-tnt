import { TNT } from 'fit-tnt';
import { Matrix, pseudoInverse } from 'ml-matrix';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

const m = 500; // use 100 to see TNT using pseudo inverse by default
const n = 200;
const p = 4;

/* first */
const [tntTime, tntErr]: number[][] = [[], []];
const [piTime, piErr]: number[][] = [[], []];
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
const table = JSON.stringify({
  TNT: {
    'Avg Exec Time': tntavgt,
    'Avg Error': avg(tntErr),
  },
  PseudoInverse: {
    'Avg Exec Time': piavgt,
    'Avg Error': avg(piErr),
  },
});
const speedUp = JSON.stringify({ SpeedUp: piavgt / tntavgt });

function avg(arr: number[]) {
  let total = 0;
  for (const item of arr) {
    total += item;
  }
  return total / arr.length;
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <div>
      <h1>Hello</h1>
      <p>{table}</p>
      <p>{speedUp}</p>
    </div>
  </StrictMode>,
);
