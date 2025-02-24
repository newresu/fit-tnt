# TNT

[![NPM version][npm-image]][npm-url]
[![build status][ci-image]][ci-url]
[![Test coverage][codecov-image]][codecov-url]
[![npm download][download-image]][download-url]
[![DOI](https://zenodo.org/badge/DOI/[DOINUMBER]/zenodo.8189402.svg)](https://doi.org/[DOINUMBER]/zenodo.8189402)

Custom implementation of [the TNT paper](https://ieeexplore.ieee.org/abstract/document/8425520) by J. M. Myre et al. In many cases, this method converges at the first iteration.

## Install and Use

```bash
npm i fit-tnt
```

```ts
import { TNT } from 'fit-tnt';

const A = [
  [1, 2, 3],
  [4, 5, 6],
]; // 2x3
const b = [6, 7]; // or [[6],[7]]

try {
  const tnt = new TNT(A, b);
  console.log(tnt.xBest, tnt.mse, tnt.iterations);
  // use xBest.to1DArray unless you want it as Matrix instance.
} catch (e) {
  console.log(e); // just as example
}
```

After `solve` the `tnt` instance has `iterations` and `mse` populated.

## When does it fail?

If the matrix is positive-definite but the Cholesky decomposition returns some very small number in the diagonal.

This triggers a very large number in the back-substitution.

The obvious question is: can those numbers be somewhat reduced?
At the time of writing, I'm unsure.

Enabling `{pseudoInverseFallback:true}` and it will solve it in the cases where TNT fails.

In any case, TNT is substantially faster than the current pseudo-inverse method used (about 4X faster when TNT finishes successfully).

## [API Documentation](https://newresu.github.io/fit-tnt/)

<details>
<summary>
Concepts (click to open)
</summary>

The linear problem appears in all science:

$$A\,x = b$$

and methods to solve it fast abound. In practice, this equation almost never the straightforward solution $A^{-1}$, so the Least-Squares approach is used to minimize the squared error in the predictions:

$$ E(x) = \mathrm{min}\_x \left|\left| A\,x -b \right|\right|\_2^2$$

i.e to minimize the $L_2$ (or $L_2^2$ which is equivalent.); this is the Least-Squares problem.

The solution, where the error-gradient is zero i.e $\nabla_x E(x)=0$ is $$A^T\,A x = A^T b$$

When computed directly (as done here), $A^T\,A$ has a condition number $\kappa (A^T A) = \kappa (A)^2$. This affects the precision of the solutions; especially when $\kappa (A) > 10^8$.

Larger condition number also tends to slow the convergence.

**TNT**

The Conjugate Gradient for Normal Residual (CGNR) is a popular method for solving Sparse Least-Squares problems, where the design matrix has many zeros.

The reason for "Large" is that systems with $m \lt\lt n$ can be solved faster and more accurately using the Pseudo-Inverse. Even though the QR decomposition-method can be more accurate, TNT tends to be faster in overdetermined problems where $m \approx n$ or $m \gt n$.

TNT revives CGNR for Dense Large matrices. It uses a modified version Preconditioned-CGNR to update $A^T\,A$ so that it's positive definite and converges faster.

</details>

<details>
<summary>
Algorithm Description
</summary>

1. Carry out product: $N=A^T\,A$ (`N` is Symmetric.)
2. [Cholesky Decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) and factor: R, p = Cho(N)
3. `if !p: N = N + e\*I`, $\epsilon$ being a tiny number.
4. Residual $r_0 = A\,x_0 - b$
5. Gradient per coefficient ($r$), $g_0 = A^T r_0$
6. Error in the coefficients $z_0 = R^{-1}\,g_0$
7. Get $\alpha$ as `a = dot(z,g)/dot (r,r)`
8. Update $x$ as $x_{i+1}=x_{i} + a_i\times p_i$
9. Next residual $r_{i+1} = r_i - a_i \times r_i$
10. New gradient $g_{i+1} = A^T r_{i+1}$
11. New error in coefficients: $z_{i+1} = R^{-1}\,g_{i+1}$
12. Get $\beta$ `beta = dot(z_{i+1},g_{i+1})/dot (z_i,g_i)`

</details>

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/fit-tnt.svg
[npm-url]: https://www.npmjs.com/package/fit-tnt
[ci-image]: https://github.com/newresu/fit-tnt/workflows/Node.js%20CI/badge.svg?branch=main
[ci-url]: https://github.com/newresu/fit-tnt/actions?query=workflow%3A%22Node.js+CI%22
[codecov-image]: https://img.shields.io/codecov/c/github/newresu/fit-tnt.svg
[codecov-url]: https://codecov.io/gh/newresu/fit-tnt
[download-image]: https://img.shields.io/npm/dm/fit-tnt.svg
[download-url]: https://www.npmjs.com/package/fit-tnt
