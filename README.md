# TNT

[![NPM version][npm-image]][npm-url]
[![build status][ci-image]][ci-url]
[![Test coverage][codecov-image]][codecov-url]
[![npm download][download-image]][download-url]

<!--
[![DOI](https://zenodo.org/badge/DOI/[DOINUMBER]/zenodo.8189402.svg)](https://doi.org/[DOINUMBER]/zenodo.8189402) -->

A fast least squares fitting method for dense matrices.

The implementation based off [the TNT paper](https://ieeexplore.ieee.org/abstract/document/8425520) by J. M. Myre et al.

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
const b = [6, 12]; // or [[6],[12]]

try {
  const { xBest, mse, method } = new TNT(A, b);
} catch (e) {
  console.error(e);
}
```

The preconditioning was based off [Ridge Regression](https://en.wikipedia.org/wiki/Ridge_regression), aiming to improve the condition number (and convergence.) This is a bit different to the paper.

## Documentation

- [ReadTheDocs 😊](https://newresu.github.io/fit-tnt/modules.html)

## Comparison: TNT vs Pseudo-Inverse

The smaller the **rows/columns** ratio, the more one should use the
pseudo inverse method (currently this `criticalRatio` is set to 1/10)

- Matrix Shape: 500 200

```
┌───────────────┬─────────────────────┬─────────────────────┐
│ (index)       │       Avg Exec Time │           Avg Error │
├───────────────┼─────────────────────┼─────────────────────┤
│ TNT           │ 0.09470919929999999 │ 0.04945702797110891 │
│ PseudoInverse │ 0.49272041820000007 │ 0.04945702797110894 │
└───────────────┴─────────────────────┴─────────────────────┘
```

- Speed Up: 5.202455747083906

## Misc.

- In some cases it won't get to a low error, but [normalizing improves performance.](https://stats.stackexchange.com/questions/306019/in-linear-regression-why-do-we-often-have-to-normalize-independent-variables-pr)
- If it errors, it fallbacks to the pseudo-inverse method.
- Very under-determined are ran by pseudo-inverse (it's faster in those cases.)

<details>
<summary>
Concepts
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

TNT revives CGNR for Dense Large matrices. It uses a modified version Preconditioned-CGNR to update $A^T\,A$ so that $A$ becomes positive definite which means it has full column rank.

To be clear, positive definite means:
$$x^T M x \gt 0$$

In our case:

$$x^T \,(A^T A)\, x \gt 0$$

This means:

$$(A\,x)^T (A x) \gt 0$$

Which means that each $(\ldots)$ must be non-zero. This happens only when the columns are linearly independent. If the columns of $A$ are linearly independent then it's invertible/non-singular, and $A^T A$ is invertible.

So we want to pre-condition $A^T A$ so that it is invertible.

However, this can happen while also returning $L = \mathrm{Cho}(A^T\,A)$ that has some near-zero value in the diagonal, blowing up the method.

</details>

<details>

<summary>When does it fail?</summary>

If the matrix is positive-definite but the Cholesky decomposition returns some very small number in the diagonal. This triggers a very large number in the back-substitution.

The root cause seems to be very-ill-conditioned matrices. [Related post.](https://math.stackexchange.com/questions/730421/is-aat-a-positive-definite-symmetric-matrix)

The pseudoInverse will do better since the condition number is the square root of the normal equations (used by TNT.)

I suspect that one could add the value in the diagonal in a smarter way, so that no value in $L$ is very near $0$, but it's hard to know what this implies for the accuracy.

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
[ci-image]: https://github.com/newresu/fit-tnt/actions/workflows/nodejs.yml/badge.svg
[ci-url]: https://github.com/newresu/fit-tnt/actions/workflows/nodejs.yml
[codecov-image]: https://img.shields.io/codecov/c/github/newresu/fit-tnt.svg
[codecov-url]: https://codecov.io/gh/newresu/fit-tnt
[download-image]: https://img.shields.io/npm/dm/fit-tnt.svg
[download-url]: https://www.npmjs.com/package/fit-tnt
