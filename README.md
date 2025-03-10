# TNT

[![NPM version][npm-image]][npm-url]
[![build status][ci-image]][ci-url]
[![Test coverage][codecov-image]][codecov-url]
[![npm download][download-image]][download-url]

<!--
[![DOI](https://zenodo.org/badge/DOI/[DOINUMBER]/zenodo.8189402.svg)](https://doi.org/[DOINUMBER]/zenodo.8189402) -->

Least-squares solver for large, dense matrices. It is based off the [TNT](https://ieeexplore.ieee.org/abstract/document/8425520) paper by J. M. Myre et al.

Supports multiple right-hand-sides.

<details>
<summary>Recommendations</summary>

- Speed. Best when these apply:

  - $\large\frac{\mathrm{rows}}{\mathrm{cols}} \geq 1$.
  - Data columns $\geq 10$. But it's worth trying in any case.

- Accuracy: it's frequently as accurate as QR or PseudoInverse but it will have larger error (normally still acceptable) with tricky matrices.

[For speed, see comparison here.](#comparison-tnt-vs-pseudo-inverse).

_For calculations with non-zero intercept_, remember to push a $1$ to each row. The coefficient will be the last item in **XBest**.

A more thorough webpage to compare speed/accuracy will hopefully be included soon.

</details>

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
  const { XBest, metadata } = new TNT(A, b);
} catch (e) {
  console.error(e);
}
```

A related method is [Ridge Regression](https://en.wikipedia.org/wiki/Ridge_regression).

## Documentation

- [ReadTheDocs 😊](https://newresu.github.io/fit-tnt/modules.html)

## Comparison: TNT vs Pseudo-Inverse

The larger the **rows/columns** ratio, the more convenient to use TNT.
This is a benchmark on random matrices.

Inverting the shape below, TNT is slower.

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

<details>
<summary>
Concepts
</summary>

The linear problem appears in all science: $A\,x = b$. Methods to solve it fast abound. But $A^{-1}$ rarely exists in practice; the Least-Squares approach is used to minimize the squared error in the predictions:

$E(x) = \mathrm{min}_x \lVert A\,x -b \rVert_2^2$

We then look for $\nabla_x E(x)=0$ that is $A^T\,A x = A^T b$

When computed directly (as done here), $A^T\,A$ has a condition number $\kappa (A^T A) = \kappa (A)^2$. We try to reduce this problem with preconditioning. Larger condition number also tends to slow the convergence.

**TNT**

The Conjugate Gradient for Normal Residual (CGNR) is a popular method for solving Sparse Least-Squares problems, where the design matrix has many zeros.

For wide-$A$, where $\frac{n}{m} \gt 1$ calculating and factoring $A^T A$ becomes computationally demanding, given its $n^2$ separate elements. Here pseudo-inverse will be faster. TNT tends to be faster when $m \geq n$.

TNT preconditions $A^T\,A$ so that it has an inverse and a smaller condition number, then iteratively solves using CGNR.

Positive definite means that $x^T M x \gt 0$. In our case: $x^T \,(A^T A)\, x \gt 0$, and $(A\,x)^T (A x) \gt 0$

The $(\ldots)$ are non-zero when the columns are linearly independent. If the columns of $A$ are linearly independent then it's invertible/non-singular, and $A^T A$ is invertible.

So we want to pre-condition $A^T A$ so that it is invertible, we also want to avoid tiny numbers in the diagonal of the decomposition.

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
