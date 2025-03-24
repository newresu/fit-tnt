export class PreconditionError extends Error {
  constructor() {
    const message =
      'Preconditioning AtA failed. This may be due to ill-conditioning.';
    super(message); // (1)

    this.name = 'PreconditionError'; // (2)
  }
}

export class NaNOrNonFiniteError extends Error {
  constructor() {
    const message =
      'Infinite or NaN coefficients were found. This may be due to a very ill-conditioned matrix. Try `{pseudoInverseFallback:true}`.';
    super(message); // (1)
    this.name = 'NaNOrNonFiniteError'; // (2)
  }
}
