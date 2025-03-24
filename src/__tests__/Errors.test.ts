import { describe, expect, it } from 'vitest';

import { NaNOrNonFiniteError, PreconditionError } from '../Errors';

describe('Custom Errors', () => {
  describe('PreconditionError', () => {
    it('should create a PreconditionError with correct message and name', () => {
      const error = new PreconditionError();
      expect(error).toBeInstanceOf(Error);
      expect(error).toBeInstanceOf(PreconditionError);
      expect(error).toHaveProperty('message');
      expect(error.message).toBe(
        'Preconditioning AtA failed. This may be due to ill-conditioning.',
      );
    });
  });

  describe('NaNOrNonFiniteError', () => {
    it('should create a NaNOrNonFiniteError with correct message and name', () => {
      const error = new NaNOrNonFiniteError();
      expect(error).toBeInstanceOf(NaNOrNonFiniteError);
      expect(error).toHaveProperty('message');
      expect(error.message).toBe(
        'Infinite or NaN coefficients were found. This may be due to a very ill-conditioned matrix. Try `{pseudoInverseFallback:true}`.',
      );
    });
  });
});
