import { Matrix } from "ml-matrix";
import { AsymmetricError } from "./Errors";

export function initSafetyChecks(AtA: Matrix, A: Matrix, y: Matrix) {
  if (!AtA.isSymmetric()) {
    throw new AsymmetricError("AtA must be symmetric.");
  }
  if (A.rows !== y.rows) {
    throw new RangeError(
      `Rows of A and y must match. Found dim(A)=(${A.rows}, ${A.columns}) and dim(y)=(${y.rows}, ${y.columns})`
    );
  }
  if (!y.isColumnVector()) {
    throw new Error(`One Right-Hand-Side is supported. Found ${y.columns}`);
  }
  
}
