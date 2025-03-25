import { Matrix, MatrixColumnSelectionView } from 'ml-matrix';

import { AnyMatrix, Array1D, Array2D } from './types';

/**
 * The output $B$ can be passed as flat array, nested array or matrix.
 * This function ensures that it is correctly converted to matrix in those
 * cases.
 * @param input matrix or array
 * @returns B matrix
 */
export function ensureMatrixB(input: Array1D | Array2D | AnyMatrix): AnyMatrix {
  if (Matrix.isMatrix(input)) {
    return input;
  } else if (Array.isArray(input[0])) {
    return new Matrix(input as number[][]);
  }
  return Matrix.columnVector(input as number[]);
}

/**
 * These uses one of the arrays as filter for the others.
 * @param indices these are the indices from the original columns of X.
 * @param alpha conjugate gradient value
 * @param subsetIndices X indices but shifted to zero, for other matrices.
 * @returns filtered arrays with non NaN or Inf values.
 */
export function filterIndices(
  indices: number[],
  alpha: number[],
  subsetIndices?: number[],
) {
  const tmpIndices: number[] = [];
  const tmpAlpha: number[] = [];
  const tmpSubsetIndices: number[] = [];

  for (let i = 0; i < indices.length; i++) {
    if (Number.isFinite(indices[i])) {
      tmpIndices.push(indices[i]);
      tmpAlpha.push(alpha[i]);
      if (subsetIndices) {
        tmpSubsetIndices.push(subsetIndices[i]);
      } else {
        tmpSubsetIndices.push(i);
      }
    }
  }
  return [tmpIndices, tmpAlpha, tmpSubsetIndices];
}

/**
 * Generate a matrix column view for the matrices from the indices.
 * @param indices the indices that we want to use from the matrices.
 * @param matrices the matrices we will select columns from.
 * @returns views of each matrix.
 */
export function getColumnViews(indices: number[], ...matrices: AnyMatrix[]) {
  return matrices.map((m) => new MatrixColumnSelectionView(m, indices));
}
