import { Matrix, MatrixColumnSelectionView } from 'ml-matrix';

import { AnyMatrix, Array1D, Array2D } from './types';

/**
 * The output $B$ can be passed as flat array, nested array or matrix.
 * This function ensures that it is correctly converted to matrix in those
 * cases.
 *
 * @param B as input
 * @returns B as a matrix.
 */
export function ensureMatrix(input: Array1D | Array2D | AnyMatrix): AnyMatrix {
  if (Matrix.isMatrix(input)) {
    return input;
  } else if (Array.isArray(input[0])) {
    return new Matrix(input as number[][]);
  }
  return Matrix.columnVector(input as number[]);
}

/**
 * These uses one of the arrays as filter for the others.
 * @param indices
 * @param alpha
 * @param subsetIndices
 * @returns filtered arrays.
 */
export function filterIndices(
  indices: number[],
  alpha: number[],
  subsetIndices?: number[],
) {
  const [tmpIndices, tmpAlpha, tmpSubsetIndices]: number[][] = [[], [], []];
  if (subsetIndices) {
    for (let i = 0; i < indices.length; i++) {
      if (Number.isFinite(indices[i])) {
        tmpIndices.push(indices[i]);
        tmpSubsetIndices.push(subsetIndices[i]);
        tmpAlpha.push(alpha[i]);
      }
    }
  } else {
    for (let i = 0; i < alpha.length; i++) {
      if (Number.isFinite(alpha[i])) {
        tmpIndices.push(indices[i]);
        tmpAlpha.push(alpha[i]);
        tmpSubsetIndices.push(i);
      }
    }
  }
  return [tmpIndices, tmpAlpha, tmpSubsetIndices];
}

/**
 * Generate a matrix column view for the matrices from the indices.
 * @param indices
 * @param matrices
 * @returns views.
 */
export function getColumnViews(indices: number[], ...matrices: AnyMatrix[]) {
  return matrices.map((m) => new MatrixColumnSelectionView(m, indices));
}
