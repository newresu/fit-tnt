import { Matrix} from "ml-matrix";
export function meanSquaredError(A: Matrix, x: Matrix, b: Matrix) {
  return A.mmul(x).sub(b).pow(2).mean();
}