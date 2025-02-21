import { makeData } from "../makeData";
import { tnt } from "../tnt";

for (let i = 0; i < 10e4; i++) {
  const { inputs: A, outputs: b } = makeData(4, 5);
  const result = tnt(A, b, { maxIterations: 100, tolerance: 1e-25 });
  if (!isFinite(result.solution[0])) {
    console.log(A);
  }
}
