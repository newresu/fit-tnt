import { makeData } from "../makeData";
import { tnt } from "../tnt";
const { inputs: A, outputs: y, coefficients: answer } = makeData(5, 2);
console.log(answer);
console.log(tnt(A, y));
