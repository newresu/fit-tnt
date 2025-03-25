interface Criteria {
  /**
   * epsilon added to the diagonal of L
   */
  eps: number;
  /**
   * min / avg used as stop condition.
   */
  ratio: number;
}
/**
 * Calculate `eps` and `ratio` used in the loops.
 * values all positive | 0 -> don't take `abs(item)`
 * @param nnArr non-negative array of numbers.
 * @param power such that `eps = average * 10 ^ power`
 * @returns Criteria
 */
export function getCriteria(nnArr: number[], power: number): Criteria {
  // below this it's inaccurate.
  if (nnArr.length === 0) {
    throw new Error('Array cannot be empty');
  }
  const delta = Number.EPSILON * 1000;
  let min = Infinity;
  let avg = 0;
  let counter = 0;
  for (const item of nnArr) {
    if (Number.isFinite(item)) {
      avg += item;
      if (item < min) {
        min = item;
      }
      counter++;
    }
  }
  min += delta;
  avg /= counter;
  return {
    eps: avg * Math.pow(10, power),
    ratio: min / avg,
  };
}
