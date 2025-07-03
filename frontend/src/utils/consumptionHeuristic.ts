export function ledLightBulbFromTokens(tokens: number): number {
  // the heuristic estimates an energy
  // consumption of 3J per token generated
  // which is equivalent to the consumption
  // of a typical 10W LED light bulb during
  // a third of a second
  const joules = tokens * 3;
  const seconds: number = joules / 10;
  return Math.round(seconds / 60);
}
