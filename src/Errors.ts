export class AsymmetricError extends Error {
  constructor(message: string) {
    super(message); // (1)
    this.name = 'AsymmetricError'; // (2)
  }
}