export class AsymmetricError extends Error {
  constructor(message: string) {
    super(message); // (1)
    this.name = "AsymmetricError"; // (2)
  }
}

export class ExpectedVectorGotMatrixError extends Error{
  constructor(message: string) {
    super(message); // (1)
    this.name = "ExpectedVectorGotMatrixError"; // (2)
  }
}