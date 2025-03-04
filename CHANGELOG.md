# Changelog

## [6.0.0](https://github.com/newresu/fit-tnt/compare/v5.1.1...v6.0.0) (2025-03-04)


### ⚠ BREAKING CHANGES

* the pseudo inverse and options are removed since it makes sense to have this a separate program, especially given that it seems to do fine.

### Features

* when doing UL and given U^T = L and given that the result is symmetric, it's possible to save a little bit of time (actually only 1% speed up in large a matrix) ([6d46b4c](https://github.com/newresu/fit-tnt/commit/6d46b4c3aed2c50584ced8960b245be390992888))


### Code Refactoring

* the pseudo inverse and options are removed since it makes sense to have this a separate program, especially given that it seems to do fine. ([6d46b4c](https://github.com/newresu/fit-tnt/commit/6d46b4c3aed2c50584ced8960b245be390992888))

## [5.1.1](https://github.com/newresu/fit-tnt/compare/v5.1.0...v5.1.1) (2025-03-03)


### Bug Fixes

* do not add multiple rhs until fully tested. ([2f114f8](https://github.com/newresu/fit-tnt/commit/2f114f820b514364c3e1f6d1efc330fb907d4666))

## [5.1.0](https://github.com/newresu/fit-tnt/compare/v5.0.0...v5.1.0) (2025-03-03)


### Features

* experimental support for multiple right hand sides ([#15](https://github.com/newresu/fit-tnt/issues/15)) ([e0c126b](https://github.com/newresu/fit-tnt/commit/e0c126be0c5f474e47a39946e9ef87daae5c4862))

## [5.0.0](https://github.com/newresu/fit-tnt/compare/v4.0.2...v5.0.0) (2025-02-28)


### ⚠ BREAKING CHANGES

* Remove Cholesky Precondition Trick option (it's the only option.)  ([#13](https://github.com/newresu/fit-tnt/issues/13))

### Code Refactoring

* Remove Cholesky Precondition Trick option (it's the only option.)  ([#13](https://github.com/newresu/fit-tnt/issues/13)) ([db31e31](https://github.com/newresu/fit-tnt/commit/db31e315cd7f4e0b0a44558004e8f355a59517a4))

## [4.0.2](https://github.com/newresu/fit-tnt/compare/v4.0.1...v4.0.2) (2025-02-27)


### Bug Fixes

* replace where to throw the error from. ([95efa32](https://github.com/newresu/fit-tnt/commit/95efa328a8febc4e42384a65e091d36d972a7bc2))

## [4.0.1](https://github.com/newresu/fit-tnt/compare/v4.0.0...v4.0.1) (2025-02-27)


### Bug Fixes

* ensures throwing on pseudo inverse doesn't re-execute it. ([6d1569e](https://github.com/newresu/fit-tnt/commit/6d1569ec3fdf5f1433c0f24f06fe2b5a743d97a3))

## [4.0.0](https://github.com/newresu/fit-tnt/compare/v3.0.0...v4.0.0) (2025-02-27)


### ⚠ BREAKING CHANGES

* minError parameter instead of unacceptable error.

### Bug Fixes

* make all methods private ([9e8f065](https://github.com/newresu/fit-tnt/commit/9e8f06574755800880cc1d60f701403bd44ade91))
* pass to pseudo inverse when erroring or above max error ([9e8f065](https://github.com/newresu/fit-tnt/commit/9e8f06574755800880cc1d60f701403bd44ade91))


### Code Refactoring

* minError parameter instead of unacceptable error. ([9e8f065](https://github.com/newresu/fit-tnt/commit/9e8f06574755800880cc1d60f701403bd44ade91))

## [3.0.0](https://github.com/newresu/fit-tnt/compare/v2.0.0...v3.0.0) (2025-02-26)


### ⚠ BREAKING CHANGES

* number of cycles to a normal number, use pseudo inverse fallback.

### Bug Fixes

* number of cycles to a normal number, use pseudo inverse fallback. ([4cd2b31](https://github.com/newresu/fit-tnt/commit/4cd2b3135e9abf01f462a446237b0955a1a4d029))

## [2.0.0](https://github.com/newresu/fit-tnt/compare/v1.0.0...v2.0.0) (2025-02-25)


### ⚠ BREAKING CHANGES

* update default value

### Features

* improve accuracy  ([#4](https://github.com/newresu/fit-tnt/issues/4)) ([f714b29](https://github.com/newresu/fit-tnt/commit/f714b29502cef944560d407020bbd12c0422d1d6))


### Bug Fixes

* update default value ([2edf280](https://github.com/newresu/fit-tnt/commit/2edf2806954147e22e73c73d12c20414cfad6e1b))

## 1.0.0 (2025-02-24)


### Bug Fixes

* workflow token ([c9c72ad](https://github.com/newresu/fit-tnt/commit/c9c72ad95e03ba3a10c5719c9ad6d102817144e7))
