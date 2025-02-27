// @ts-check
import eslint from '@eslint/js';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { // applies to all: https://eslint.org/docs/latest/use/configure/ignore#ignoring-files
    ignores: ['coverage', 'docs', 'benchmark', 'node_modules'],
  },
  {
    languageOptions: { globals: { ...globals.browser, ...globals.node } },
  },
  eslint.configs.recommended,
  tseslint.configs.strict,
  tseslint.configs.stylistic,
);
