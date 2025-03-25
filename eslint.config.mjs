// @ts-check
import eslint from '@eslint/js';
import jsdoc from 'eslint-plugin-jsdoc';
import { globalIgnores } from 'eslint/config';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  // applies to all: https://eslint.org/docs/latest/use/configure/ignore#ignoring-files
  globalIgnores(['coverage', 'docs', '**/node_modules'], 'Ignore common'),
  {
    languageOptions: { globals: { ...globals.browser, ...globals.node } },
  },
  eslint.configs.recommended,
  tseslint.configs.strictTypeChecked,
  tseslint.configs.stylistic,
  {
    // override some previous rule
    rules: {
      '@typescript-eslint/restrict-template-expressions': [
        'error',
        {
          allowNumber: true,
        },
      ],
    },
  },
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
  jsdoc.configs['flat/recommended-typescript'],
  {
    // specific to a plugin
    files: ['src/**/*.ts'],
    ignores: ['src/**/*test.ts'],
    plugins: { jsdoc },
    rules: {
      'jsdoc/require-description': 'warn',
    },
  },
);
