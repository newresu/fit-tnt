// @ts-check
import js from '@eslint/js';
import markdown from '@eslint/markdown';
import vitest from '@vitest/eslint-plugin';
import jsdoc from 'eslint-plugin-jsdoc';
import { globalIgnores } from 'eslint/config';
import globals from 'globals';
import ts from 'typescript-eslint';

const lOpts = {
  parser: ts.parser,
  parserOptions: {
    projectService: {
      allowDefaultProject: [//expands ts linting 
        'eslint.config.mjs',
        '*.ts',
        '*.mts',
        'benchmark/*.ts',
      ],
      defaultProject: 'tsconfig.json',
    },
    tsconfigRootDir: import.meta.dirname,
  },
};
export default ts.config(
  globalIgnores(
    //shared for all objects. Matches files and dirs.
    ['coverage', 'docs', '**/node_modules', 'demo'],
    'Ignore Common',
  ),
  {
    name: 'JS/TS Linting',
    languageOptions: lOpts,
    files: ['benchmark/*.ts', 'src/**/*.ts'],
    extends: [
      js.configs.recommended,
      ts.configs.strictTypeChecked,
      ts.configs.stylistic,
    ],
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
    name: 'Tests',
    files: ['src/**__tests__/**'],
    plugins: {
      vitest,
    },
    rules: {
      ...vitest.configs.recommended.rules, // you can also use vitest.configs.all.rules to enable all rules
      'vitest/max-nested-describe': ['error', { max: 3 }], // you can also modify rules' behavior using option like this
    },
  },
  {
    name: 'Benchmark',
    files: ['benchmark/*.ts'],
    languageOptions: {
      ...lOpts,
      globals: { ...globals.browser, ...globals.node },
    },
  },
  {
    name: 'JSDoc',
    files: ['src/**/*.ts'], // always use files when using ignores.
    ignores: ['src/**/*test.ts'], // applies to files only
    plugins: { jsdoc },
    rules: {
      'jsdoc/require-description': 'warn',
    },
    extends: [jsdoc.configs['flat/recommended-typescript']],
  },
  {
    name: 'Readme Codeblocks',
    files: ['*.md'],
    plugins: { markdown },
    language: 'markdown/gfm', //github flavoured.
    extends: [markdown.configs.recommended],
  },
);
