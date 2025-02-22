// @ts-check
import globals from 'globals';
import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';
import jsdoc from 'eslint-plugin-jsdoc';
import markdown from '@eslint/markdown';
import json from '@eslint/json';

export default tseslint.config([
  eslint.configs.recommended,
  tseslint.configs.recommended,
  {
    files: ['src/**/*.ts', 'src/**/*.js'],
    languageOptions: { globals: { ...globals.browser, ...globals.node } },
    plugins: {
      jsdoc,
    },
  },
  {
    files: ['*.md'],
    plugins: {
      markdown,
    },
    processor: 'markdown/markdown',
  },
  {
    files: ['*.json'],
    language: 'json/json',
    plugins: {
      json,
    },
  },
]);
