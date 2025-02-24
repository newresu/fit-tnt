// @ts-check
import globals from 'globals';
import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config([
  { ignores: ['docs', 'profile'] },
  {
    files: ['**/*.ts'],
    languageOptions: { globals: { ...globals.browser, ...globals.node } },
    rules: eslint.configs.recommended.rules,
  },
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
]);
