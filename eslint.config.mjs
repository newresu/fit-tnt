// @ts-check
import eslint from '@eslint/js';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default tseslint.config([
    { ignores: ['docs/**', 'benchmark/**'] },
    {
        files: ['**/*.ts'],
        languageOptions: { globals: { ...globals.browser, ...globals.node } },
        rules: eslint.configs.recommended.rules,
    },
    eslint.configs.recommended,
    ...tseslint.configs.recommended,
]);
