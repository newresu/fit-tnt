/// <reference types="vitest" />
import { defineConfig } from 'vite';
export default defineConfig({
  test: {
    coverage: {
      extension: ['.ts'],
      reportOnFailure: true,
      enabled: true,
      provider: 'v8',
    },
  },
});
