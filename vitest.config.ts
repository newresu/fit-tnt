/// <reference types="vitest" />
import { defineConfig } from 'vite';

export default defineConfig({
    test: {
        testTimeout: 12000,
        logHeapUsage: true,
        coverage: {
            extension: ['.ts'],
            provider: 'v8',
            include: ['src/**'],
            skipFull: true,
        },
    },
});
