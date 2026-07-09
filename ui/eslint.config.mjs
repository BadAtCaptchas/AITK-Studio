import { FlatCompat } from '@eslint/eslintrc';
import path from 'path';
import { fileURLToPath } from 'url';

const baseDirectory = path.dirname(fileURLToPath(import.meta.url));
const compat = new FlatCompat({ baseDirectory });

export default [
  {
    ignores: [
      '.next/**',
      '.test-tmp/**',
      'dist/**',
      'next-env.d.ts',
      'node_modules/**',
      'src/generated/**',
      'worker-dist/**',
    ],
  },
  ...compat.extends('next/core-web-vitals', 'next/typescript'),
  {
    // Keep legacy debt visible without making the initial lint gate unusable.
    // New code should continue to follow the repository's no-new-any policy.
    rules: {
      '@typescript-eslint/no-explicit-any': 'warn',
      'prefer-const': 'warn',
    },
  },
];
