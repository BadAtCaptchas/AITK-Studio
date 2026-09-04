import { FlatCompat } from '@eslint/eslintrc';
import path from 'path';
import { fileURLToPath } from 'url';

const baseDirectory = path.dirname(fileURLToPath(import.meta.url));
const compat = new FlatCompat({ baseDirectory });

const eslintConfig = [
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
    // These rules currently report repository-wide legacy patterns. TypeScript's
    // strict compiler remains the correctness gate while those areas are migrated.
    // Keeping them disabled also prevents editor diagnostics from obscuring
    // actionable lint findings.
    rules: {
      '@next/next/no-img-element': 'off',
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-unused-vars': 'off',
      'prefer-const': 'warn',
      'react-hooks/exhaustive-deps': 'off',
    },
  },
];

export default eslintConfig;
