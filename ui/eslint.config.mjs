import nextVitals from 'eslint-config-next/core-web-vitals';
import nextTypescript from 'eslint-config-next/typescript';

const eslintConfig = [
  {
    ignores: [
      '.next/**',
      '.review-build/**',
      '.test-tmp/**',
      'dist/**',
      'next-env.d.ts',
      'node_modules/**',
      'src/generated/**',
      'worker-dist/**',
    ],
  },
  ...nextVitals,
  ...nextTypescript,
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
