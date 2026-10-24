import ts from 'typescript-eslint';
import react from 'eslint-plugin-react';
import hooks from 'eslint-plugin-react-hooks';

const eslintConfig = [
  {
    ignores: [
      '.next/**',
      '.runtime/**',
      'web-dist/**',
      '.review-build/**',
      '.test-tmp/**',
      'dist/**',
      'next-env.d.ts',
      'node_modules/**',
      'src/generated/**',
      'worker-dist/**',
    ],
  },
  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: { parser: ts.parser, parserOptions: { ecmaFeatures: { jsx: true } } },
    plugins: { '@typescript-eslint': ts.plugin, react, 'react-hooks': hooks },
    settings: { react: { version: 'detect' } },
    // These rules currently report repository-wide legacy patterns. TypeScript's
    // strict compiler remains the correctness gate while those areas are migrated.
    // Keeping them disabled also prevents editor diagnostics from obscuring
    // actionable lint findings.
    rules: {
      ...ts.configs.recommended.at(-1).rules,
      ...react.configs.flat.recommended.rules,
      ...react.configs.flat['jsx-runtime'].rules,
      ...hooks.configs.recommended.rules,
      'react/prop-types': 'off',
      'no-undef': 'off',
      'no-unused-vars': 'off',
      '@typescript-eslint/no-require-imports': 'off',
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-unused-vars': 'off',
      'prefer-const': 'warn',
      'react-hooks/exhaustive-deps': 'off',
    },
  },
];

export default eslintConfig;
