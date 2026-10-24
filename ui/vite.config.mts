import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const version = readFileSync(path.resolve('..', 'version.py'), 'utf8').match(/VERSION\s*=\s*["']([^"']+)["']/)?.[1] ?? 'unknown';

export default defineConfig({
  plugins: [react(), {
    name: 'studio-bootstrap',
    transformIndexHtml: () => [{ tag: 'script', attrs: { src: '/app-config.js' }, injectTo: 'head-prepend' }],
  }],
  resolve: { alias: { '@': path.resolve('src') } },
  optimizeDeps: { entries: ['index.html'] },
  define: { 'import.meta.env.VITE_APP_VERSION': JSON.stringify(version) },
  build: { outDir: 'web-dist', sourcemap: false },
  server: {
    host: '127.0.0.1',
    strictPort: true,
    forwardConsole: false,
    watch: { ignored: ['**/.runtime/**', '**/web-dist/**', '**/dist/**', '**/.next/**', '**/.review-build/**'] },
    fs: {
      strict: true,
      allow: [path.resolve('.')],
      deny: [
        '.env', '.env.*', '*.{crt,pem}', '**/.git/**',
        '**/src/server/**', '**/src/app/api/**', '**/src/generated/**', '**/src/proxy.ts',
        '**/cron/**', '**/scripts/**', '**/.runtime/**', '**/.next/**', '**/.review-build/**', '**/*.sqlite*', '**/aitk_db.db*',
      ],
    },
  },
});
