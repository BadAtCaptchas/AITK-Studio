import { build } from 'esbuild';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { generateApiRegistry } from './api-registry.mjs';

export const uiRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export async function buildServer() {
  await generateApiRegistry(uiRoot);
  await build({
    absWorkingDir: uiRoot,
    entryPoints: ['src/server/http/app.ts'],
    outfile: '.runtime/server.cjs',
    bundle: true,
    platform: 'node',
    format: 'cjs',
    target: 'node22',
    packages: 'external',
    sourcemap: true,
    logLevel: 'warning',
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await buildServer();
