import { spawnSync } from 'child_process';
import { readdirSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const scriptsDirectory = path.dirname(fileURLToPath(import.meta.url));
const uiDirectory = path.dirname(scriptsDirectory);
const testFiles = readdirSync(scriptsDirectory)
  .filter(filename => filename.endsWith('.test.mjs'))
  .sort()
  .map(filename => path.join('scripts', filename));

if (testFiles.length === 0) {
  throw new Error('No UI test files were found.');
}

// Several tests deliberately exercise process-wide environment variables and
// filesystem state. Serial execution keeps the aggregate command deterministic.
const result = spawnSync(process.execPath, ['--test', '--test-concurrency=1', ...testFiles], {
  cwd: uiDirectory,
  env: process.env,
  stdio: 'inherit',
});

if (result.error) {
  throw result.error;
}

process.exit(result.status ?? 1);
