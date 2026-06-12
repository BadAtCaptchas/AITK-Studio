import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '..', '..');
const syncScript = path.join(repoRoot, 'scripts', 'sync_local_changes.py');
const forwardedArgs = process.argv.slice(2);

const candidates = [
  ...(process.platform === 'win32'
    ? [[path.join(repoRoot, 'venv', 'Scripts', 'python.exe')], [path.join(repoRoot, '.venv', 'Scripts', 'python.exe')]]
    : [[path.join(repoRoot, 'venv', 'bin', 'python')], [path.join(repoRoot, '.venv', 'bin', 'python')]]),
  ...(process.env.PYTHON ? [[process.env.PYTHON]] : []),
  ...(process.platform === 'win32' ? [['py', '-3'], ['python']] : [['python3'], ['python']]),
];

const seen = new Set();
const uniqueCandidates = candidates.filter((candidate) => {
  const key = candidate.join('\0');
  if (seen.has(key)) return false;
  seen.add(key);
  return true;
});

for (const candidate of uniqueCandidates) {
  const [command, ...baseArgs] = candidate;
  const result = spawnSync(command, [...baseArgs, syncScript, ...forwardedArgs], {
    cwd: repoRoot,
    stdio: 'inherit',
    windowsHide: true,
  });

  if (result.error?.code === 'ENOENT') {
    continue;
  }

  if (result.error) {
    console.error(`Failed to run ${command}: ${result.error.message}`);
    process.exit(1);
  }

  process.exit(result.status ?? 1);
}

console.error('Could not find Python. Install Python 3 or set the PYTHON environment variable.');
process.exit(1);
