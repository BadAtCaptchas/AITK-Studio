import { spawnSync } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const defaultUiRoot = path.dirname(scriptDirectory);
const defaultStatePath = path.join(defaultUiRoot, '..', '.tmp', 'ui-dependencies.json');

function isRecord(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

async function atomicWrite(target, contents) {
  await fsp.mkdir(path.dirname(target), { recursive: true });
  const temporary = path.join(
    path.dirname(target),
    `.${path.basename(target)}.${process.pid}.${randomBytes(6).toString('hex')}.tmp`,
  );
  try {
    await fsp.writeFile(temporary, contents);
    await fsp.rename(temporary, target);
  } finally {
    await fsp.rm(temporary, { force: true }).catch(() => undefined);
  }
}

export async function dependencyManifestFingerprint(uiRoot) {
  const hash = createHash('sha256');
  hash.update(`${process.platform}\0${process.arch}\0${process.versions.modules || ''}\0`);
  for (const filename of ['package.json', 'package-lock.json']) {
    hash.update(filename);
    hash.update('\0');
    hash.update(await fsp.readFile(path.join(uiRoot, filename)));
    hash.update('\0');
  }
  return hash.digest('hex');
}

async function readInstalledFingerprint(statePath) {
  try {
    const parsed = JSON.parse(await fsp.readFile(statePath, 'utf8'));
    return isRecord(parsed) && typeof parsed.fingerprint === 'string' ? parsed.fingerprint : null;
  } catch {
    return null;
  }
}

function runNpmInstall(uiRoot) {
  const npmArguments = ['install', '--no-save', '--no-audit', '--no-fund'];
  const npmCli = process.env.npm_execpath;
  const result = npmCli && fs.existsSync(npmCli)
    ? spawnSync(process.execPath, [npmCli, ...npmArguments], { cwd: uiRoot, stdio: 'inherit' })
    : spawnSync(process.platform === 'win32' ? 'npm.cmd' : 'npm', npmArguments, {
        cwd: uiRoot,
        stdio: 'inherit',
        shell: process.platform === 'win32',
      });
  if (result.error) throw result.error;
  return result.status ?? 1;
}

export async function ensureUiDependencies(options = {}) {
  const uiRoot = path.resolve(options.uiRoot || defaultUiRoot);
  const statePath = path.resolve(options.statePath || defaultStatePath);
  const lockPath = path.join(uiRoot, 'package-lock.json');
  const nodeModulesPath = path.join(uiRoot, 'node_modules');
  const fingerprint = await dependencyManifestFingerprint(uiRoot);
  if (
    (await readInstalledFingerprint(statePath)) === fingerprint &&
    (await fsp.stat(nodeModulesPath).then(stat => stat.isDirectory()).catch(() => false))
  ) {
    console.log('UI dependencies already match package manifests.');
    return { installed: false, fingerprint };
  }

  const lockSnapshot = await fsp.readFile(lockPath);
  let installStatus;
  try {
    installStatus = await (options.runInstall || runNpmInstall)(uiRoot);
  } finally {
    const currentLock = await fsp.readFile(lockPath).catch(() => null);
    if (!currentLock || !currentLock.equals(lockSnapshot)) {
      await atomicWrite(lockPath, lockSnapshot);
      console.log('Restored repository-managed package-lock.json.');
    }
  }

  if (installStatus !== 0) {
    throw new Error(`npm install failed with exit code ${installStatus}`);
  }

  await atomicWrite(
    statePath,
    `${JSON.stringify({ schemaVersion: 1, fingerprint }, null, 2)}\n`,
  );
  console.log('UI dependencies are ready.');
  return { installed: true, fingerprint };
}

const invokedPath = process.argv[1] ? pathToFileURL(path.resolve(process.argv[1])).href : '';
if (invokedPath === import.meta.url) {
  ensureUiDependencies().catch(error => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
