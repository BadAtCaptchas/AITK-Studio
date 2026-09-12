import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import sqlite3 from 'sqlite3';
import { fileURLToPath } from 'node:url';
const execute = promisify(execFile);
const ui = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-browser-fixture-'));
const env = {
  ...process.env,
  AITK_SQLITE_PATH: path.join(root, 'test.sqlite'),
  AITK_DB_PROVIDER: 'sqlite',
  AITK_BIND_HOST: '127.0.0.1',
  AITK_NETWORK_MODE: '0',
  AITK_PUBLIC_URL: '',
  MODELS_PATH: '',
  AITK_ALLOWED_HOSTS: '',
  AITK_TRUSTED_PROXY_IPS: '',
  AI_TOOLKIT_AUTH: 'browser-fixture-local-only',
  AITK_ENABLE_TENSORBOARD: '0',
  AITK_CLOUDFLARED_ENABLED: '0',
  AI_TOOLKIT_FILE_SERVER_WORKERS: '1',
  AITK_INTERNAL_TOKEN: 'browser-fixture-internal-only',
  AITK_INTERNAL_URL: 'http://127.0.0.1:15875',
  HF_HUB_OFFLINE: '1',
  TRANSFORMERS_OFFLINE: '1',
  NO_ALBUMENTATIONS_UPDATE: '1',
};
await execute(process.execPath, ['scripts/patch-next-middleware.mjs'], { cwd: ui, env });
await execute(process.execPath, ['scripts/prepare-db.mjs'], { cwd: ui, env, timeout: 60_000 });
const db = new sqlite3.Database(env.AITK_SQLITE_PATH);
for (const [key, name] of [
  ['DATASETS_FOLDER', 'datasets'],
  ['TRAINING_FOLDER', 'output'],
  ['MODELS_PATH', 'models'],
]) {
  const folder = path.join(root, name);
  await fs.mkdir(folder);
  await new Promise((resolve, reject) =>
    db.run('INSERT OR REPLACE INTO Settings(key,value) VALUES(?,?)', [key, folder], error =>
      error ? reject(error) : resolve(),
    ),
  );
}
await new Promise(resolve => db.close(resolve));
const children = [];
let stopping = false;
async function stop() {
  if (stopping) return;
  stopping = true;
  for (const child of children) {
    if (child.connected) child.send({ type: 'aitk-shutdown', signal: 'SIGTERM' });
    else child.kill('SIGTERM');
  }
  await Promise.all(
    children.map(child =>
      child.exitCode !== null ? Promise.resolve() : new Promise(resolve => child.once('exit', resolve)),
    ),
  );
  await fs.rm(root, { recursive: true, force: true });
  process.exit(0);
}
for (const [script, args] of [
  ['cron/fileServer.js', ['start', '--port', '15875']],
  ['cron/worker.js', []],
]) {
  const child = spawn(process.execPath, [path.join(process.env.AITK_TEST_WORKER_DIR || 'dist', script), ...args], {
    cwd: ui,
    env,
    stdio: script.includes('fileServer') ? ['inherit', 'inherit', 'inherit', 'ipc'] : 'inherit',
    windowsHide: true,
  });
  children.push(child);
  child.on('exit', code => {
    if (!stopping) {
      console.error('Test service exited', script, code);
      void stop();
    }
  });
}
process.on('message', message => {
  if (message?.type === 'aitk-test-shutdown') void stop();
});
process.on('SIGINT', stop);
process.on('SIGTERM', stop);
