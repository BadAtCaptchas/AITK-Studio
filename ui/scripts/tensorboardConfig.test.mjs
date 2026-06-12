import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';

const require = createRequire(import.meta.url);
const childProcess = require('child_process');
const tensorBoardModulePath = require.resolve('../dist/src/server/tensorboard.js');
const originalSpawnSync = childProcess.spawnSync;
const originalExistsSync = fs.existsSync;
const originalEnableEnv = process.env.AITK_ENABLE_TENSORBOARD;

function restoreEnvironment() {
  childProcess.spawnSync = originalSpawnSync;
  fs.existsSync = originalExistsSync;
  if (originalEnableEnv === undefined) {
    delete process.env.AITK_ENABLE_TENSORBOARD;
  } else {
    process.env.AITK_ENABLE_TENSORBOARD = originalEnableEnv;
  }
  delete require.cache[tensorBoardModulePath];
}

function loadTensorBoardWithProbeStatus(status) {
  delete require.cache[tensorBoardModulePath];
  let probeCount = 0;

  childProcess.spawnSync = () => {
    probeCount += 1;
    return { status };
  };

  return {
    tensorBoard: require(tensorBoardModulePath),
    getProbeCount: () => probeCount,
  };
}

afterEach(restoreEnvironment);

test('auto-enables TensorBoard when env is unset and package probe succeeds', () => {
  delete process.env.AITK_ENABLE_TENSORBOARD;

  const { tensorBoard, getProbeCount } = loadTensorBoardWithProbeStatus(0);

  assert.equal(tensorBoard.isTensorBoardEnabled(), true);
  assert.equal(getProbeCount(), 1);
});

test('auto-disables TensorBoard when env is unset and package probe fails', () => {
  delete process.env.AITK_ENABLE_TENSORBOARD;

  const { tensorBoard, getProbeCount } = loadTensorBoardWithProbeStatus(1);

  assert.equal(tensorBoard.isTensorBoardEnabled(), false);
  assert.equal(getProbeCount(), 1);
});

test('explicit false disables TensorBoard without probing package availability', () => {
  process.env.AITK_ENABLE_TENSORBOARD = '0';

  const { tensorBoard, getProbeCount } = loadTensorBoardWithProbeStatus(0);

  assert.equal(tensorBoard.isTensorBoardEnabled(), false);
  assert.equal(getProbeCount(), 0);
});

test('explicit true enables TensorBoard without probing package availability', () => {
  process.env.AITK_ENABLE_TENSORBOARD = '1';

  const { tensorBoard, getProbeCount } = loadTensorBoardWithProbeStatus(1);

  assert.equal(tensorBoard.isTensorBoardEnabled(), true);
  assert.equal(getProbeCount(), 0);
});

test('TensorBoard launch uses pythonw on Windows when resolving bare python.exe', () => {
  const { tensorBoard } = loadTensorBoardWithProbeStatus(0);

  assert.equal(tensorBoard.getTensorBoardLaunchPythonPath('python.exe', 'win32'), 'pythonw.exe');
});

test('TensorBoard launch uses sibling pythonw in a Windows virtualenv', () => {
  const { tensorBoard } = loadTensorBoardWithProbeStatus(0);
  const pythonPath = path.win32.join('C:\\toolkit', '.venv', 'Scripts', 'python.exe');
  const pythonwPath = path.win32.join('C:\\toolkit', '.venv', 'Scripts', 'pythonw.exe');
  fs.existsSync = target => target === pythonwPath || originalExistsSync(target);

  assert.equal(tensorBoard.getTensorBoardLaunchPythonPath(pythonPath, 'win32'), pythonwPath);
});

test('TensorBoard launch keeps console Python outside Windows', () => {
  const { tensorBoard } = loadTensorBoardWithProbeStatus(0);

  assert.equal(tensorBoard.getTensorBoardLaunchPythonPath('/venv/bin/python', 'linux'), '/venv/bin/python');
});

test('stopTensorBoard is safe when no managed process is running', async () => {
  const { tensorBoard } = loadTensorBoardWithProbeStatus(0);

  await assert.doesNotReject(() => tensorBoard.stopTensorBoard());
});

test('TensorBoard launch is managed by the worker process', () => {
  const source = fs.readFileSync(tensorBoardModulePath, 'utf8');

  assert.doesNotMatch(source, /detached:\s*true/);
  assert.doesNotMatch(source, /\.unref\(/);
});
