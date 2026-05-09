import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const childProcess = require('child_process');
const tensorBoardModulePath = require.resolve('../dist/src/server/tensorboard.js');
const originalSpawnSync = childProcess.spawnSync;
const originalEnableEnv = process.env.AITK_ENABLE_TENSORBOARD;

function restoreEnvironment() {
  childProcess.spawnSync = originalSpawnSync;
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
