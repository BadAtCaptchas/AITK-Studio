import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import vm from 'node:vm';
import { buildAppCommands } from './runtime-processes.mjs';

test('compiled startup does not resolve development-only dependencies', () => {
  const construct = vm.runInNewContext(`(${buildAppCommands.toString()})`, {
    path,
    process,
    UI_ROOT: '/fixture/ui',
    require: {
      resolve() {
        throw new Error('No development dependencies');
      },
    },
  });
  assert.equal(construct('start', 8675).length, 3);
  assert.throws(() => construct('dev', 3000), /Development startup requires/);
});
