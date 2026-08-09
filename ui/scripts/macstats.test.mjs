import assert from 'node:assert/strict';
import test from 'node:test';
import { firstFanRpm, numberFromRecord, parseMacstatsModule } from '../dist/src/server/macstats.js';

const completeModule = {
  getCpuDataSync() {},
  getFanDataSync() {},
  getGpuDataSync() {},
  getPowerDataSync() {},
  getRAMUsageSync() {},
};

test('macstats loader only accepts the complete native API', () => {
  assert.equal(parseMacstatsModule(null), null);
  assert.equal(parseMacstatsModule({ getCpuDataSync() {} }), null);
  assert.equal(parseMacstatsModule(completeModule), completeModule);
});

test('macstats values are narrowed and non-finite values are rejected', () => {
  assert.equal(numberFromRecord({ temperature: 42 }, 'temperature'), 42);
  assert.equal(numberFromRecord({ temperature: Number.NaN }, 'temperature'), 0);
  assert.equal(numberFromRecord({ temperature: '42' }, 'temperature'), 0);
  assert.equal(firstFanRpm({ left: { rpm: 1200 }, right: { rpm: 900 } }), 1200);
  assert.equal(firstFanRpm({ left: { rpm: 'fast' } }), 0);
});
