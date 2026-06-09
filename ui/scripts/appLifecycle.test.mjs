import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { test } from 'node:test';
import {
  TOOLKIT_ROOT,
  UI_ROOT,
  buildAppCommands,
  collectRuntimePids,
  describeStopFailure,
} from './runtime-processes.mjs';

test('production supervisor starts Next and worker directly', () => {
  const commands = buildAppCommands('start', 8675);
  const ui = commands.find(command => command.label === 'UI');
  const worker = commands.find(command => command.label === 'WORKER');

  assert.equal(ui.command, process.execPath);
  assert.equal(ui.critical, true);
  assert.equal(ui.args[0], path.join(UI_ROOT, 'node_modules', 'next', 'dist', 'bin', 'next'));
  assert.deepEqual(ui.args.slice(1), ['start', '--port', '8675']);

  assert.equal(worker.command, process.execPath);
  assert.equal(worker.critical, true);
  assert.deepEqual(worker.args, [path.join(UI_ROOT, 'dist', 'cron', 'worker.js')]);
});

test('development supervisor starts direct Next and ts-node-dev entrypoints', () => {
  const commands = buildAppCommands('dev', 3000);
  const ui = commands.find(command => command.label === 'UI');
  const worker = commands.find(command => command.label === 'WORKER');

  assert.equal(ui.command, process.execPath);
  assert.equal(ui.args[0], path.join(UI_ROOT, 'node_modules', 'next', 'dist', 'bin', 'next'));
  assert.deepEqual(ui.args.slice(1), ['dev', '--turbopack']);

  assert.equal(worker.command, process.execPath);
  assert.equal(worker.args[0], path.join(UI_ROOT, 'node_modules', 'ts-node-dev', 'lib', 'bin.js'));
  assert.ok(worker.args.includes('--exit-child'));
  assert.ok(worker.args.includes('cron/worker.ts'));
});

test('runtime PID collection excludes training jobs even when they are descendants', () => {
  const processes = [
    {
      pid: 100,
      ppid: 1,
      commandLine: `${process.execPath} ${path.join(UI_ROOT, 'scripts', 'run-app.mjs')} --mode start`,
    },
    {
      pid: 101,
      ppid: 100,
      commandLine: `${process.execPath} ${path.join(UI_ROOT, 'dist', 'cron', 'worker.js')}`,
    },
    {
      pid: 102,
      ppid: 101,
      commandLine: `python ${path.join(TOOLKIT_ROOT, 'run.py')} ${path.join(TOOLKIT_ROOT, 'output', 'job', '.job_config.json')}`,
    },
    {
      pid: 103,
      ppid: 100,
      commandLine: `${process.execPath} ${path.join(UI_ROOT, 'node_modules', 'next', 'dist', 'bin', 'next')} start --port 8675`,
    },
  ];

  const pids = collectRuntimePids(processes, { rootPid: 100 }).sort((a, b) => a - b);

  assert.deepEqual(pids, [100, 101, 103]);
});

test('shutdown implementation does not use taskkill', async () => {
  const sources = await Promise.all([
    readFile(new URL('./run-app.mjs', import.meta.url), 'utf8'),
    readFile(new URL('./runtime-processes.mjs', import.meta.url), 'utf8'),
    readFile(new URL('./restart-ui.mjs', import.meta.url), 'utf8'),
  ]);

  assert.equal(sources.some(source => source.includes('taskkill')), false);
});

test('permission failure message is explicit and does not recommend elevation', () => {
  const message = describeStopFailure({ denied: [1234], remaining: [] });

  assert.match(message, /permission was denied/i);
  assert.match(message, /1234/);
  assert.doesNotMatch(message, /admin|administrator|elevat/i);
});
