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
import {
  runStartupWithCleanup,
  stopManagedChildren,
} from './app-supervisor-lifecycle.mjs';

test('production supervisor starts the front file server and worker directly', () => {
  const commands = buildAppCommands('start', 8675);
  const ui = commands.find(command => command.label === 'UI');
  const worker = commands.find(command => command.label === 'WORKER');

  assert.equal(ui.command, process.execPath);
  assert.equal(ui.critical, true);
  assert.equal(ui.args[0], path.join(UI_ROOT, 'dist', 'cron', 'fileServer.js'));
  assert.deepEqual(ui.args.slice(1), ['start', '--port', '8675']);

  assert.equal(worker.command, process.execPath);
  assert.equal(worker.critical, true);
  assert.deepEqual(worker.args, [path.join(UI_ROOT, 'dist', 'cron', 'worker.js')]);
});

test('development supervisor starts file server and worker ts-node-dev entrypoints', () => {
  const commands = buildAppCommands('dev', 3000);
  const ui = commands.find(command => command.label === 'UI');
  const worker = commands.find(command => command.label === 'WORKER');

  assert.equal(ui.command, process.execPath);
  assert.match(ui.args[0].replace(/\\/g, '/'), /ts-node\/dist\/bin\.js$/);
  assert.equal(ui.ipc, true);
  assert.ok(ui.args.includes('cron/fileServer.ts'));
  assert.deepEqual(ui.args.slice(-4), ['cron/fileServer.ts', 'dev', '--port', '3000']);

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
      commandLine: `${process.execPath} ${path.join(UI_ROOT, 'dist', 'cron', 'fileServer.js')} start --port 8675`,
    },
    {
      pid: 104,
      ppid: 103,
      commandLine: `${process.execPath} ${path.join(UI_ROOT, 'node_modules', 'next', 'dist', 'bin', 'next')} start --port 8675`,
    },
  ];

  const pids = collectRuntimePids(processes, { rootPid: 100 }).sort((a, b) => a - b);

  assert.deepEqual(pids, [100, 101, 103, 104]);
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

test('startup failures clean every child accumulated before a spawn or runtime-write failure', async () => {
  for (const failureStage of ['partial-spawn', 'runtime-write']) {
    const children = [];
    const stopped = [];
    const cleanupErrors = [];

    const started = await runStartupWithCleanup(
      async () => {
        children.push({ spec: { label: 'WORKER' }, child: { id: 'worker' } });
        if (failureStage === 'partial-spawn') {
          throw new Error('UI failed to spawn');
        }
        children.push({ spec: { label: 'UI' }, child: { id: 'file-server' } });
        children.push({ spec: { label: 'UPDATER' }, child: { id: 'updater' } });
        throw new Error('runtime marker write failed');
      },
      async () => {
        await stopManagedChildren(
          children,
          async entry => {
            stopped.push(entry.child.id);
            if (entry.spec.label === 'UI') throw new Error('simulated stop failure');
          },
          (error, entry) => cleanupErrors.push([entry.spec.label, error.message]),
        );
      },
    );

    assert.equal(started, false);
    assert.deepEqual(
      stopped,
      failureStage === 'partial-spawn' ? ['worker'] : ['worker', 'file-server', 'updater'],
    );
    assert.deepEqual(
      cleanupErrors,
      failureStage === 'runtime-write' ? [['UI', 'simulated stop failure']] : [],
    );
  }
});
