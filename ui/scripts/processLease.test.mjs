import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import crypto from 'node:crypto';
import childProcess from 'node:child_process';
import { once } from 'node:events';
import { promisify } from 'node:util';
import test from 'node:test';
import ts from 'typescript';

function loadSource(relative, dependencies, processObject = process) {
  const code = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const evaluatedModule = { exports: {} };
  new Function('require', 'module', 'exports', 'process', code)(
    name => {
      assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency: ${name}`);
      return dependencies[name];
    },
    evaluatedModule,
    evaluatedModule.exports,
    processObject,
  );
  return evaluatedModule.exports;
}

const nativeBirth = loadSource('../src/server/processBirth.ts', {
  child_process: childProcess,
  util: { promisify },
});

function birthFixture(platform, query, kill = () => assert.fail('unexpected liveness probe')) {
  return loadSource('../src/server/processBirth.ts', {
    child_process: { execFile: query },
    util: { promisify: fn => fn },
  }, { platform, env: {}, kill });
}

test('Windows process birth uses a hidden native query and preserves legacy Unix timestamps', async () => {
  const fixture = birthFixture('win32', async (command, args, options) => {
    assert.equal(command, 'powershell.exe');
    assert.ok(args.includes('-NoProfile'));
    assert.ok(args.includes('-NonInteractive'));
    assert.match(args.at(-1), /Get-Process -Id 123 /);
    assert.equal(options.windowsHide, true);
    assert.equal(options.timeout, 5000);
    return { stdout: '2026-09-21T12:34:56.1234567Z\r\n' };
  });
  const birth = await fixture.getProcessBirth(123);
  assert.equal(birth, Date.parse('2026-09-21T12:34:56.123Z') / 1000);
  assert.equal(fixture.processBirthMatches(birth, birth + 0.0004567), true);
  assert.equal(fixture.processBirthMatches(birth, birth + 1), false);
});

for (const platform of ['linux', 'darwin']) {
  test(`${platform} process birth uses stable locale/timezone and protects fractional legacy timestamps`, async () => {
    const fixture = birthFixture(platform, async (command, args, options) => {
      assert.equal(command, 'ps');
      assert.deepEqual(args, ['-p', '123', '-o', 'lstart=']);
      assert.equal(options.env.LC_ALL, 'C');
      assert.equal(options.env.TZ, 'UTC0');
      return { stdout: 'Mon Sep 21 12:34:56 2026\n' };
    });
    const birth = await fixture.getProcessBirth(123);
    assert.equal(birth, Date.parse('2026-09-21T12:34:56Z') / 1000);
    assert.equal(fixture.processBirthMatches(birth, birth + 0.999), true);
    assert.equal(fixture.processBirthMatches(birth, birth + 2), false);
  });
}

test('only an OS no-such-process result can turn a failed query into an absent owner', async () => {
  const queryError = new Error('query failed');
  for (const code of ['ESRCH', 'EPERM', null]) {
    const fixture = birthFixture('win32', async () => { throw queryError; }, (pid, signal) => {
      assert.equal(pid, 123);
      assert.equal(signal, 0);
      if (code) throw Object.assign(new Error(code), { code });
    });
    if (code === 'ESRCH') assert.equal(await fixture.getProcessBirth(123), null);
    else await assert.rejects(fixture.getProcessBirth(123), error => error === queryError);
  }
});

test('malformed process output and invalid PIDs cannot prove ownership', async () => {
  for (const output of ['', 'garbage', '0', 'absent']) {
    const fixture = birthFixture('win32', async () => ({ stdout: output }), () => {});
    await assert.rejects(fixture.getProcessBirth(123), /Cannot verify process ownership/);
  }
  for (const pid of [0, -1, 1.5, NaN, Infinity, 2 ** 31, '123; exit']) {
    const fixture = birthFixture('win32', async () => assert.fail('queried invalid PID'));
    await assert.rejects(fixture.getProcessBirth(pid), /Invalid process id/);
  }
});

function leaseFixture(birth = nativeBirth) {
  const records = new Map();
  const runtime = {
    get: async key => structuredClone(records.get(key) ?? null),
    compareAndSwap: async (key, version, value) => {
      const previous = records.get(key);
      if ((previous?.version ?? null) !== version) return false;
      records.set(key, { version: (version ?? 0) + 1, value: structuredClone(value) });
      return true;
    },
    delete: async (key, version) => {
      if (records.get(key)?.version !== version) return false;
      return records.delete(key);
    },
  };
  return {
    records,
    ...loadSource('../src/server/processLease.ts', {
      os,
      crypto,
      './db': { db: { runtime } },
      './processBirth': birth,
    }),
  };
}

test('background leases work without a usable Python interpreter and retain live ownership', async t => {
  const previous = process.env.AITK_PYTHON_PATH;
  process.env.AITK_PYTHON_PATH = 'nonexistent-python-for-process-lease-test';
  t.after(() => {
    if (previous === undefined) delete process.env.AITK_PYTHON_PATH;
    else process.env.AITK_PYTHON_PATH = previous;
  });
  const fixture = leaseFixture();
  await fixture.withProcessLease('background:remote-discovery', async lease => {
    await lease.assertOwned();
    fixture.records.get('lease:background:remote-discovery').value.heartbeat = 0;
    await assert.rejects(fixture.acquireProcessLease('background:remote-discovery'), fixture.LeaseBusyError);
  });
  assert.equal(fixture.records.size, 0);
});

test('native birth checks identify a live child and then its exit', async t => {
  const child = childProcess.spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], {
    stdio: 'ignore', windowsHide: true,
  });
  const exited = once(child, 'exit');
  t.after(async () => {
    if (child.exitCode === null && child.signalCode === null) child.kill();
    await exited;
  });
  await once(child, 'spawn');
  const first = await nativeBirth.getProcessBirth(child.pid);
  assert.ok(first > 0);
  assert.ok(Math.abs(first - Date.now() / 1000) < 10);
  assert.equal(await nativeBirth.getProcessBirth(child.pid), first);
  child.kill();
  await exited;
  assert.equal(await nativeBirth.getProcessBirth(child.pid), null);
});

test('lease recovery distinguishes live owners, exited owners, PID reuse, and query failures', async () => {
  for (const actual of [1000, null, 2000, new Error('access denied')]) {
    const fixture = leaseFixture({
      ...nativeBirth,
      getProcessBirth: async pid => {
        if (pid === process.pid) return 3000;
        if (actual instanceof Error) throw actual;
        return actual;
      },
    });
    const prior = { token: 'previous', host: os.hostname(), pid: process.pid + 1, birth: 1000, heartbeat: 0 };
    fixture.records.set('lease:recovery', { version: 1, value: prior });
    if (actual === 1000 || actual instanceof Error) {
      await assert.rejects(fixture.acquireProcessLease('recovery'), actual instanceof Error ? /access denied/ : fixture.LeaseBusyError);
      assert.deepEqual(fixture.records.get('lease:recovery').value, prior);
    } else {
      const lease = await fixture.acquireProcessLease('recovery');
      await lease.assertOwned();
      assert.notEqual(lease.token, prior.token);
      await lease.release();
      assert.equal(fixture.records.size, 0);
    }
  }
});
