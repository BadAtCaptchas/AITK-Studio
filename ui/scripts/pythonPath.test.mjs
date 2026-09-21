import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { promisify } from 'node:util';
import test from 'node:test';
import ts from 'typescript';

const root = 'C:\\toolkit';
const localPython = path.win32.join(root, '.venv', 'Scripts', 'python.exe');
const source = ts.transpileModule(
  fs.readFileSync(new URL('../src/server/pythonPath.ts', import.meta.url), 'utf8'),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true } },
).outputText;

function fixture({ version = [3, 12], executable = localPython, stdout, env = {} } = {}) {
  const calls = [];
  const files = new Set([path.win32.join(root, '.venv'), localPython, executable]);
  const dependencies = {
    fs: { existsSync: filename => files.has(filename), statSync: () => ({ mtimeMs: 1 }) },
    path: path.win32,
    child_process: {
      execFile() { assert.fail('Unexpected runtime preflight'); },
      spawnSync(candidate) {
        calls.push(candidate);
        return { status: 0, stdout: stdout ?? JSON.stringify({ executable, version }) };
      },
    },
    util: { promisify },
    '../paths': { TOOLKIT_ROOT: root },
  };
  const evaluatedModule = { exports: {} };
  new Function('require', 'module', 'exports', 'process', source)(
    name => {
      assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency: ${name}`);
      return dependencies[name];
    },
    evaluatedModule,
    evaluatedModule.exports,
    { platform: 'win32', env },
  );
  return { ...evaluatedModule.exports, calls };
}

test('Python 3.12 and the supported Python 3.11 profile resolve and cache their interpreter', () => {
  for (const minor of [11, 12]) {
    const state = fixture({ version: [3, minor] });
    assert.equal(state.getToolkitPythonPath(), localPython);
    assert.equal(state.getToolkitPythonPath(), localPython);
    assert.deepEqual(state.calls, [localPython]);
  }
});

test('unsupported versions report the actual interpreter and actionable repair', () => {
  for (const minor of [10, 13, 14]) {
    const state = fixture({ version: [3, minor] });
    assert.throws(() => state.getToolkitPythonPath(), error => {
      assert.ok(error.message.includes(`Unsupported Python 3.${minor}`));
      assert.ok(error.message.includes(localPython));
      assert.match(error.message, /manager environment sync/);
      assert.match(error.message, /AITK_PYTHON_PATH/);
      return true;
    });
  }
});

test('an explicit interpreter remains authoritative and its path appears in failures', () => {
  const executable = 'C:\\custom-python\\python.exe';
  const state = fixture({ version: [3, 13], executable, env: { AITK_PYTHON_PATH: executable } });
  assert.throws(() => state.getToolkitPythonPath(), error => error.message.includes(executable));
  assert.deepEqual(state.calls, [executable]);
});

test('invalid probe output is distinguished from an unsupported Python version', () => {
  for (const stdout of [
    'not json', 'null', '[]',
    JSON.stringify({ executable: localPython, version: [3, '12'] }),
    JSON.stringify({ executable: localPython, version: [3] }),
    JSON.stringify({ executable: localPython, version: [3, 12.5] }),
    JSON.stringify({ executable: '', version: [3, 12] }),
  ]) {
    const state = fixture({ stdout });
    assert.throws(() => state.getToolkitPythonPath(), error => {
      assert.match(error.message, /Cannot read Python interpreter details/);
      assert.ok(error.message.includes(localPython));
      return true;
    });
  }
});
