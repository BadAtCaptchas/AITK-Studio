import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

function fixture() {
  let message = null;
  const dependencies = {
    'react-global-hooks': { createGlobalState: () => ({ set: value => { message = value; } }) },
    'react/jsx-runtime': {}, 'lucide-react': {},
  };
  const code = ts.transpileModule(fs.readFileSync(new URL('../src/components/WorkflowFeedback.tsx', import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX },
  }).outputText;
  const loaded = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => {
    assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency ${name}`);
    return dependencies[name];
  }, loaded, loaded.exports);
  return { report: loaded.exports.reportWorkflowError, get message() { return message; } };
}

test('recovering a workflow clears its own error and cleanup remains idempotent', () => {
  const f = fixture();
  const clear = f.report('Caption start failed');
  assert.equal(f.message, 'Caption start failed');
  clear();
  assert.equal(f.message, null);
  clear();
  assert.equal(f.message, null);
});

test('an older recovery cannot hide a newer failure, even with identical wording', () => {
  const f = fixture();
  const old = f.report('Save failed');
  const current = f.report('Save failed');
  old();
  assert.equal(f.message, 'Save failed');
  current();
  assert.equal(f.message, null);
  f.report('Another workflow failed');
  current();
  assert.equal(f.message, 'Another workflow failed');
});
