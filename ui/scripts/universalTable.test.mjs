import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';
import ts from 'typescript';

const require = createRequire(import.meta.url);
const source = ts.transpileModule(fs.readFileSync(new URL('../src/components/UniversalTable.tsx', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
}).outputText;
const module = { exports: {} };
new Function('require', 'module', 'exports', source)(name => {
  if (name === './Loading') return () => null;
  if (name === '@/components/OperatorPrimitives') return { PageNotice: () => null };
  return require(name);
}, module, module.exports);
const UniversalTable = module.exports.default;

function rowKeys(rows) {
  const tree = UniversalTable({ rows, columns: [{ key: 'name', title: 'Name' }], isLoading: false });
  return tree.props.children.props.children.props.children[1].props.children.map(row => row.key);
}

test('deleting or reordering rows keeps React identity attached to the record ID', () => {
  const rows = [{ id: 0, name: 'First' }, { id: 'second', name: 'Second' }, { id: 'third', name: 'Third' }];
  const original = rowKeys(rows);
  assert.deepEqual(rowKeys(rows.slice(1)), original.slice(1));
  assert.deepEqual(rowKeys([rows[2], rows[0]]), [original[2], original[0]]);
  assert.deepEqual(rowKeys([{ name: 'No ID' }]), ['0']);
});
