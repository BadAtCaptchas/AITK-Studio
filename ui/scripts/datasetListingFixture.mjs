import fs from 'node:fs';
import { createRequire } from 'node:module';
import ts from 'typescript';

// Run the current source against the real compiled storage/security dependencies.
export function loadDatasetListing() {
  const source = fs.readFileSync(new URL('../src/server/datasetListing.ts', import.meta.url), 'utf8');
  const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(createRequire(new URL('../dist/src/server/datasetListing.js', import.meta.url)), module, module.exports);
  return module.exports;
}
