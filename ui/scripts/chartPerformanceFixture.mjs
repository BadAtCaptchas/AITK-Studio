import fs from 'node:fs';
import ts from 'typescript';

export function loadTs(relative) {
  const module = { exports: {} };
  const source = fs.readFileSync(new URL(relative, import.meta.url), 'utf8');
  new Function('module', 'exports', ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText)(module, module.exports);
  return module.exports;
}

export const snapshot = loadTs('../src/utils/metricSnapshot.ts');
export const smoothing = loadTs('../src/utils/lossSmoothing.ts');
export const graphSource = fs.readFileSync(new URL('../src/components/JobLossGraph.tsx', import.meta.url), 'utf8');

// Exercise the actual pure chart builder without React/browser dependency mocks.
export function loadChart(source = graphSource, ema = smoothing.emaWithNulls) {
  const parsed = ts.createSourceFile('graph.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const names = new Set(['clamp', 'emaPass', 'emaWithNulls', 'sortedNumericPoints', 'cleanLabel', 'colorWithAlpha', 'buildChartData']);
  const fragments = parsed.statements.filter(statement =>
    (ts.isFunctionDeclaration(statement) && names.has(statement.name?.text)) ||
    (ts.isVariableStatement(statement) && statement.declarationList.declarations.every(declaration => /^[A-Z_]+$/.test(declaration.name.getText(parsed)))),
  ).map(statement => statement.getText(parsed)).join('\n');
  const compiled = ts.transpileModule(fragments, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
  return new Function('emaWithNulls', `${compiled}\nreturn buildChartData;`)(ema);
}
