import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

const require = createRequire(import.meta.url);
const src = fileURLToPath(new URL('../src/', import.meta.url));
const cache = new Map();
export function loadTrainingSource(relative) {
  const filename = path.resolve(src, relative);
  if (cache.has(filename)) return cache.get(filename);
  if (filename.endsWith('.json')) return JSON.parse(fs.readFileSync(filename, 'utf8'));
  const result = { exports: {} };
  cache.set(filename, result.exports);
  const code = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  new Function('require', 'module', 'exports', code)(
    name => {
      if (name === '@/helpers/basic') return { isMac: () => false };
      if (name.startsWith('@/') || name.startsWith('.')) {
        const target = name.startsWith('@/')
          ? path.resolve(src, name.slice(2))
          : path.resolve(path.dirname(filename), name);
        return loadTrainingSource(target.endsWith('.json') ? target : `${target}.ts`);
      }
      return require(name);
    },
    result,
    result.exports,
  );
  return result.exports;
}

export const migration = loadTrainingSource('domain/trainingModelMigration.ts');
const { defaultJobConfig } = loadTrainingSource('app/jobs/new/jobConfig.ts');
export function trainingFixture() {
  const config = structuredClone(defaultJobConfig);
  config.config.name = 'source_run';
  const p = config.config.process[0];
  p.model.arch = 'flux';
  p.model.name_or_path = 'black-forest-labs/FLUX.1-dev';
  p.train.steps = 4200;
  p.train.lr = 0.00037;
  p.train.batch_size = 3;
  p.train.gradient_accumulation = 5;
  p.train.optimizer = 'adamw';
  p.train.optimizer_params = { weight_decay: 0.017, betas: [0.8, 0.95] };
  p.train.phases = [
    { name: 'warmup', steps: 200, lr: 0.0001 },
    { name: 'main', steps: 4000, lr: 0.00037 },
  ];
  p.network.linear = 64;
  p.network.linear_alpha = 48;
  p.datasets[0].folder_path = 'fixture/dataset';
  p.datasets[0].resolution = [640, 960];
  p.sample.samples = [{ prompt: 'my custom trigger in a garden', seed: 97 }];
  return config;
}
