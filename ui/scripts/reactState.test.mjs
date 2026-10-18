import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import { createHookHarness } from './reactHookHarness.mjs';

function load(relative, dependencies = {}) {
  const code = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX },
  }).outputText;
  const loadedModule = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => {
    assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency: ${name}`);
    return dependencies[name];
  }, loadedModule, loadedModule.exports);
  return loadedModule.exports;
}

function hookFixture() {
  const slots = [];
  let cursor = 0, changed = false;
  const react = {
    useState(initial) {
      const index = cursor++;
      if (!(index in slots)) slots[index] = typeof initial === 'function' ? initial() : initial;
      return [slots[index], action => {
        const next = typeof action === 'function' ? action(slots[index]) : action;
        if (!Object.is(next, slots[index])) { changed = true; slots[index] = next; }
      }];
    },
    useCallback(callback, dependencies) {
      const index = cursor++;
      const previous = slots[index];
      if (!previous || dependencies.some((value, i) => !Object.is(value, previous.dependencies[i]))) {
        slots[index] = { callback, dependencies };
      }
      return slots[index].callback;
    },
  };
  const renderHook = load('../src/hooks/useScopedState.ts', { react }).default;
  return (scope, initial) => {
    for (let attempt = 0; attempt < 10; attempt++) {
      cursor = 0; changed = false;
      const result = renderHook(scope, initial);
      if (!changed) return result;
    }
    throw new Error('State reset did not settle');
  };
}

test('resource changes reset before rendering and late responses cannot overwrite a new visit', () => {
  const render = hookFixture();
  const [, finishFirstA] = render('A', 'loading A');
  finishFirstA('first response');
  const [first, sameSetter] = render('A', 'unused initializer');
  assert.equal(first, 'first response');
  assert.equal(sameSetter, finishFirstA);
  assert.equal(render('B', 'loading B')[0], 'loading B');
  finishFirstA('late A response');
  assert.equal(render('B', 'loading B')[0], 'loading B');
  const [, finishSecondA] = render('A', 'new A visit');
  finishFirstA('response from abandoned A visit');
  assert.equal(render('A', '')[0], 'new A visit');
  finishSecondA(previous => previous + ' completed');
  assert.equal(render('A', '')[0], 'new A visit completed');
});

test('resource initializers run only when the resource identity changes', () => {
  const render = hookFixture();
  let reads = 0;
  const initial = () => ++reads;
  assert.equal(render('A', initial)[0], 1);
  assert.equal(render('A', initial)[0], 1);
  assert.equal(render('B', initial)[0], 2);
  assert.equal(reads, 2);
});

test('architecture changes reapply the same fallback profile without overwriting ordinary parent edits', () => {
  const harness = createHookHarness();
  const fallback = { id: 'fallback', name: 'Fallback', train: { lr: 0.001 }, phases: [] };
  const specialized = { id: 'specialized', name: 'Specialized', modelArchs: ['A'], phases: [] };
  const jsx = (type, props) => ({ type, props });
  const renderEditor = load('../src/app/jobs/new/TrainingPhasesEditor.tsx', {
    react: harness.react,
    'react/jsx-runtime': { jsx, jsxs: jsx },
    '@/hooks/useStoredString': { default: () => ['[]', () => {}] },
    '@/components/formInputs': { Checkbox: 'checkbox', NumberInput: 'number', SelectInput: 'select', TextInput: 'text' },
    'lucide-react': { Copy: 'copy', Plus: 'plus', Save: 'save', Trash2: 'trash' },
    './autoTrainingProfiles': {
      builtInAutoTrainingProfiles: [fallback, specialized],
      isAutoTrainingProfileCompatible: (profile, arch) => !profile.modelArchs || profile.modelArchs.includes(arch),
    },
  }).default;
  const writes = [];
  const render = arch => harness.render(() => renderEditor({
    train: { auto_train: true, phases: [] },
    currentArch: arch,
    defaultAutoTrainingProfileId: 'specialized',
    setJobConfig: (value, key) => writes.push({ value, key }),
  }));
  const fallbackApplications = () => writes.filter(write => write.key === 'config.process[0].train.lr').length;
  render('A');
  assert.equal(fallbackApplications(), 0);
  render('B');
  assert.equal(fallbackApplications(), 1);
  render('B');
  assert.equal(fallbackApplications(), 1, 'ordinary parent renders must not reapply a profile');
  render('A');
  render('B');
  assert.equal(fallbackApplications(), 2, 'the same fallback must be applied on each incompatible switch');
  harness.cleanup();
});

const { createCaptionCache } = load('../src/components/dataset-image-studio/captionCache.ts');
test('caption scans publish new snapshots without mutating an editor render or emitting unchanged entries', () => {
  const cache = createCaptionCache();
  let changes = 0;
  const unsubscribe = cache.subscribe(() => changes++);
  const empty = cache.getSnapshot();
  cache.set('image-a', { caption: 'draft', saved: 'original', loaded: true });
  const editing = cache.getSnapshot();
  assert.equal(empty.entries.size, 0);
  cache.set('image-a', { caption: 'draft', saved: 'original', loaded: true });
  assert.equal(cache.getSnapshot(), editing);
  assert.equal(changes, 1);
  cache.set('image-a', { caption: 'new draft', saved: 'original', loaded: true });
  assert.equal(editing.entries.get('image-a').caption, 'draft');
  assert.equal(cache.get('image-a').caption, 'new draft');
  cache.delete('image-a');
  assert.equal(cache.getSnapshot().entries.size, 0);
  assert.equal(editing.entries.size, 1);
  assert.equal(cache.delete('missing'), false);
  assert.equal(changes, 3);
  unsubscribe();
  cache.set('image-b', { caption: '', saved: '', loaded: true });
  assert.equal(changes, 3);
});
