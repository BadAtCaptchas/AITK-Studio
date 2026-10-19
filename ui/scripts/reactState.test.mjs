import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

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

test('manual phase editor never rewrites saved settings on mount and synchronizes explicit edits', () => {
  const jsx = (type, props) => ({ type, props });
  const renderEditor = load('../src/app/jobs/new/TrainingPhasesEditor.tsx', {
    'react/jsx-runtime': { jsx, jsxs: jsx },
    '@/components/formInputs': { Checkbox: 'checkbox', NumberInput: 'number', SelectInput: 'select', TextInput: 'text' },
    'lucide-react': { Copy: 'copy', Plus: 'plus', Trash2: 'trash' },
  }).default;
  const train = { steps: 1000, optimizer: 'adamw', lr: 0.000007, loss_type: 'mae', phases: [
    { name: 'Saved custom phase', steps: 750, lr: 0.000003 }, { name: 'Detail', steps: 250 },
  ] };
  const before = structuredClone(train);
  const writes = [];
  const render = () => renderEditor({ train, setJobConfig: (value, key) => writes.push({ value, key }) });
  const tree = render();
  render();
  assert.deepEqual(writes, []);
  assert.deepEqual(train, before);
  const nodes = [];
  function visit(node) {
    if (Array.isArray(node)) return node.forEach(visit);
    if (!node || typeof node !== 'object') return;
    nodes.push(node);
    visit(node.props?.children);
  }
  visit(tree);
  const steps = nodes.filter(node => node.type === 'number' && node.props.label === 'Steps');
  assert.equal(steps.length, 2);
  steps[0].props.onChange(600);
  assert.deepEqual(writes.find(write => write.key.endsWith('.phases')).value.map(phase => phase.steps), [600, 250]);
  assert.equal(writes.find(write => write.key.endsWith('.train.steps')).value, 850);
  assert.deepEqual(train, before);
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
