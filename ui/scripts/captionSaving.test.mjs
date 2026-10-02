import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

function loadSource(relative, dependencies, overrides = {}) {
  const code = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
      jsx: ts.JsxEmit.React,
      esModuleInterop: true,
    },
  }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', 'console', code)(
    name => {
      assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency: ${name}`);
      return dependencies[name];
    }, module, module.exports, overrides.console ?? console,
  );
  return module.exports;
}

const { createCaptionSaveQueue } = loadSource('../src/utils/captionSaveQueue.ts', {});
const settle = () => new Promise(resolve => setImmediate(resolve));

function fixture() {
  const slots = [], effects = [], requests = [], cached = new Map(), errors = [];
  let cursor = 0, pendingEffects = [], fetchedCaption = 'original';
  const sameDependencies = (a, b) => a && b && a.length === b.length && a.every((value, i) => value === b[i]);
  const react = {
    createElement: (type, props, ...children) => ({ type, props: { ...props, children } }),
    useState(initial) {
      const i = cursor++;
      if (!(i in slots)) slots[i] = typeof initial === 'function' ? initial() : initial;
      return [slots[i], value => { slots[i] = typeof value === 'function' ? value(slots[i]) : value; }];
    },
    useRef(initial) {
      const i = cursor++;
      if (!(i in slots)) slots[i] = { current: initial };
      return slots[i];
    },
    useCallback(callback, deps) {
      const i = cursor++;
      if (!slots[i] || !sameDependencies(slots[i].deps, deps)) slots[i] = { callback, deps };
      return slots[i].callback;
    },
    useEffect(callback, deps) {
      const i = cursor++, previous = effects[i];
      if (!previous || !sameDependencies(previous.deps, deps)) pendingEffects.push(() => {
        previous?.cleanup?.();
        effects[i] = { deps, cleanup: callback() };
      });
    },
  };
  const component = loadSource('../src/components/DatasetImageCard.tsx', {
    react: { __esModule: true, default: react, ...react },
    'react-icons/fa': { FaTrashAlt: () => null, FaPlay: () => null },
    './ConfirmModal': { openConfirm() {} },
    classnames: { __esModule: true, default: () => '' },
    '@/utils/api': { apiClient: { post(url, body) {
      let resolve, reject;
      const result = new Promise((yes, no) => { resolve = yes; reject = no; });
      requests.push({ url, body, resolve, reject });
      return result;
    } } },
    './AudioPlayer': { __esModule: true, default: () => null },
    '@/utils/basic': { isVideo: () => false, isAudio: () => false },
    '@/hooks/useCaptionBatch': {
      __esModule: true,
      default: () => ({ caption: fetchedCaption, isLoaded: true }),
      setCachedCaption: (path, caption) => cached.set(path, caption),
    },
    '@/utils/media': { getDisplayPath: x => x, getMediaUrl: x => x },
    '@/utils/captionSaveQueue': { createCaptionSaveQueue },
  }, { console: { ...console, error: (...args) => errors.push(args) } }).default;
  function render() {
    cursor = 0; pendingEffects = [];
    const tree = component({ imageUrl: '/fixture/image.png', alt: '', isAutoCaptioning: false });
    pendingEffects.forEach(callback => callback());
    return tree;
  }
  function find(tree, type) {
    if (!tree || typeof tree !== 'object') return null;
    if (tree.type === type) return tree;
    return tree.props?.children?.flat(Infinity).map(child => find(child, type)).find(Boolean);
  }
  render();
  return {
    requests, cached, errors,
    edit(value) { find(render(), 'textarea').props.onChange({ target: { value } }); render(); },
    save() { find(render(), 'form').props.onBlur(); },
    refresh(value) { fetchedCaption = value; render(); },
    caption() { return find(render(), 'textarea').props.value; },
    unmount() { for (const effect of effects) effect?.cleanup?.(); },
  };
}

test('an older save acknowledgement leaves newer text dirty and unmount saves it', async () => {
  const card = fixture();
  card.edit('first edit'); card.save(); await settle();
  card.edit('newer edit');
  card.requests[0].resolve({}); await settle();
  card.refresh('stale fetched caption');
  assert.equal(card.caption(), 'newer edit');
  card.unmount(); await settle();
  assert.deepEqual(card.requests.map(request => request.body.caption), ['first edit', 'newer edit']);
  card.requests[1].resolve({}); await settle();
  assert.equal(card.cached.get('/fixture/image.png'), 'newer edit');
});

test('blur and unmount saves are serialized so an earlier write cannot finish last', async () => {
  const card = fixture();
  card.edit('A'); card.save(); await settle();
  card.edit('B'); card.save();
  card.edit('C'); card.unmount(); await settle();
  assert.equal(card.requests.length, 1);
  for (const [index, caption] of ['A', 'B', 'C'].entries()) {
    assert.equal(card.requests[index].body.caption, caption);
    card.requests[index].resolve({}); await settle();
  }
  assert.equal(card.requests.length, 3);
  assert.equal(card.cached.get('/fixture/image.png'), 'C');
});

test('reverting to the original caption while a save is pending still persists the revert', async () => {
  const card = fixture();
  card.edit('temporary edit'); card.save(); await settle();
  card.edit('original'); card.unmount();
  card.requests[0].resolve({}); await settle();
  assert.equal(card.requests[1].body.caption, 'original');
  card.requests[1].resolve({}); await settle();
  assert.equal(card.cached.get('/fixture/image.png'), 'original');
});

test('a failed request leaves edits retryable and does not block queued newer text', async () => {
  const card = fixture();
  card.edit('A'); card.save(); await settle();
  card.edit('B'); card.unmount();
  card.requests[0].reject(new Error('temporary failure')); await settle();
  assert.equal(card.errors.length, 1);
  assert.equal(card.requests[1].body.caption, 'B');
  card.requests[1].resolve({}); await settle();
  assert.equal(card.cached.get('/fixture/image.png'), 'B');
});

test('different image saves run independently while remounts share their image queue', async () => {
  const writes = [];
  const save = createCaptionSaveQueue((path, caption) => new Promise(resolve => writes.push({ path, caption, resolve })));
  const first = save('one', 'old card');
  const second = save('one', 'remounted card');
  const other = save('two', 'unrelated image');
  await settle();
  assert.deepEqual(writes.map(write => write.path), ['one', 'two']);
  writes[0].resolve(); await settle();
  assert.equal(writes[2].caption, 'remounted card');
  writes[1].resolve(); writes[2].resolve();
  await Promise.all([first, second, other]);
});
