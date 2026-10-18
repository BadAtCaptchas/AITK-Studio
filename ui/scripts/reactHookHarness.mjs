// A small hook runner for the existing dependency-injected request tests.
// Model lazy initialization, render-phase resets, commit ordering and cleanup.
export function createHookHarness() {
  const slots = [], committedEffects = [];
  let cursor = 0, needsRender = false, pendingEffects = [];
  const same = (left, right) => left && right && left.length === right.length && left.every((value, i) => Object.is(value, right[i]));
  const effect = (callback, dependencies, layout) => {
    const index = cursor++;
    if (!same(committedEffects[index]?.dependencies, dependencies)) pendingEffects.push({ index, callback, dependencies, layout });
  };
  const react = {
    useState(initial) {
      const index = cursor++;
      if (!(index in slots)) slots[index] = typeof initial === 'function' ? initial() : initial;
      return [slots[index], action => {
        const value = typeof action === 'function' ? action(slots[index]) : action;
        if (!Object.is(slots[index], value)) { slots[index] = value; needsRender = true; }
      }];
    },
    useRef(initial) { const index = cursor++; return slots[index] ??= { current: initial }; },
    useCallback(callback, dependencies) {
      const index = cursor++;
      if (!same(slots[index]?.dependencies, dependencies)) slots[index] = { value: callback, dependencies };
      return slots[index].value;
    },
    useMemo(callback, dependencies) {
      const index = cursor++;
      if (!same(slots[index]?.dependencies, dependencies)) slots[index] = { value: callback(), dependencies };
      return slots[index].value;
    },
    useEffect: (callback, dependencies) => effect(callback, dependencies, false),
    useLayoutEffect: (callback, dependencies) => effect(callback, dependencies, true),
  };
  return {
    react,
    render(callback) {
      for (let attempt = 0; attempt < 50; attempt++) {
        cursor = 0; needsRender = false; pendingEffects = [];
        const result = callback();
        if (needsRender) continue;
        for (const { index } of pendingEffects) committedEffects[index]?.cleanup?.();
        for (const next of pendingEffects.sort((a, b) => Number(b.layout) - Number(a.layout))) {
          committedEffects[next.index] = { dependencies: next.dependencies, cleanup: next.callback() };
        }
        if (!needsRender) return result;
      }
      throw new Error('Hook render did not settle');
    },
    cleanup() { committedEffects.forEach(effect => effect?.cleanup?.()); },
  };
}
