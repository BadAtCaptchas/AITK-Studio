'use client';

import { useCallback, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';

/** State owned by one resource identity. Reset before its children render, and
 * ignore updates from callbacks belonging to a previous resource. */
export default function useScopedState<T>(scope: unknown, initialValue: T | (() => T)): [T, Dispatch<SetStateAction<T>>] {
  const [snapshot, setSnapshot] = useState(() => ({
    scope,
    owner: {},
    value: typeof initialValue === 'function' ? (initialValue as () => T)() : initialValue,
  }));
  let current = snapshot;
  if (snapshot.scope !== scope) {
    current = { scope, owner: {}, value: typeof initialValue === 'function' ? (initialValue as () => T)() : initialValue };
    setSnapshot(current);
  }
  const owner = current.owner;
  const setValue = useCallback<Dispatch<SetStateAction<T>>>(action => {
    setSnapshot(previous => {
      if (previous.owner !== owner) return previous;
      const value = typeof action === 'function' ? (action as (value: T) => T)(previous.value) : action;
      return Object.is(value, previous.value) ? previous : { ...previous, value };
    });
  }, [owner]);
  return [current.value, setValue];
}
