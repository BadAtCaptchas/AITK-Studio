'use client';

import { useLayoutEffect, useRef } from 'react';

/** Event handlers and asynchronous work must only observe committed renders. */
export default function useCommittedRef<T>(value: T) {
  const ref = useRef(value);
  useLayoutEffect(() => { ref.current = value; }, [value]);
  return ref;
}
