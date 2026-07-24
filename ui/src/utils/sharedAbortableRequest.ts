type SharedRequestFactory<Key, Value> = (key: Key, signal: AbortSignal) => Promise<Value>;

type SharedRequestEntry<Value> = {
  controller: AbortController;
  promise: Promise<Value>;
  subscribers: number;
  settled: boolean;
};

function createAbortError() {
  const error = new Error('Request aborted');
  error.name = 'AbortError';
  return error;
}

function waitForSharedRequest<Value>(promise: Promise<Value>, signal?: AbortSignal) {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(createAbortError());

  return new Promise<Value>((resolve, reject) => {
    let finished = false;
    const cleanup = () => signal.removeEventListener('abort', onAbort);
    const onAbort = () => {
      if (finished) return;
      finished = true;
      cleanup();
      reject(createAbortError());
    };

    signal.addEventListener('abort', onAbort, { once: true });
    promise.then(
      value => {
        if (finished) return;
        finished = true;
        cleanup();
        resolve(value);
      },
      error => {
        if (finished) return;
        finished = true;
        cleanup();
        reject(error);
      },
    );
  });
}

/**
 * Deduplicate an in-flight request while giving each caller its own abortable
 * subscription. The shared transport is cancelled only after its final
 * subscriber leaves.
 */
export class SharedAbortableRequestPool<Key, Value> {
  private readonly entries = new Map<Key, SharedRequestEntry<Value>>();

  constructor(private readonly factory: SharedRequestFactory<Key, Value>) {}

  subscribe(key: Key, signal?: AbortSignal): Promise<Value> {
    if (signal?.aborted) return Promise.reject(createAbortError());

    let entry = this.entries.get(key);
    if (!entry) {
      const controller = new AbortController();
      const promise = Promise.resolve().then(() => this.factory(key, controller.signal));
      entry = {
        controller,
        promise,
        subscribers: 0,
        settled: false,
      };
      this.entries.set(key, entry);

      const markSettled = () => {
        entry!.settled = true;
        if (this.entries.get(key) === entry) {
          this.entries.delete(key);
        }
      };
      void promise.then(markSettled, markSettled);
    }

    entry.subscribers += 1;
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      entry!.subscribers = Math.max(0, entry!.subscribers - 1);
      if (entry!.subscribers === 0 && !entry!.settled) {
        if (this.entries.get(key) === entry) {
          this.entries.delete(key);
        }
        entry!.controller.abort();
      }
    };

    return waitForSharedRequest(entry.promise, signal).finally(release);
  }
}
