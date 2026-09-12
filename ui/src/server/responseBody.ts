/** Read a response with an actual byte limit, including chunked/error bodies. */
export async function readBoundedResponseText(response: Response, maxBytes: number, truncate = false): Promise<string> {
  if (!response.body) return '';
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let bytes = 0;
  let text = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return text + decoder.decode();
      bytes += value.byteLength;
      if (bytes > maxBytes) {
        if (truncate)
          return (
            text +
            decoder.decode(value.subarray(0, Math.max(0, maxBytes - (bytes - value.byteLength)))) +
            '\n[truncated]'
          );
        throw new Error(`Response exceeds the ${maxBytes} byte limit`);
      }
      text += decoder.decode(value, { stream: true });
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

/** Own cancellation until the caller finishes or cancels the response body. */
export function ownResponseBody(
  response: Response,
  signal: AbortSignal,
  dispose: () => void,
  idleMs = 60_000,
): Response {
  if (!response.body) {
    dispose();
    return response;
  }
  const reader = response.body.getReader();
  let finished = false;
  let onAbort: () => void = () => undefined;
  let idle: ReturnType<typeof setTimeout> | undefined;
  const finish = () => {
    if (finished) return;
    finished = true;
    clearTimeout(idle);
    signal.removeEventListener('abort', onAbort);
    dispose();
  };
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      onAbort = () => {
        if (finished) return;
        controller.error(signal.reason ?? new Error('Remote request aborted'));
        void reader.cancel(signal.reason).catch(() => undefined);
        finish();
      };
      signal.addEventListener('abort', onAbort, { once: true });
      if (signal.aborted) onAbort();
    },
    async pull(controller) {
      try {
        idle = setTimeout(() => {
          if (finished) return;
          const error = new Error('Remote response body idle deadline exceeded');
          controller.error(error);
          void reader.cancel(error).catch(() => undefined);
          finish();
        }, idleMs);
        idle.unref?.();
        const { value, done } = await reader.read();
        clearTimeout(idle);
        if (finished) return;
        if (done) {
          finish();
          controller.close();
        } else controller.enqueue(value);
      } catch (error) {
        if (finished) return;
        finish();
        controller.error(error);
      }
    },
    async cancel(reason) {
      finish();
      await reader.cancel(reason).catch(() => undefined);
    },
  });
  return new Response(stream, { status: response.status, statusText: response.statusText, headers: response.headers });
}
