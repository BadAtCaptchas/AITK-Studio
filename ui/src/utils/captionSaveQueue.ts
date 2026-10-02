type CaptionWriter = (imagePath: string, caption: string) => Promise<void>;

/** Keep saves ordered across blur, unmount, and remount of the same image. */
export function createCaptionSaveQueue(write: CaptionWriter): CaptionWriter {
  const pending = new Map<string, Promise<void>>();
  return (imagePath, caption) => {
    const previous = pending.get(imagePath) ?? Promise.resolve();
    const result = previous.then(() => write(imagePath, caption));
    // A failed save must not block a later edit or retry.
    const settled = result.catch(() => undefined);
    pending.set(imagePath, settled);
    void settled.then(() => {
      if (pending.get(imagePath) === settled) pending.delete(imagePath);
    });
    return result;
  };
}
