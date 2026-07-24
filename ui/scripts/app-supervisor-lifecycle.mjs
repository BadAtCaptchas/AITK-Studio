export async function runStartupWithCleanup(start, cleanup) {
  try {
    await start();
    return true;
  } catch (error) {
    await cleanup(error);
    return false;
  }
}

export async function stopManagedChildren(entries, stopChild, onError = () => undefined) {
  for (const entry of entries) {
    try {
      await stopChild(entry);
    } catch (error) {
      onError(error, entry);
    }
  }
}
