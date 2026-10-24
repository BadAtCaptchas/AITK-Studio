const PREFIX = 'aitk-lora://remote/';
export function makeRemoteLoraRef(workerID: string, remotePath: string): string {
  return PREFIX + encodeURIComponent(workerID) + '/' + encodeURIComponent(remotePath);
}
export function parseRemoteLoraRef(value: unknown): { workerID: string; path: string } | null {
  if (typeof value !== 'string' || !value.startsWith(PREFIX)) return null;
  const parts = value.slice(PREFIX.length).split('/');
  if (parts.length !== 2) throw new Error('Invalid remote LoRA reference.');
  const [workerID, path] = parts.map(decodeURIComponent);
  if (!/^[a-zA-Z0-9_-]{1,128}$/.test(workerID) || !path || path.length > 4096 || path.includes('\0') || path.startsWith(PREFIX))
    throw new Error('Invalid remote LoRA reference.');
  return { workerID, path };
}
