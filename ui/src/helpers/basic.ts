export const isMac = (): boolean =>
  typeof window !== 'undefined' ? window.server_platform === 'darwin' : process.platform === 'darwin';
