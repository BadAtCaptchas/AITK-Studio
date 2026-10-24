/** Join a configured storage root and a single dataset name in browser code.
 * The server still resolves and validates every path before accessing storage. */
export function datasetDisplayPath(root: string, name: string): string {
  const separator = root.includes('\\') ? '\\' : '/';
  return `${root.replace(/[\\/]+$/, '')}${separator}${name}`;
}
