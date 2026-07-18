export function assignSeriesColors(
  keys: readonly string[],
  palette: readonly string[],
): Record<string, string> {
  if (palette.length === 0) return {};
  return Object.fromEntries(
    keys.map((key, index) => [key, palette[index % palette.length]]),
  );
}
