export const TELEMETRY_ENABLED_SETTING_KEY = 'TELEMETRY_ENABLED';

const TELEMETRY_TRUE_VALUES = new Set(['1', 'true', 'yes', 'on', 'enabled']);

export function normalizeTelemetrySetting(value: unknown): boolean {
  if (typeof value === 'boolean') return value;
  if (typeof value !== 'string') return false;
  return TELEMETRY_TRUE_VALUES.has(value.trim().toLowerCase());
}

export function telemetryChildProcessEnv(enabled: boolean): Record<string, string> {
  return {
    AITK_TELEMETRY_ENABLED: enabled ? '1' : '0',
    DISABLE_TELEMETRY: enabled ? 'NO' : 'YES',
    HF_HUB_DISABLE_TELEMETRY: enabled ? '0' : '1',
  };
}
