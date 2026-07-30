import { db } from './db';
import { normalizeTelemetrySetting, TELEMETRY_ENABLED_SETTING_KEY } from '../utils/telemetry';

export async function isTelemetryEnabled(): Promise<boolean> {
  const row = await db.settings.get(TELEMETRY_ENABLED_SETTING_KEY);
  return normalizeTelemetrySetting(row?.value);
}
