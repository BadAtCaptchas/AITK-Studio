'use client';

import { useCallback, useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

const DEFAULT_EXTERNAL_COMFY_URL = 'http://127.0.0.1:8188';
export const SETTINGS_UPDATED_EVENT = 'aitk-settings-updated';

export interface Settings {
  HF_TOKEN: string;
  HF_TOKEN_SET: boolean;
  OPENROUTER_API_KEY: string;
  OPENROUTER_API_KEY_SET: boolean;
  TRAINING_FOLDER: string;
  DATASETS_FOLDER: string;
  PROJECTS_FOLDER: string;
  MODELS_PATH: string;
  MODELS_PATH_LOCKED: string;
  PROJECTS_ENABLED: string;
  OFFLINE_MODE: string;
  OFFLINE_MODE_LOCKED: string;
  TRAINING_ADVISOR_ENABLED: string;
  TRAINING_LEGACY_VIEW: string;
  TELEMETRY_ENABLED: string;
  COMFY_AUTO_INSTALL: string;
  COMFY_EXTERNAL_URL: string;
  COMFY_EXTERNAL_LORA_DIR: string;
}

const defaultSettings: Settings = {
  HF_TOKEN: '',
  HF_TOKEN_SET: false,
  OPENROUTER_API_KEY: '',
  OPENROUTER_API_KEY_SET: false,
  TRAINING_FOLDER: '',
  DATASETS_FOLDER: '',
  PROJECTS_FOLDER: '',
  MODELS_PATH: '',
  MODELS_PATH_LOCKED: 'false',
  PROJECTS_ENABLED: 'false',
  OFFLINE_MODE: 'false',
  OFFLINE_MODE_LOCKED: 'false',
  TRAINING_ADVISOR_ENABLED: 'false',
  TRAINING_LEGACY_VIEW: 'false',
  TELEMETRY_ENABLED: 'false',
  COMFY_AUTO_INSTALL: 'false',
  COMFY_EXTERNAL_URL: DEFAULT_EXTERNAL_COMFY_URL,
  COMFY_EXTERNAL_LORA_DIR: '',
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function stringSetting(data: Record<string, unknown>, key: keyof Settings, fallback = '') {
  return typeof data[key] === 'string' ? data[key] : fallback;
}

function booleanStringSetting(data: Record<string, unknown>, key: keyof Settings) {
  return data[key] === 'true' ? 'true' : 'false';
}

function normalizeSettings(value: unknown = {}): Settings {
  const data = isRecord(value) ? value : {};
  return {
    HF_TOKEN: stringSetting(data, 'HF_TOKEN'),
    HF_TOKEN_SET: data.HF_TOKEN_SET === true,
    OPENROUTER_API_KEY: stringSetting(data, 'OPENROUTER_API_KEY'),
    OPENROUTER_API_KEY_SET: data.OPENROUTER_API_KEY_SET === true,
    TRAINING_FOLDER: stringSetting(data, 'TRAINING_FOLDER'),
    DATASETS_FOLDER: stringSetting(data, 'DATASETS_FOLDER'),
    PROJECTS_FOLDER: stringSetting(data, 'PROJECTS_FOLDER'),
    MODELS_PATH: stringSetting(data, 'MODELS_PATH'),
    MODELS_PATH_LOCKED: booleanStringSetting(data, 'MODELS_PATH_LOCKED'),
    PROJECTS_ENABLED: booleanStringSetting(data, 'PROJECTS_ENABLED'),
    OFFLINE_MODE: booleanStringSetting(data, 'OFFLINE_MODE'),
    OFFLINE_MODE_LOCKED: booleanStringSetting(data, 'OFFLINE_MODE_LOCKED'),
    TRAINING_ADVISOR_ENABLED: booleanStringSetting(data, 'TRAINING_ADVISOR_ENABLED'),
    TRAINING_LEGACY_VIEW: booleanStringSetting(data, 'TRAINING_LEGACY_VIEW'),
    TELEMETRY_ENABLED: booleanStringSetting(data, 'TELEMETRY_ENABLED'),
    COMFY_AUTO_INSTALL: booleanStringSetting(data, 'COMFY_AUTO_INSTALL'),
    COMFY_EXTERNAL_URL: stringSetting(data, 'COMFY_EXTERNAL_URL', DEFAULT_EXTERNAL_COMFY_URL),
    COMFY_EXTERNAL_LORA_DIR: stringSetting(data, 'COMFY_EXTERNAL_LORA_DIR'),
  };
}

export function notifySettingsChanged(settings?: Partial<Settings>) {
  if (typeof window === 'undefined') return;
  window.dispatchEvent(new CustomEvent(SETTINGS_UPDATED_EVENT, { detail: settings }));
}

export default function useSettings() {
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  const [isSettingsLoaded, setIsLoaded] = useState(false);
  const refreshSettings = useCallback(async () => {
    const res = await apiClient.get('/api/settings');
    setSettings(normalizeSettings(res.data || {}));
    setIsLoaded(true);
  }, []);

  useEffect(() => {
    refreshSettings().catch(error => console.error('Error fetching settings:', error));

    const handleSettingsUpdated = (event: Event) => {
      const detail = (event as CustomEvent<Partial<Settings> | undefined>).detail;
      if (detail) {
        setSettings(normalizeSettings(detail));
        setIsLoaded(true);
        return;
      }
      refreshSettings().catch(error => console.error('Error fetching settings:', error));
    };

    window.addEventListener(SETTINGS_UPDATED_EVENT, handleSettingsUpdated);
    return () => window.removeEventListener(SETTINGS_UPDATED_EVENT, handleSettingsUpdated);
  }, [refreshSettings]);

  return { settings, setSettings, isSettingsLoaded };
}
