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
  PROJECTS_ENABLED: string;
  OFFLINE_MODE: string;
  OFFLINE_MODE_LOCKED: string;
  TRAINING_ADVISOR_ENABLED: string;
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
  PROJECTS_ENABLED: 'false',
  OFFLINE_MODE: 'false',
  OFFLINE_MODE_LOCKED: 'false',
  TRAINING_ADVISOR_ENABLED: 'false',
  COMFY_AUTO_INSTALL: 'false',
  COMFY_EXTERNAL_URL: DEFAULT_EXTERNAL_COMFY_URL,
  COMFY_EXTERNAL_LORA_DIR: '',
};

function normalizeSettings(data: Partial<Settings> = {}): Settings {
  return {
    HF_TOKEN: data.HF_TOKEN || '',
    HF_TOKEN_SET: data.HF_TOKEN_SET === true,
    OPENROUTER_API_KEY: data.OPENROUTER_API_KEY || '',
    OPENROUTER_API_KEY_SET: data.OPENROUTER_API_KEY_SET === true,
    TRAINING_FOLDER: data.TRAINING_FOLDER || '',
    DATASETS_FOLDER: data.DATASETS_FOLDER || '',
    PROJECTS_FOLDER: data.PROJECTS_FOLDER || '',
    PROJECTS_ENABLED: data.PROJECTS_ENABLED === 'true' ? 'true' : 'false',
    OFFLINE_MODE: data.OFFLINE_MODE === 'true' ? 'true' : 'false',
    OFFLINE_MODE_LOCKED: data.OFFLINE_MODE_LOCKED === 'true' ? 'true' : 'false',
    TRAINING_ADVISOR_ENABLED: data.TRAINING_ADVISOR_ENABLED === 'true' ? 'true' : 'false',
    COMFY_AUTO_INSTALL: data.COMFY_AUTO_INSTALL === 'true' ? 'true' : 'false',
    COMFY_EXTERNAL_URL: data.COMFY_EXTERNAL_URL || DEFAULT_EXTERNAL_COMFY_URL,
    COMFY_EXTERNAL_LORA_DIR: data.COMFY_EXTERNAL_LORA_DIR || '',
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
