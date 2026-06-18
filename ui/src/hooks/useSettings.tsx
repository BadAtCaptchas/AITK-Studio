'use client';

import { useCallback, useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

const DEFAULT_EXTERNAL_COMFY_URL = 'http://127.0.0.1:8188';
export const SETTINGS_UPDATED_EVENT = 'aitk-settings-updated';

export interface Settings {
  HF_TOKEN: string;
  OPENROUTER_API_KEY: string;
  TRAINING_FOLDER: string;
  DATASETS_FOLDER: string;
  PROJECTS_FOLDER: string;
  PROJECTS_ENABLED: string;
  TRAINING_ADVISOR_ENABLED: string;
  COMFY_AUTO_INSTALL: string;
  COMFY_EXTERNAL_URL: string;
  COMFY_EXTERNAL_LORA_DIR: string;
}

const defaultSettings: Settings = {
  HF_TOKEN: '',
  OPENROUTER_API_KEY: '',
  TRAINING_FOLDER: '',
  DATASETS_FOLDER: '',
  PROJECTS_FOLDER: '',
  PROJECTS_ENABLED: 'true',
  TRAINING_ADVISOR_ENABLED: 'false',
  COMFY_AUTO_INSTALL: 'false',
  COMFY_EXTERNAL_URL: DEFAULT_EXTERNAL_COMFY_URL,
  COMFY_EXTERNAL_LORA_DIR: '',
};

function normalizeSettings(data: Partial<Settings> = {}): Settings {
  return {
    HF_TOKEN: data.HF_TOKEN || '',
    OPENROUTER_API_KEY: data.OPENROUTER_API_KEY || '',
    TRAINING_FOLDER: data.TRAINING_FOLDER || '',
    DATASETS_FOLDER: data.DATASETS_FOLDER || '',
    PROJECTS_FOLDER: data.PROJECTS_FOLDER || '',
    PROJECTS_ENABLED: data.PROJECTS_ENABLED === 'false' ? 'false' : 'true',
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
    console.log('Settings:', res.data);
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
