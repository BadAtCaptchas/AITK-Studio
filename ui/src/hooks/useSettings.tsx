'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

const DEFAULT_EXTERNAL_COMFY_URL = 'http://127.0.0.1:8188';

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

export default function useSettings() {
  const [settings, setSettings] = useState({
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
  });
  const [isSettingsLoaded, setIsLoaded] = useState(false);
  useEffect(() => {
    apiClient
      .get('/api/settings')
      .then(res => res.data)
      .then(data => {
        console.log('Settings:', data);
        setSettings({
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
        });
        setIsLoaded(true);
      })
      .catch(error => console.error('Error fetching settings:', error));
  }, []);

  return { settings, setSettings, isSettingsLoaded };
}
