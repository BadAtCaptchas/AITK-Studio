import { NextRequest, NextResponse } from 'next/server';
import { defaultTrainFolder, defaultDatasetsFolder, defaultProjectsFolder } from '@/paths';
import { flushCache, normalizeBooleanSetting, PROJECTS_ENABLED_KEY } from '@/server/settings';
import { db } from '@/server/db';
import { isEncryptedDatasetSecretSettingKey } from '@/server/encryptedDatasetSecrets';
import { isSecureCaptionSystemPromptSettingKey } from '@/server/secureCaptionSettings';
import { isRemoteOllamaWorkersSettingKey } from '@/server/remoteOllamaWorkers';
import { getOfflineModeState, OFFLINE_MODE_SETTING_KEY } from '@/server/networkPolicy';
import { DEFAULT_EXTERNAL_COMFY_URL, normalizeExternalComfyLoraDir, normalizeExternalComfyUrl } from '@/server/externalComfy';
import { IDEOGRAM_WORKFLOW_HISTORY_KEY } from '@/server/ideogramWorkflowHistory';
import path from 'path';

type SettingsAccess = {
  authenticated: boolean;
  response: NextResponse | null;
};

function ensureSettingsAccess(request: NextRequest): SettingsAccess {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  const token = request.headers.get('authorization')?.split(' ')[1];

  if (!tokenToUse) {
    return { authenticated: false, response: null };
  }

  if (token !== tokenToUse) {
    return {
      authenticated: false,
      response: NextResponse.json({ error: 'Unauthorized' }, { status: 401 }),
    };
  }

  return { authenticated: true, response: null };
}

export async function GET(request: NextRequest) {
  const access = ensureSettingsAccess(request);
  if (access.response) {
    return access.response;
  }

  try {
    const settings = await db.settings.list();
    const settingsObject = settings.reduce((acc: any, setting) => {
      if (isEncryptedDatasetSecretSettingKey(setting.key)) return acc;
      if (isSecureCaptionSystemPromptSettingKey(setting.key)) return acc;
      if (isRemoteOllamaWorkersSettingKey(setting.key)) return acc;
      if (setting.key === IDEOGRAM_WORKFLOW_HISTORY_KEY) return acc;
      acc[setting.key] = setting.value;
      return acc;
    }, {});
    // if TRAINING_FOLDER is not set, use default
    if (!settingsObject.TRAINING_FOLDER || settingsObject.TRAINING_FOLDER === '') {
      settingsObject.TRAINING_FOLDER = defaultTrainFolder;
    }
    // if DATASETS_FOLDER is not set, use default
    if (!settingsObject.DATASETS_FOLDER || settingsObject.DATASETS_FOLDER === '') {
      settingsObject.DATASETS_FOLDER = defaultDatasetsFolder;
    }
    if (!settingsObject.PROJECTS_FOLDER || settingsObject.PROJECTS_FOLDER === '') {
      settingsObject.PROJECTS_FOLDER = defaultProjectsFolder;
    }
    settingsObject.PROJECTS_ENABLED = normalizeBooleanSetting(settingsObject.PROJECTS_ENABLED, true);
    const offlineModeState = await getOfflineModeState();
    settingsObject[OFFLINE_MODE_SETTING_KEY] = offlineModeState.enabled ? 'true' : 'false';
    settingsObject.OFFLINE_MODE_LOCKED = offlineModeState.lockedByEnv ? 'true' : 'false';
    settingsObject.TRAINING_ADVISOR_ENABLED = normalizeBooleanSetting(
      settingsObject.TRAINING_ADVISOR_ENABLED,
      false,
    );
    settingsObject.COMFY_AUTO_INSTALL = normalizeBooleanSetting(settingsObject.COMFY_AUTO_INSTALL, false);
    settingsObject.COMFY_EXTERNAL_URL = normalizeExternalComfyUrl(
      settingsObject.COMFY_EXTERNAL_URL || DEFAULT_EXTERNAL_COMFY_URL,
    );
    settingsObject.COMFY_EXTERNAL_LORA_DIR = normalizeExternalComfyLoraDir(settingsObject.COMFY_EXTERNAL_LORA_DIR || '');
    if (!access.authenticated) {
      settingsObject.HF_TOKEN_SET = Boolean(settingsObject.HF_TOKEN);
      settingsObject.HF_TOKEN = '';
      settingsObject.OPENROUTER_API_KEY_SET = Boolean(settingsObject.OPENROUTER_API_KEY);
      settingsObject.OPENROUTER_API_KEY = '';
    }
    return NextResponse.json(settingsObject);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch settings' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  const access = ensureSettingsAccess(request);
  if (access.response) {
    return access.response;
  }

  try {
    const body = await request.json();
    const {
      HF_TOKEN,
      OPENROUTER_API_KEY,
      TRAINING_FOLDER,
      DATASETS_FOLDER,
      PROJECTS_FOLDER,
      PROJECTS_ENABLED,
      OFFLINE_MODE,
      TRAINING_ADVISOR_ENABLED,
      COMFY_AUTO_INSTALL,
      COMFY_EXTERNAL_URL,
      COMFY_EXTERNAL_LORA_DIR,
    } = body;

    let normalizedDatasetsFolder = DATASETS_FOLDER;
    if (typeof DATASETS_FOLDER === 'string' && DATASETS_FOLDER !== '') {
      const resolvedDatasetsFolder = path.resolve(DATASETS_FOLDER);
      if (resolvedDatasetsFolder === path.parse(resolvedDatasetsFolder).root) {
        return NextResponse.json({ error: 'DATASETS_FOLDER cannot be filesystem root' }, { status: 400 });
      }
      normalizedDatasetsFolder = resolvedDatasetsFolder;
    }

    let normalizedProjectsFolder = PROJECTS_FOLDER;
    if (typeof PROJECTS_FOLDER === 'string' && PROJECTS_FOLDER !== '') {
      const resolvedProjectsFolder = path.resolve(PROJECTS_FOLDER);
      if (resolvedProjectsFolder === path.parse(resolvedProjectsFolder).root) {
        return NextResponse.json({ error: 'PROJECTS_FOLDER cannot be filesystem root' }, { status: 400 });
      }
      normalizedProjectsFolder = resolvedProjectsFolder;
    }

    let normalizedExternalComfyUrl = '';
    let normalizedExternalComfyLoraDir = '';
    try {
      normalizedExternalComfyUrl = normalizeExternalComfyUrl(COMFY_EXTERNAL_URL || DEFAULT_EXTERNAL_COMFY_URL);
      normalizedExternalComfyLoraDir = normalizeExternalComfyLoraDir(COMFY_EXTERNAL_LORA_DIR || '');
    } catch (error) {
      return NextResponse.json(
        { error: error instanceof Error ? error.message : 'Invalid external ComfyUI setting' },
        { status: 400 },
      );
    }

    const existingProjectsEnabled =
      PROJECTS_ENABLED === undefined ? (await db.settings.get(PROJECTS_ENABLED_KEY))?.value : PROJECTS_ENABLED;
    const offlineModeState = await getOfflineModeState();
    const existingOfflineMode =
      OFFLINE_MODE === undefined ? (await db.settings.get(OFFLINE_MODE_SETTING_KEY))?.value : OFFLINE_MODE;

    const settingsToUpdate: Record<string, string> = {
      TRAINING_FOLDER,
      DATASETS_FOLDER: normalizedDatasetsFolder,
      PROJECTS_FOLDER: normalizedProjectsFolder,
      PROJECTS_ENABLED: normalizeBooleanSetting(existingProjectsEnabled, true),
      [OFFLINE_MODE_SETTING_KEY]: offlineModeState.lockedByEnv
        ? 'true'
        : normalizeBooleanSetting(existingOfflineMode, false),
      TRAINING_ADVISOR_ENABLED: normalizeBooleanSetting(TRAINING_ADVISOR_ENABLED, false),
      COMFY_AUTO_INSTALL: normalizeBooleanSetting(COMFY_AUTO_INSTALL, false),
      COMFY_EXTERNAL_URL: normalizedExternalComfyUrl,
      COMFY_EXTERNAL_LORA_DIR: normalizedExternalComfyLoraDir,
    };

    if (typeof HF_TOKEN === 'string' && (access.authenticated || HF_TOKEN !== '')) {
      settingsToUpdate.HF_TOKEN = HF_TOKEN;
    }
    if (typeof OPENROUTER_API_KEY === 'string' && (access.authenticated || OPENROUTER_API_KEY !== '')) {
      settingsToUpdate.OPENROUTER_API_KEY = OPENROUTER_API_KEY;
    }

    await db.settings.upsertMany(settingsToUpdate);

    flushCache();

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to update settings' }, { status: 500 });
  }
}
