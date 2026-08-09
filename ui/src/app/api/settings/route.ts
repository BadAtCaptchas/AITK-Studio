import { NextRequest, NextResponse } from 'next/server';
import { defaultTrainFolder, defaultDatasetsFolder, defaultModelsFolder, defaultProjectsFolder } from '@/paths';
import { flushCache, normalizeBooleanSetting, PROJECTS_ENABLED_KEY } from '@/server/settings';
import { normalizeStoragePathSetting } from '@/server/pathContainment';
import { db } from '@/server/db';
import { isEncryptedDatasetSecretSettingKey } from '@/server/encryptedDatasetSecrets';
import { isSecureCaptionSystemPromptSettingKey } from '@/server/secureCaptionSettings';
import { isRemoteOllamaWorkersSettingKey } from '@/server/remoteOllamaWorkers';
import { isDatasetWatchersSettingKey } from '@/server/datasetWatchers';
import { getOfflineModeState, OFFLINE_MODE_SETTING_KEY } from '@/server/networkPolicy';
import { DEFAULT_EXTERNAL_COMFY_URL, normalizeExternalComfyLoraDir, normalizeExternalComfyUrl } from '@/server/externalComfy';
import { IDEOGRAM_WORKFLOW_HISTORY_KEY } from '@/server/ideogramWorkflowHistory';
import { isRequestAuthenticated } from '@/utils/authSession';
import { TELEMETRY_ENABLED_SETTING_KEY } from '@/utils/telemetry';
import { modelsPathFromEnv, resolveModelsPathState } from '@/server/modelsPath';

type SettingsAccess = {
  authenticated: boolean;
  response: NextResponse | null;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

const storagePathSettings = [
  ['TRAINING_FOLDER', defaultTrainFolder],
  ['DATASETS_FOLDER', defaultDatasetsFolder],
  ['PROJECTS_FOLDER', defaultProjectsFolder],
] as const;

async function ensureSettingsAccess(request: NextRequest): Promise<SettingsAccess> {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;

  if (!tokenToUse) {
    return { authenticated: false, response: null };
  }

  if (!(await isRequestAuthenticated(request, tokenToUse))) {
    return {
      authenticated: false,
      response: NextResponse.json({ error: 'Unauthorized' }, { status: 401 }),
    };
  }

  return { authenticated: true, response: null };
}

export async function GET(request: NextRequest) {
  const access = await ensureSettingsAccess(request);
  if (access.response) {
    return access.response;
  }

  try {
    const settings = await db.settings.list();
    const settingsObject = settings.reduce<Record<string, unknown>>((acc, setting) => {
      if (isEncryptedDatasetSecretSettingKey(setting.key)) return acc;
      if (isSecureCaptionSystemPromptSettingKey(setting.key)) return acc;
      if (isRemoteOllamaWorkersSettingKey(setting.key)) return acc;
      if (isDatasetWatchersSettingKey(setting.key)) return acc;
      if (setting.key === IDEOGRAM_WORKFLOW_HISTORY_KEY) return acc;
      acc[setting.key] = setting.value;
      return acc;
    }, {});
    for (const [key, fallbackRoot] of storagePathSettings) {
      settingsObject[key] =
        (await normalizeStoragePathSetting(settingsObject[key], fallbackRoot, {
          allowExternal: access.authenticated,
        })) || fallbackRoot;
    }
    const modelsPathState = await resolveModelsPathState({
      defaultRoot: defaultModelsFolder,
      settingValue: settingsObject.MODELS_PATH,
      allowExternal: access.authenticated,
    });
    settingsObject.MODELS_PATH = modelsPathState.path;
    settingsObject.MODELS_PATH_LOCKED = modelsPathState.lockedByEnv ? 'true' : 'false';
    settingsObject.PROJECTS_ENABLED = normalizeBooleanSetting(settingsObject.PROJECTS_ENABLED, false);
    const offlineModeState = await getOfflineModeState();
    settingsObject[OFFLINE_MODE_SETTING_KEY] = offlineModeState.enabled ? 'true' : 'false';
    settingsObject.OFFLINE_MODE_LOCKED = offlineModeState.lockedByEnv ? 'true' : 'false';
    settingsObject.TRAINING_ADVISOR_ENABLED = normalizeBooleanSetting(
      settingsObject.TRAINING_ADVISOR_ENABLED,
      false,
    );
    settingsObject[TELEMETRY_ENABLED_SETTING_KEY] = normalizeBooleanSetting(
      settingsObject[TELEMETRY_ENABLED_SETTING_KEY],
      false,
    );
    settingsObject.COMFY_AUTO_INSTALL = normalizeBooleanSetting(settingsObject.COMFY_AUTO_INSTALL, false);
    settingsObject.COMFY_EXTERNAL_URL = normalizeExternalComfyUrl(
      settingsObject.COMFY_EXTERNAL_URL || DEFAULT_EXTERNAL_COMFY_URL,
    );
    settingsObject.COMFY_EXTERNAL_LORA_DIR = normalizeExternalComfyLoraDir(settingsObject.COMFY_EXTERNAL_LORA_DIR || '');
    settingsObject.HF_TOKEN_SET = Boolean(settingsObject.HF_TOKEN);
    settingsObject.HF_TOKEN = '';
    settingsObject.OPENROUTER_API_KEY_SET = Boolean(settingsObject.OPENROUTER_API_KEY);
    settingsObject.OPENROUTER_API_KEY = '';
    return NextResponse.json(settingsObject);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch settings' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  const access = await ensureSettingsAccess(request);
  if (access.response) {
    return access.response;
  }

  try {
    const body: unknown = await request.json();
    if (!isRecord(body)) {
      return NextResponse.json({ error: 'Settings payload must be an object' }, { status: 400 });
    }
    const {
      HF_TOKEN,
      OPENROUTER_API_KEY,
      TRAINING_FOLDER,
      DATASETS_FOLDER,
      PROJECTS_FOLDER,
      MODELS_PATH,
      PROJECTS_ENABLED,
      OFFLINE_MODE,
      TRAINING_ADVISOR_ENABLED,
      TELEMETRY_ENABLED,
      COMFY_AUTO_INSTALL,
      COMFY_EXTERNAL_URL,
      COMFY_EXTERNAL_LORA_DIR,
    } = body;

    const normalizedTrainingFolder = await normalizeStoragePathSetting(TRAINING_FOLDER, defaultTrainFolder, {
      allowExternal: access.authenticated,
    });
    if (!normalizedTrainingFolder) {
      return NextResponse.json(
        { error: 'TRAINING_FOLDER must stay inside the default training folder unless authentication is enabled' },
        { status: 400 },
      );
    }

    const normalizedDatasetsFolder = await normalizeStoragePathSetting(DATASETS_FOLDER, defaultDatasetsFolder, {
      allowExternal: access.authenticated,
    });
    if (!normalizedDatasetsFolder) {
      return NextResponse.json(
        { error: 'DATASETS_FOLDER must stay inside the default datasets folder unless authentication is enabled' },
        { status: 400 },
      );
    }

    const normalizedProjectsFolder = await normalizeStoragePathSetting(PROJECTS_FOLDER, defaultProjectsFolder, {
      allowExternal: access.authenticated,
    });
    if (!normalizedProjectsFolder) {
      return NextResponse.json(
        { error: 'PROJECTS_FOLDER must stay inside the default projects folder unless authentication is enabled' },
        { status: 400 },
      );
    }

    const configuredModelsPath = modelsPathFromEnv();
    let normalizedModelsPath: string | null = null;
    if (!configuredModelsPath) {
      const requestedModelsPath =
        MODELS_PATH === undefined ? (await db.settings.get('MODELS_PATH'))?.value : MODELS_PATH;
      normalizedModelsPath = await normalizeStoragePathSetting(requestedModelsPath, defaultModelsFolder, {
        allowExternal: access.authenticated,
      });
      if (!normalizedModelsPath) {
        return NextResponse.json(
          { error: 'MODELS_PATH must stay inside the default models folder unless authentication is enabled' },
          { status: 400 },
        );
      }
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
      TRAINING_FOLDER: normalizedTrainingFolder,
      DATASETS_FOLDER: normalizedDatasetsFolder,
      PROJECTS_FOLDER: normalizedProjectsFolder,
      ...(normalizedModelsPath ? { MODELS_PATH: normalizedModelsPath } : {}),
      PROJECTS_ENABLED: normalizeBooleanSetting(existingProjectsEnabled, false),
      [OFFLINE_MODE_SETTING_KEY]: offlineModeState.lockedByEnv
        ? 'true'
        : normalizeBooleanSetting(existingOfflineMode, false),
      TRAINING_ADVISOR_ENABLED: normalizeBooleanSetting(TRAINING_ADVISOR_ENABLED, false),
      [TELEMETRY_ENABLED_SETTING_KEY]: normalizeBooleanSetting(TELEMETRY_ENABLED, false),
      COMFY_AUTO_INSTALL: normalizeBooleanSetting(COMFY_AUTO_INSTALL, false),
      COMFY_EXTERNAL_URL: normalizedExternalComfyUrl,
      COMFY_EXTERNAL_LORA_DIR: normalizedExternalComfyLoraDir,
    };

    if (typeof HF_TOKEN === 'string' && HF_TOKEN.trim() !== '') {
      settingsToUpdate.HF_TOKEN = HF_TOKEN;
    }
    if (typeof OPENROUTER_API_KEY === 'string' && OPENROUTER_API_KEY.trim() !== '') {
      settingsToUpdate.OPENROUTER_API_KEY = OPENROUTER_API_KEY;
    }

    await db.settings.upsertMany(settingsToUpdate);

    flushCache();

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to update settings' }, { status: 500 });
  }
}
