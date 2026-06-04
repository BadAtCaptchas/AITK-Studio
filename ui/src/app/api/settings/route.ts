import { NextRequest, NextResponse } from 'next/server';
import { defaultTrainFolder, defaultDatasetsFolder } from '@/paths';
import { flushCache } from '@/server/settings';
import { db } from '@/server/db';
import { isEncryptedDatasetSecretSettingKey } from '@/server/encryptedDatasetSecrets';
import { isSecureCaptionSystemPromptSettingKey } from '@/server/secureCaptionSettings';
import path from 'path';

type SettingsAccess = {
  authenticated: boolean;
  response: NextResponse | null;
};

function normalizeBooleanSetting(value: unknown, defaultValue: boolean) {
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['false', '0', 'off', 'disabled'].includes(normalized)) return 'false';
    if (['true', '1', 'on', 'enabled'].includes(normalized)) return 'true';
  }
  return defaultValue ? 'true' : 'false';
}

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
    settingsObject.TRAINING_ADVISOR_ENABLED = normalizeBooleanSetting(
      settingsObject.TRAINING_ADVISOR_ENABLED,
      false,
    );
    settingsObject.COMFY_AUTO_INSTALL = normalizeBooleanSetting(settingsObject.COMFY_AUTO_INSTALL, false);
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
      TRAINING_ADVISOR_ENABLED,
      COMFY_AUTO_INSTALL,
    } = body;

    let normalizedDatasetsFolder = DATASETS_FOLDER;
    if (typeof DATASETS_FOLDER === 'string' && DATASETS_FOLDER !== '') {
      const resolvedDatasetsFolder = path.resolve(DATASETS_FOLDER);
      if (resolvedDatasetsFolder === path.parse(resolvedDatasetsFolder).root) {
        return NextResponse.json({ error: 'DATASETS_FOLDER cannot be filesystem root' }, { status: 400 });
      }
      normalizedDatasetsFolder = resolvedDatasetsFolder;
    }

    const settingsToUpdate: Record<string, string> = {
      TRAINING_FOLDER,
      DATASETS_FOLDER: normalizedDatasetsFolder,
      TRAINING_ADVISOR_ENABLED: normalizeBooleanSetting(TRAINING_ADVISOR_ENABLED, false),
      COMFY_AUTO_INSTALL: normalizeBooleanSetting(COMFY_AUTO_INSTALL, false),
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
