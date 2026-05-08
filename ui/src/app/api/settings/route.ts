import { NextRequest, NextResponse } from 'next/server';
import { defaultTrainFolder, defaultDatasetsFolder } from '@/paths';
import { flushCache } from '@/server/settings';
import { db } from '@/server/db';
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
    if (!access.authenticated) {
      settingsObject.HF_TOKEN_SET = Boolean(settingsObject.HF_TOKEN);
      settingsObject.HF_TOKEN = '';
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
    const { HF_TOKEN, TRAINING_FOLDER, DATASETS_FOLDER } = body;

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
    };

    if (typeof HF_TOKEN === 'string' && (access.authenticated || HF_TOKEN !== '')) {
      settingsToUpdate.HF_TOKEN = HF_TOKEN;
    }

    await db.settings.upsertMany(settingsToUpdate);

    flushCache();

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to update settings' }, { status: 500 });
  }
}
