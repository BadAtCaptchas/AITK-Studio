import { NextResponse } from 'next/server';
import { defaultTrainFolder, defaultDatasetsFolder } from '@/paths';
import { flushCache } from '@/server/settings';
import { db } from '@/server/db';
import path from 'path';

export async function GET() {
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
    return NextResponse.json(settingsObject);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch settings' }, { status: 500 });
  }
}

export async function POST(request: Request) {
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

    await db.settings.upsertMany({
      HF_TOKEN,
      TRAINING_FOLDER,
      DATASETS_FOLDER: normalizedDatasetsFolder,
    });

    flushCache();

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to update settings' }, { status: 500 });
  }
}
