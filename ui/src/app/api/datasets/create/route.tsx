import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';
import {
  cleanDatasetName,
  resolveDatasetFolder,
  validateEncryptedManifest,
  writeEncryptedManifest,
} from '@/server/encryptedDatasets';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    let { name, encrypted, encryptedManifest } = body;
    name = cleanDatasetName(name || '');
    if (!name) {
      return NextResponse.json({ error: 'Dataset name is required' }, { status: 400 });
    }

    let datasetsPath = await getDatasetsRoot();
    let datasetPath = resolveDatasetFolder(datasetsPath, name);

    // if folder doesnt exist, create it
    if (!fs.existsSync(datasetPath)) {
      fs.mkdirSync(datasetPath);
    } else if (fs.readdirSync(datasetPath).length > 0) {
      return NextResponse.json({ error: 'Dataset already exists' }, { status: 409 });
    }

    if (encrypted) {
      const manifest = validateEncryptedManifest(encryptedManifest);
      fs.mkdirSync(path.join(datasetPath, 'objects'), { recursive: true });
      await writeEncryptedManifest(datasetPath, manifest);
    }

    return NextResponse.json({ success: true, name: name, encrypted: !!encrypted });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
