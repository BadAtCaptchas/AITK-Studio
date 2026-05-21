import { NextResponse } from 'next/server';
import { getDatasetsRoot } from '@/server/settings';
import { listDatasetSummaries } from '@/server/encryptedDatasets';

export async function GET() {
  try {
    let datasetsPath = await getDatasetsRoot();
    return NextResponse.json(await listDatasetSummaries(datasetsPath));
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch datasets' }, { status: 500 });
  }
}
