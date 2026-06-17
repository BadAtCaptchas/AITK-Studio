'use client';

import { use } from 'react';
import DatasetEditorPage from '@/components/DatasetEditorPage';

export default function DatasetPage({ params }: { params: Promise<{ datasetName: string }> }) {
  const usableParams = use(params);
  return <DatasetEditorPage datasetName={usableParams.datasetName} />;
}
