'use client';

import { useParams } from 'react-router-dom';
import DatasetEditorPage from '@/components/DatasetEditorPage';

export default function DatasetPage() {
  const { datasetName = '' } = useParams<'datasetName'>();
  return <DatasetEditorPage datasetName={datasetName} />;
}
