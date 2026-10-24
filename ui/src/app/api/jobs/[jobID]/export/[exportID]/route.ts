import {
  getTrainingJobExportProgress,
  requestTrainingJobExportCancellation,
  updateTrainingJobExportProgress,
} from '@/server/trainingJobExportProgress';
import { getRemoteTrainingJobExport, clearRemoteTrainingJobExport } from '@/server/remoteExportProgress';
import { getRemoteWorker, remoteJson } from '@/server/remoteClient';
import { makeRemoteAssetRef } from '@/server/remoteAssets';


export async function GET(
  _request: Request,
  { params }: { params: Promise<{ jobID: string; exportID: string }> },
) {
  const { jobID, exportID } = await params;
  const progress = getTrainingJobExportProgress(exportID);

  if (!progress || progress.jobID !== jobID) {
    return Response.json({ error: 'Export not found' }, { status: 404 });
  }

  const remote = getRemoteTrainingJobExport(exportID);
  if (remote) {
    try {
      const worker = await getRemoteWorker(remote.workerID);
      const remoteProgress = await remoteJson<any>(
        worker,
        `/api/jobs/${encodeURIComponent(remote.remoteJobID)}/export/${encodeURIComponent(remote.remoteExportID)}`,
      );
      const mapped = updateTrainingJobExportProgress(exportID, {
        status: remoteProgress.status,
        message: remoteProgress.message,
        percent: remoteProgress.percent,
        entriesProcessed: remoteProgress.entriesProcessed,
        entriesTotal: remoteProgress.entriesTotal,
        bytesProcessed: remoteProgress.bytesProcessed,
        bytesTotal: remoteProgress.bytesTotal,
        warnings: remoteProgress.warnings || [],
        error: remoteProgress.error || null,
        zipPath:
          remoteProgress.zipPath && remoteProgress.fileName
            ? makeRemoteAssetRef(jobID, 'file', remoteProgress.zipPath)
            : null,
        fileName: remoteProgress.fileName || null,
      });
      if (['completed', 'failed', 'canceled'].includes(remoteProgress.status)) {
        clearRemoteTrainingJobExport(exportID);
      }
      return Response.json(mapped || progress);
    } catch (error) {
      const mapped = updateTrainingJobExportProgress(exportID, {
        status: 'failed',
        message: 'Remote export failed',
        error: error instanceof Error ? error.message : 'Remote export failed',
      });
      clearRemoteTrainingJobExport(exportID);
      return Response.json(mapped || progress);
    }
  }

  return Response.json(progress);
}

export async function DELETE(
  _request: Request,
  { params }: { params: Promise<{ jobID: string; exportID: string }> },
) {
  const { jobID, exportID } = await params;
  const progress = getTrainingJobExportProgress(exportID);

  if (!progress || progress.jobID !== jobID) {
    return Response.json({ error: 'Export not found' }, { status: 404 });
  }

  const remote = getRemoteTrainingJobExport(exportID);
  if (remote) {
    try {
      const worker = await getRemoteWorker(remote.workerID);
      await remoteJson(
        worker,
        `/api/jobs/${encodeURIComponent(remote.remoteJobID)}/export/${encodeURIComponent(remote.remoteExportID)}`,
        { method: 'DELETE' },
      );
    } catch (error) {
      console.error('Unable to cancel remote export:', error);
    }
  }

  const updated = requestTrainingJobExportCancellation(exportID);
  return Response.json(updated || progress);
}
