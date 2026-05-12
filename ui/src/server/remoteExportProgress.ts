type RemoteExportMapping = {
  jobID: string;
  workerID: string;
  remoteJobID: string;
  remoteExportID: string;
};

declare global {
  // eslint-disable-next-line no-var
  var __remoteTrainingJobExportMap: Map<string, RemoteExportMapping> | undefined;
}

const remoteExportMap = globalThis.__remoteTrainingJobExportMap ?? new Map<string, RemoteExportMapping>();
if (!globalThis.__remoteTrainingJobExportMap) {
  globalThis.__remoteTrainingJobExportMap = remoteExportMap;
}

export function registerRemoteTrainingJobExport(localExportID: string, mapping: RemoteExportMapping) {
  remoteExportMap.set(localExportID, mapping);
}

export function getRemoteTrainingJobExport(localExportID: string) {
  return remoteExportMap.get(localExportID) || null;
}

export function clearRemoteTrainingJobExport(localExportID: string) {
  remoteExportMap.delete(localExportID);
}
