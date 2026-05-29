export function getSecureRemoteOllamaWorkerId(jobConfig: any) {
  const processes = Array.isArray(jobConfig?.config?.process) ? jobConfig.config.process : [];
  for (const processConfig of processes) {
    if (processConfig?.type !== 'SecureRemoteOllamaCaptioner') continue;
    const workerId = processConfig?.caption?.remote_worker_id;
    if (typeof workerId === 'string' && workerId.trim() && workerId !== 'local') {
      return workerId.trim();
    }
  }
  return null;
}

export function isSecureRemoteOllamaCaptionJob(jobConfig: any) {
  return getSecureRemoteOllamaWorkerId(jobConfig) !== null;
}
