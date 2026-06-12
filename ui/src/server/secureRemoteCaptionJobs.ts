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

export function getDirectRemoteOllamaWorkerId(jobConfig: any) {
  const processes = Array.isArray(jobConfig?.config?.process) ? jobConfig.config.process : [];
  for (const processConfig of processes) {
    if (processConfig?.type !== 'SecureRemoteOllamaCaptioner' && processConfig?.type !== 'OllamaCaptioner') continue;
    const workerId = processConfig?.caption?.remote_ollama_worker_id;
    if (typeof workerId === 'string' && workerId.trim() && workerId !== 'local') {
      return workerId.trim();
    }
  }
  return null;
}

export function isDirectRemoteOllamaCaptionJob(jobConfig: any) {
  return getDirectRemoteOllamaWorkerId(jobConfig) !== null;
}

export function isAnyRemoteOllamaCaptionJob(jobConfig: any) {
  return isSecureRemoteOllamaCaptionJob(jobConfig) || isDirectRemoteOllamaCaptionJob(jobConfig);
}

export function rewriteDirectRemoteOllamaCaptionersForLocalOllama(jobConfig: any) {
  const processes = Array.isArray(jobConfig?.config?.process) ? jobConfig.config.process : [];
  for (const processConfig of processes) {
    if (processConfig?.type !== 'SecureRemoteOllamaCaptioner' && processConfig?.type !== 'OllamaCaptioner') continue;
    const workerId = processConfig?.caption?.remote_ollama_worker_id;
    if (typeof workerId !== 'string' || !workerId.trim() || workerId === 'local') continue;
    processConfig.type = 'OllamaCaptioner';
  }
  return jobConfig;
}
