'use client';

import { useCallback, useEffect, useState, useRef } from 'react';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';

interface FileObject {
  path: string;
  size: number;
}

export default function useFilesList(jobID: string, reloadInterval: null | number = null) {
  const [files, setFiles] = useState<FileObject[]>([]);
  const didInitialLoadRef = useRef(false);
  const activeJobIDRef = useRef(jobID);
  activeJobIDRef.current = jobID;
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const refreshFiles = useCallback((signal?: AbortSignal) => {
    const requestJobID = jobID;
    if (activeJobIDRef.current !== requestJobID) return;
    let loadStatus: 'loading' | 'refreshing' = 'loading';
    if (didInitialLoadRef.current) {
      loadStatus = 'refreshing';
    }
    setStatus(loadStatus);
    return apiClient
      .get(`/api/jobs/${jobID}/files`, { signal })
      .then(res => res.data)
      .then(data => {
        if (signal?.aborted || activeJobIDRef.current !== requestJobID) return;
        console.log('Fetched files:', data);
        if (data.files) {
          setFiles(data.files);
        }
        setStatus('success');
        didInitialLoadRef.current = true;
      })
      .catch(error => {
        if (signal?.aborted || activeJobIDRef.current !== requestJobID) return;
        console.error('Error fetching datasets:', error);
        setStatus('error');
      });
  }, [jobID]);

  useEffect(() => {
    didInitialLoadRef.current = false;
    setFiles([]);
    setStatus('idle');
  }, [jobID]);

  usePollLoop(signal => refreshFiles(signal), reloadInterval, [jobID]);

  return { files, setFiles, status, refreshFiles: () => refreshFiles() };
}
