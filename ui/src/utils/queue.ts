import { apiClient } from '@/utils/api';

export const startQueue = (queueID: string, workerID = 'local') => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/queue/${queueID}/start`, { params: { worker_id: workerID } })
      .then(res => res.data)
      .then(data => {
        console.log('Queue started:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error starting queue:', error);
        reject(error);
      });
  });
};
export const stopQueue = (queueID: string, workerID = 'local') => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/queue/${queueID}/stop`, { params: { worker_id: workerID } })
      .then(res => res.data)
      .then(data => {
        console.log('Queue stopped:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error stopping queue:', error);
        reject(error);
      });
  });
};
