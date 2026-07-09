import axios from 'axios';
import { createGlobalState } from 'react-global-hooks';

export const isAuthorizedState = createGlobalState(false);
export const authRequiredState = createGlobalState(false);

export const apiClient = axios.create({ withCredentials: true });

// Add a response interceptor to handle 401 errors
apiClient.interceptors.response.use(
  response => response, // Return successful responses as-is
  error => {
    // Check if the error is a 401 Unauthorized
    if (error.response && error.response.status === 401) {
      isAuthorizedState.set(false);
    }

    // Reject the promise with the error so calling code can still catch it
    return Promise.reject(error);
  },
);

export async function logout() {
  try {
    await apiClient.delete('/api/auth');
  } finally {
    isAuthorizedState.set(false);
  }
}
