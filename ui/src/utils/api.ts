import axios from 'axios';
import { createGlobalState } from 'react-global-hooks';

export const isAuthorizedState = createGlobalState(false);

export const apiClient = axios.create();

const AUTH_STORAGE_KEY = 'AI_TOOLKIT_AUTH';

const getAuthToken = () => {
  if (typeof window === 'undefined') {
    return null;
  }
  return window.localStorage.getItem(AUTH_STORAGE_KEY);
};

const clearAuthToken = () => {
  if (typeof window === 'undefined') {
    return;
  }
  window.localStorage.removeItem(AUTH_STORAGE_KEY);
};

// Add a request interceptor to add token from localStorage
apiClient.interceptors.request.use(config => {
  const token = getAuthToken();
  if (token) {
    config.headers['Authorization'] = `Bearer ${token}`;
  }
  return config;
});

// Add a response interceptor to handle 401 errors
apiClient.interceptors.response.use(
  response => response, // Return successful responses as-is
  error => {
    // Check if the error is a 401 Unauthorized
    if (error.response && error.response.status === 401) {
      // Clear the auth token from localStorage
      clearAuthToken();
      isAuthorizedState.set(false);
    }

    // Reject the promise with the error so calling code can still catch it
    return Promise.reject(error);
  },
);
