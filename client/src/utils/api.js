import axios from 'axios';
import { apiUrl, apiTimeout } from '../config';

const api = axios.create({
  baseURL: apiUrl,
  timeout: apiTimeout,
  headers: {
    'Content-Type': 'application/json',
  },
});

const handleResponse = (response) => {
  const { data, status } = response;
  if (status >= 200 && status < 300) {
    return data;
  } else {
    throw new Error(`API error: ${status}`);
  }
};

const handleError = (error) => {
  if (error.response) {
    const { status, data } = error.response;
    throw new Error(`API error: ${status} - ${data.message}`);
  } else {
    throw error;
  }
};

const get = (endpoint, params = {}) => {
  return api.get(endpoint, { params }).then(handleResponse).catch(handleError);
};

const post = (endpoint, data = {}) => {
  return api.post(endpoint, data).then(handleResponse).catch(handleError);
};

const put = (endpoint, data = {}) => {
  return api.put(endpoint, data).then(handleResponse).catch(handleError);
};

const del = (endpoint) => {
  return api.delete(endpoint).then(handleResponse).catch(handleError);
};

export { get, post, put, del };
