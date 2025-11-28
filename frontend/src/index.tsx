import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// Global error handler for WebSocket errors (HMR, DevTools, etc.)
// These are non-critical and shouldn't break the app
const isWebSocketError = (error: any, message?: string): boolean => {
  const errorStr = error?.toString() || '';
  const messageStr = message?.toString() || '';
  const combined = `${errorStr} ${messageStr}`.toLowerCase();
  
  return (
    combined.includes('websocket') ||
    combined.includes('websocketclient') ||
    combined.includes('initsocket') ||
    combined.includes('hmr') ||
    combined.includes('hot module') ||
    error?.message?.toLowerCase().includes('websocket') ||
    error?.stack?.toLowerCase().includes('websocket')
  );
};

// Check if error is WebGL context loss (non-critical, handled by component)
const isWebGLError = (error: any, message?: string): boolean => {
  const errorStr = error?.toString() || '';
  const messageStr = message?.toString() || '';
  const combined = `${errorStr} ${messageStr}`.toLowerCase();
  
  return (
    combined.includes('webgl') ||
    combined.includes('context lost') ||
    combined.includes('webglrenderer') ||
    error?.message?.toLowerCase().includes('webgl') ||
    error?.stack?.toLowerCase().includes('webgl')
  );
};

window.addEventListener('error', (event) => {
  if (isWebSocketError(event.error, event.message)) {
    // Suppress WebSocket errors - they're from HMR and non-critical
    event.preventDefault();
    event.stopPropagation();
    return false;
  }
  if (isWebGLError(event.error, event.message)) {
    // Suppress WebGL context loss errors - handled by component recovery
    if (process.env.NODE_ENV === 'development') {
      console.warn('WebGL context error (handled):', event.message);
    }
    event.preventDefault();
    event.stopPropagation();
    return false;
  }
}, true); // Use capture phase to catch early

// Handle unhandled promise rejections related to WebSockets and WebGL
window.addEventListener('unhandledrejection', (event) => {
  if (isWebSocketError(event.reason)) {
    // Suppress WebSocket promise rejections
    event.preventDefault();
    event.stopPropagation();
  }
  if (isWebGLError(event.reason)) {
    // Suppress WebGL promise rejections (handled by component)
    if (process.env.NODE_ENV === 'development') {
      console.warn('WebGL promise rejection (handled):', event.reason);
    }
    event.preventDefault();
    event.stopPropagation();
  }
}, true); // Use capture phase

// Also suppress console errors for WebSocket and WebGL issues
const originalConsoleError = console.error;
console.error = (...args: any[]) => {
  const message = args.join(' ').toLowerCase();
  if (
    message.includes('websocket') ||
    message.includes('websocketclient') ||
    message.includes('initsocket')
  ) {
    // Suppress WebSocket console errors
    return;
  }
  if (
    message.includes('webgl') ||
    message.includes('context lost') ||
    message.includes('webglrenderer')
  ) {
    // Suppress WebGL context loss console errors (handled by component)
    return;
  }
  originalConsoleError.apply(console, args);
};

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <App />
);

