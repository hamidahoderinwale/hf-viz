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

window.addEventListener('error', (event) => {
  if (isWebSocketError(event.error, event.message)) {
    // Suppress WebSocket errors - they're from HMR and non-critical
    event.preventDefault();
    event.stopPropagation();
    return false;
  }
}, true); // Use capture phase to catch early

// Handle unhandled promise rejections related to WebSockets
window.addEventListener('unhandledrejection', (event) => {
  if (isWebSocketError(event.reason)) {
    // Suppress WebSocket promise rejections
    event.preventDefault();
    event.stopPropagation();
  }
}, true); // Use capture phase

// Also suppress console errors for WebSocket issues
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
  originalConsoleError.apply(console, args);
};

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <App />
);

