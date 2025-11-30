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
  const stack = error?.stack?.toLowerCase() || '';
  
  return (
    combined.includes('webgl') ||
    combined.includes('context lost') ||
    combined.includes('webglrenderer') ||
    combined.includes('three.module.js') ||
    combined.includes('three.js') ||
    error?.message?.toLowerCase().includes('webgl') ||
    stack.includes('webgl') ||
    stack.includes('three.module.js') ||
    stack.includes('three.js')
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
  // Suppress 404 errors for expected API endpoints
  if (
    event.message &&
    (event.message.includes('404') || event.message.includes('Failed to load resource')) &&
    (event.message.includes('/api/family/') ||
     (event.message.includes('/api/model/') && event.message.includes('/papers')) ||
     event.message.includes('/api/family/path/'))
  ) {
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
  // Suppress 404 promise rejections for expected API endpoints
  const reasonStr = String(event.reason || event.promise || '');
  if (
    reasonStr.includes('404') &&
    (reasonStr.includes('/api/family/') ||
     (reasonStr.includes('/api/model/') && reasonStr.includes('/papers')) ||
     reasonStr.includes('/api/family/path/'))
  ) {
    event.preventDefault();
    event.stopPropagation();
  }
}, true); // Use capture phase

// Suppress console errors and warnings for WebSocket and WebGL issues
// Must override BEFORE any imports that might log
const originalConsoleError = console.error;
const originalConsoleWarn = console.warn;
const originalConsoleLog = console.log;

// Comprehensive error suppression
const shouldSuppress = (args: any[]): boolean => {
  const message = args.join(' ').toLowerCase();
  const source = args.find(arg => typeof arg === 'string' && arg.includes('.js'));
  
  // Check for deprecated MouseEvent warnings (from Three.js OrbitControls)
  if (
    message.includes('mouseevent.mozpressure') ||
    message.includes('mouseevent.mozinputsource') ||
    message.includes('is deprecated')
  ) {
    return true;
  }
  
  // Check for WebSocket errors
  if (
    message.includes('websocket') ||
    message.includes('websocketclient') ||
    message.includes('initsocket')
  ) {
    return true;
  }
  
  // Check for WebGL/Three.js errors (including three.module.js and bundle.js)
  if (
    message.includes('webgl') ||
    message.includes('context lost') ||
    message.includes('webglrenderer') ||
    message.includes('three.webglrenderer') ||
    message.includes('three.webglrenderer: context lost') ||
    (source && (source.includes('three.module.js') || 
                source.includes('three.js') || 
                source.includes('bundle.js')))
  ) {
    return true;
  }
  
  // Check for NetworkError (expected during startup)
  if (message.includes('networkerror') || message.includes('network error')) {
    return true;
  }
  
  // Suppress 404 errors for expected API endpoints (family, papers, path)
  if (
    message.includes('404') &&
    (message.includes('/api/family/') ||
     (message.includes('/api/model/') && message.includes('/papers')) ||
     message.includes('/api/family/path/'))
  ) {
    return true;
  }
  
  return false;
};

console.error = (...args: any[]) => {
  if (shouldSuppress(args)) {
    return; // Suppress
  }
  originalConsoleError.apply(console, args);
};

console.warn = (...args: any[]) => {
  if (shouldSuppress(args)) {
    return; // Suppress
  }
  originalConsoleWarn.apply(console, args);
};

// Also suppress console.log for Three.js WebGL messages
console.log = (...args: any[]) => {
  if (shouldSuppress(args)) {
    return; // Suppress
  }
  originalConsoleLog.apply(console, args);
};

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <App />
);

