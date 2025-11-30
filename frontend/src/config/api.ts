/**
 * API configuration constants
 * Uses relative URL in production (HF Spaces serves frontend + backend together)
 */
export const API_BASE = process.env.REACT_APP_API_URL || 
  (window.location.hostname === 'localhost' ? 'http://localhost:8000' : '');

