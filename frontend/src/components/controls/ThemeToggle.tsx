/**
 * Toggle button for switching between light and dark themes.
 */
import React from 'react';
import { useFilterStore } from '../../stores/filterStore';

export default function ThemeToggle() {
  const theme = useFilterStore((state) => state.theme);
  const toggleTheme = useFilterStore((state) => state.toggleTheme);

  return (
    <button
      onClick={toggleTheme}
      className="theme-toggle"
      title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
      aria-label={`Current theme: ${theme}. Click to switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
    >
      {theme === 'light' ? 'Dark' : 'Light'}
    </button>
  );
}

