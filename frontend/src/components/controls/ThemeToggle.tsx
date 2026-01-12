/**
 * Toggle button for switching between light and dark themes.
 */
import React, { useEffect } from 'react';
import { Moon, Sun } from 'lucide-react';
import { useFilterStore } from '../../stores/filterStore';

export default function ThemeToggle() {
  const theme = useFilterStore((state) => state.theme);
  const toggleTheme = useFilterStore((state) => state.toggleTheme);

  // Listen for system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (e: MediaQueryListEvent) => {
      // Only auto-switch if user hasn't manually set a preference
      const saved = localStorage.getItem('theme');
      if (!saved) {
        const newTheme = e.matches ? 'dark' : 'light';
        useFilterStore.getState().setTheme(newTheme);
      }
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  return (
    <button
      onClick={toggleTheme}
      className="theme-toggle"
      title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
      aria-label={`Current theme: ${theme}. Click to switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
    >
      {theme === 'light' ? <Moon size={16} /> : <Sun size={16} />}
      <span className="theme-toggle-label">{theme === 'light' ? 'Dark' : 'Light'}</span>
    </button>
  );
}

