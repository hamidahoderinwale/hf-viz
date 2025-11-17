/**
 * Advanced debounce utility with request cancellation support.
 */

export interface DebouncedFunction<T extends (...args: any[]) => any> {
  (...args: Parameters<T>): void;
  cancel: () => void;
  flush: () => void;
}

/**
 * Creates a debounced function that delays invoking func until after wait milliseconds
 * have elapsed since the last time the debounced function was invoked.
 * 
 * @param func The function to debounce
 * @param wait The number of milliseconds to delay
 * @param immediate If true, trigger the function on the leading edge instead of trailing
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number,
  immediate = false
): DebouncedFunction<T> {
  let timeout: NodeJS.Timeout | null = null;
  let lastArgs: Parameters<T> | null = null;
  let cancelled = false;

  const debounced = function (this: any, ...args: Parameters<T>) {
    const context = this;
    lastArgs = args;
    cancelled = false;

    const later = () => {
      timeout = null;
      if (!immediate && !cancelled) {
        func.apply(context, lastArgs!);
      }
    };

    const callNow = immediate && !timeout;

    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(later, wait);

    if (callNow && !cancelled) {
      func.apply(context, lastArgs!);
    }
  } as DebouncedFunction<T>;

  debounced.cancel = () => {
    if (timeout) {
      clearTimeout(timeout);
      timeout = null;
    }
    cancelled = true;
  };

  debounced.flush = () => {
    if (timeout && lastArgs) {
      clearTimeout(timeout);
      timeout = null;
      func.apply(debounced, lastArgs);
    }
  };

  return debounced;
}

/**
 * Throttle function - ensures function is called at most once per wait period.
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): DebouncedFunction<T> {
  let timeout: NodeJS.Timeout | null = null;
  let lastCall = 0;
  let lastArgs: Parameters<T> | null = null;

  const throttled = function (this: any, ...args: Parameters<T>) {
    const context = this;
    const now = Date.now();
    lastArgs = args;

    const later = () => {
      lastCall = Date.now();
      timeout = null;
      func.apply(context, lastArgs!);
    };

    if (!timeout) {
      if (now - lastCall >= wait) {
        later();
      } else {
        timeout = setTimeout(later, wait - (now - lastCall));
      }
    }
  } as DebouncedFunction<T>;

  throttled.cancel = () => {
    if (timeout) {
      clearTimeout(timeout);
      timeout = null;
    }
  };

  throttled.flush = () => {
    if (timeout && lastArgs) {
      clearTimeout(timeout);
      timeout = null;
      func.apply(throttled, lastArgs);
    }
  };

  return throttled;
}

