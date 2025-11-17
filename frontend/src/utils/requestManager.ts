/**
 * Request manager for handling API requests with cancellation and deduplication.
 */

class RequestManager {
  private pendingRequests = new Map<string, AbortController>();

  /**
   * Create a cancellable fetch request.
   * If a request with the same key is already pending, it will be cancelled.
   */
  async fetch(
    url: string,
    options: RequestInit = {},
    key?: string
  ): Promise<Response> {
    const requestKey = key || url;

    // Cancel previous request with same key
    if (this.pendingRequests.has(requestKey)) {
      this.pendingRequests.get(requestKey)?.abort();
    }

    // Create new abort controller
    const controller = new AbortController();
    this.pendingRequests.set(requestKey, controller);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });

      // Remove from pending when complete
      this.pendingRequests.delete(requestKey);
      return response;
    } catch (error: any) {
      // Remove from pending on error (unless it was an abort)
      if (error.name !== 'AbortError') {
        this.pendingRequests.delete(requestKey);
      }
      throw error;
    }
  }

  /**
   * Cancel a specific request by key.
   */
  cancel(key: string): void {
    if (this.pendingRequests.has(key)) {
      this.pendingRequests.get(key)?.abort();
      this.pendingRequests.delete(key);
    }
  }

  /**
   * Cancel all pending requests.
   */
  cancelAll(): void {
    this.pendingRequests.forEach((controller) => controller.abort());
    this.pendingRequests.clear();
  }

  /**
   * Check if a request is pending.
   */
  isPending(key: string): boolean {
    return this.pendingRequests.has(key);
  }
}

// Export singleton instance
export const requestManager = new RequestManager();
export default requestManager;

