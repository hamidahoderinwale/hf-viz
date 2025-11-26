/**
 * Constructs a Hugging Face URL for a model or collection.
 * 
 * @param modelId - Model ID in format "username/model_name" or "username/collection_name"
 * @param isCollection - Whether this is a collection (default: false)
 * @returns Properly formatted Hugging Face URL
 */
export function getHuggingFaceUrl(modelId: string, isCollection: boolean = false): string {
  if (!modelId) {
    return 'https://huggingface.co';
  }

  const encodedId = encodeURIComponent(modelId).replace(/%2F/g, '/');
  
  if (isCollection) {
    return `https://huggingface.co/collections/${encodedId}`;
  }
  
  return `https://huggingface.co/${encodedId}`;
}

/**
 * Constructs a Hugging Face API URL for a model.
 * 
 * @param modelId - Model ID in format "username/model_name"
 * @returns Properly formatted Hugging Face API URL
 */
export function getHuggingFaceApiUrl(modelId: string): string {
  if (!modelId) {
    return 'https://huggingface.co/api/models';
  }

  const encodedId = encodeURIComponent(modelId);
  return `https://huggingface.co/api/models/${encodedId}`;
}

/**
 * Constructs a Hugging Face file tree URL for a model.
 * 
 * @param modelId - Model ID in format "username/model_name"
 * @param branch - Git branch name (default: "main")
 * @returns Properly formatted Hugging Face file tree URL
 */
export function getHuggingFaceFileTreeUrl(modelId: string, branch: string = 'main'): string {
  if (!modelId) {
    return 'https://huggingface.co';
  }

  const encodedId = encodeURIComponent(modelId);
  const encodedBranch = encodeURIComponent(branch);
  return `https://huggingface.co/${encodedId}/tree/${encodedBranch}`;
}


