/**
 * CHUTES AI API Provider Extension for pi
 *
 * Provides access to models from CHUTES AI platform (chutes.ai)
 * via their OpenAI-compatible API endpoint.
 *
 * Setup:
 * 1. Get an API key from https://chutes.ai
 * 2. Export it: export CHUTES_API_KEY=your-api-key
 * 3. Load the extension:
 *    pi -e ./path/to/pi-chutes-ai
 *    # or install as a package:
 *    pi install git:github.com/user/pi-chutes-ai
 *
 * Then use /model and search for "chutes-ai/" to see all available models.
 */

import type {
  Api,
  AssistantMessageEventStream,
  Context,
  Model,
  SimpleStreamOptions,
} from "@mariozechner/pi-ai";
import { streamSimpleOpenAICompletions } from "@mariozechner/pi-ai";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

// =============================================================================
// Constants
// =============================================================================
const CHUTES_BASE_URL = "https://llm.chutes.ai/v1";
const CHUTES_API_KEY_ENV = "CHUTES_API_KEY";
const PROVIDER_NAME = "chutes-ai";

// =============================================================================
// Reasoning models and their capabilities
// =============================================================================

// Models known to support reasoning/thinking
const REASONING_MODELS = new Set([
  "deepseek-ai/deepseek-r1",
  "deepseek-ai/deepseek-r1-0528",
  "deepseek-ai/deepseek-v3-0324",
  "deepseek-ai/deepseek-v3-250324",
  "qwen/qwq-32b",
  "qwen/qwen3-235b-a22b",
  "qwen/qwen3-coder-480b-a35b-instruct",
]);

// Models known to support image/vision input
const VISION_MODELS = new Set([
  "anthropic/claude-3-5-sonnet-20241022",
  "anthropic/claude-3-7-sonnet-20250219",
  "anthropic/claude-3-opus-20240229",
  "openai/gpt-4o",
  "openai/gpt-4o-mini",
  "google/gemini-2-5-pro-preview-03-25",
  "google/gemini-2-5-flash-preview",
  "google/gemini-2-0-flash",
  "google/gemini-2-0-flash-thinking-exp-01-21",
  "meta/llama-3-2-90b-vision-instruct",
  "meta/llama-3-2-11b-vision-instruct",
]);

// Embedding / non-chat models to skip
const SKIP_MODELS = new Set([
  // Add any embedding/reranking models here if discovered
]);

// Known context windows (tokens)
const CONTEXT_WINDOWS: Record<string, number> = {
  // Anthropic Claude
  "anthropic/claude-3-7-sonnet-20250219": 200000,
  "anthropic/claude-3-5-sonnet-20241022": 200000,
  "anthropic/claude-3-5-sonnet-20240620": 200000,
  "anthropic/claude-3-opus-20240229": 200000,
  "anthropic/claude-3-sonnet-20240229": 200000,
  "anthropic/claude-3-haiku-20240307": 200000,
  // OpenAI
  "openai/gpt-4o": 128000,
  "openai/gpt-4o-mini": 128000,
  "openai/gpt-4-turbo": 128000,
  "openai/gpt-4": 8192,
  "openai/gpt-3.5-turbo": 16385,
  // Google Gemini
  "google/gemini-2-5-pro-preview-03-25": 1048576,
  "google/gemini-2-5-flash-preview": 1048576,
  "google/gemini-2-0-flash": 1048576,
  "google/gemini-2-0-flash-thinking-exp-01-21": 1048576,
  "google/gemini-1-5-pro": 2097152,
  "google/gemini-1-5-flash": 1048576,
  // Meta Llama
  "meta/llama-4-maverick": 262144,
  "meta/llama-4-scout": 524288,
  "meta/llama-3-3-70b-instruct": 128000,
  "meta/llama-3-1-405b-instruct": 128000,
  "meta/llama-3-1-70b-instruct": 128000,
  "meta/llama-3-1-8b-instruct": 128000,
  "meta/llama-3-2-90b-vision-instruct": 128000,
  "meta/llama-3-2-11b-vision-instruct": 128000,
  // DeepSeek
  "deepseek-ai/deepseek-v3": 64000,
  "deepseek-ai/deepseek-v3-0324": 64000,
  "deepseek-ai/deepseek-v3-250324": 64000,
  "deepseek-ai/deepseek-r1": 64000,
  "deepseek-ai/deepseek-r1-0528": 64000,
  "deepseek-ai/deepseek-coder-v2": 64000,
  // Qwen
  "qwen/qwen3-235b-a22b": 128000,
  "qwen/qwen3-coder-480b-a35b-instruct": 128000,
  "qwen/qwen2-5-72b-instruct": 128000,
  "qwen/qwen2-5-32b-instruct": 128000,
  "qwen/qwq-32b": 128000,
  // Mistral
  "mistral/mistral-large": 32768,
  "mistral/mistral-medium": 32768,
  "mistral/mistral-small": 32768,
  "mistral/codestral": 32768,
  // Cohere
  "cohere/command-r-plus": 128000,
  "cohere/command-r": 128000,
  // Other
  "microsoft/phi-4": 16000,
  "nvidia/llama-3-1-nemotron-70b-instruct": 131072,
};

// Known max output tokens
const MAX_TOKENS: Record<string, number> = {
  "anthropic/claude-3-7-sonnet-20250219": 8192,
  "anthropic/claude-3-5-sonnet-20241022": 8192,
  "anthropic/claude-3-opus-20240229": 4096,
  "openai/gpt-4o": 16384,
  "openai/gpt-4o-mini": 16384,
  "deepseek-ai/deepseek-r1": 8192,
  "deepseek-ai/deepseek-r1-0528": 8192,
  "deepseek-ai/deepseek-v3": 8192,
  "deepseek-ai/deepseek-v3-0324": 8192,
  "deepseek-ai/deepseek-v3-250324": 8192,
  "google/gemini-2-5-pro-preview-03-25": 8192,
  "google/gemini-2-5-flash-preview": 8192,
  "meta/llama-4-maverick": 131072,
  "meta/llama-4-scout": 8192,
};

// =============================================================================
// Curated "featured" models - listed first in the model selector
// =============================================================================
const FEATURED_MODELS = [
  // Frontier / Flagship
  "anthropic/claude-3-7-sonnet-20250219",
  "anthropic/claude-3-5-sonnet-20241022",
  "openai/gpt-4o",
  "openai/gpt-4o-mini",
  "deepseek-ai/deepseek-r1",
  "deepseek-ai/deepseek-r1-0528",
  "deepseek-ai/deepseek-v3",
  "deepseek-ai/deepseek-v3-0324",
  // Reasoning
  "qwen/qwen3-235b-a22b",
  "qwen/qwen3-coder-480b-a35b-instruct",
  "qwen/qwq-32b",
  // Google
  "google/gemini-2-5-pro-preview-03-25",
  "google/gemini-2-5-flash-preview",
  "google/gemini-2-0-flash-thinking-exp-01-21",
  // Meta
  "meta/llama-4-maverick",
  "meta/llama-4-scout",
  "meta/llama-3-3-70b-instruct",
  "meta/llama-3-1-405b-instruct",
];

// =============================================================================
// Model building helpers
// =============================================================================

interface ChutesModelEntry {
  id: string;
  name: string;
  reasoning: boolean;
  input: ("text" | "image")[];
  contextWindow: number;
  maxTokens: number;
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
  };
  compat?: Record<string, unknown>;
}

function makeDisplayName(modelId: string): string {
  const parts = modelId.split("/");
  const provider = parts[0];
  const name = parts[parts.length - 1];
  
  // Provider display names
  const providerDisplay: Record<string, string> = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
    "meta": "Meta",
    "mistral": "Mistral",
    "cohere": "Cohere",
    "deepseek-ai": "DeepSeek",
    "qwen": "Qwen",
    "microsoft": "Microsoft",
    "nvidia": "NVIDIA",
  };
  
  const pDisplay = providerDisplay[provider] ?? provider;
  const nDisplay = name
    .replace(/-/g, " ")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
    
  return `${pDisplay} ${nDisplay}`;
}

function buildModelEntry(modelId: string): ChutesModelEntry | null {
  if (SKIP_MODELS.has(modelId)) return null;

  const isReasoning = REASONING_MODELS.has(modelId);
  const isVision = VISION_MODELS.has(modelId);
  const contextWindow = CONTEXT_WINDOWS[modelId] ?? 4096;
  const maxTokens = MAX_TOKENS[modelId] ?? Math.min(4096, contextWindow);

  const entry: ChutesModelEntry = {
    id: modelId,
    name: makeDisplayName(modelId),
    reasoning: isReasoning,
    input: isVision ? ["text", "image"] : ["text"],
    contextWindow,
    maxTokens,
    cost: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
    },
  };

  // Default compat for CHUTES models
  // CHUTES uses standard OpenAI API, so we use default compat
  entry.compat = {
    supportsReasoningEffort: isReasoning,
    supportsDeveloperRole: true,
    maxTokensField: "max_tokens",
  };

  return entry;
}

// =============================================================================
// Dynamic model discovery
// =============================================================================

interface ChutesApiModel {
  id: string;
  object: string;
  owned_by: string;
}

async function fetchChutesModels(apiKey: string): Promise<string[]> {
  try {
    const response = await fetch(`${CHUTES_BASE_URL}/models`, {
      headers: {
        Authorization: `Bearer ${apiKey}`,
        Accept: "application/json",
      },
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      console.warn(`CHUTES AI: Failed to fetch models: ${response.status} ${response.statusText}`);
      return [];
    }

    const data = (await response.json()) as { data: ChutesApiModel[] };
    return data.data?.map((m) => m.id) ?? [];
  } catch (error) {
    console.warn("CHUTES AI: Failed to fetch models:", error);
    return [];
  }
}

// =============================================================================
// Custom streaming - wraps standard openai-completions with CHUTES-specific fixes
// =============================================================================

/**
 * Custom streamSimple that wraps the standard OpenAI completions streamer.
 *
 * CHUTES AI uses standard OpenAI API, but we need to:
 * 1. Validate API key is present
 * 2. Pass the API key through options
 */
function chutesStreamSimple(
  model: Model<Api>,
  context: Context,
  options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  // Resolve API key at request time
  const chutesApiKey = process.env[CHUTES_API_KEY_ENV];

  if (!chutesApiKey) {
    throw new Error(
      `CHUTES AI: ${CHUTES_API_KEY_ENV} environment variable is not set. ` +
      `Get an API key at https://chutes.ai and export it: ` +
      `export ${CHUTES_API_KEY_ENV}=your-api-key`
    );
  }

  const modifiedOptions: SimpleStreamOptions = {
    ...options,
    apiKey: chutesApiKey,
  };

  return streamSimpleOpenAICompletions(
    model as Model<"openai-completions">,
    context,
    modifiedOptions
  );
}

// =============================================================================
// Extension Entry Point
// =============================================================================

export default function (pi: ExtensionAPI) {
  // Build the curated model list
  const modelMap = new Map<string, ChutesModelEntry>();

  // Add featured models first (preserves order in selector)
  for (const id of FEATURED_MODELS) {
    const entry = buildModelEntry(id);
    if (entry) modelMap.set(id, entry);
  }

  // Register with curated models immediately
  const curatedModels = Array.from(modelMap.values());
  pi.registerProvider(PROVIDER_NAME, {
    baseUrl: CHUTES_BASE_URL,
    apiKey: CHUTES_API_KEY_ENV,
    api: "openai-completions",
    authHeader: true,
    models: curatedModels,
    streamSimple: chutesStreamSimple,
  });

  // On session start, discover additional models from the API
  pi.on("session_start", async (_event: any, ctx: any) => {
    const apiKey = process.env[CHUTES_API_KEY_ENV];
    if (!apiKey) return; // API key not available, skip model discovery

    // Fetch live model list
    const liveModelIds = await fetchChutesModels(apiKey);
    if (liveModelIds.length === 0) return;

    let newModelsAdded = 0;
    for (const id of liveModelIds) {
      if (modelMap.has(id)) continue; // Already known

      const entry = buildModelEntry(id);
      if (entry) {
        modelMap.set(id, entry);
        newModelsAdded++;
      }
    }

    // Re-register with full model list if we found new ones
    if (newModelsAdded > 0) {
      const allModels = Array.from(modelMap.values());
      ctx.modelRegistry.registerProvider(PROVIDER_NAME, {
        baseUrl: CHUTES_BASE_URL,
        apiKey: CHUTES_API_KEY_ENV,
        api: "openai-completions",
        authHeader: true,
        models: allModels,
        streamSimple: chutesStreamSimple,
      });
    }
  });
}