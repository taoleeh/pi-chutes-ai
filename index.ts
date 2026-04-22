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
// Model entry type
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

// =============================================================================
// Helper to build display name from model ID
// =============================================================================

function makeDisplayName(modelId: string): string {
  // Strip -TEE suffix for cleaner display
  const cleanId = modelId.replace(/-TEE$/, "");

  const parts = cleanId.split("/");
  const provider = parts[0];
  const name = parts[parts.length - 1];

  // Provider display names
  const providerDisplay: Record<string, string> = {
    anthropic: "Anthropic",
    openai: "OpenAI",
    google: "Google",
    meta: "Meta",
    mistral: "Mistral",
    cohere: "Cohere",
    "deepseek-ai": "DeepSeek",
    qwen: "Qwen",
    microsoft: "Microsoft",
    nvidia: "NVIDIA",
    moonshotai: "Moonshot AI",
    minimaxai: "MiniMax",
    "zai-org": "Zhipu AI",
    tngtech: "TNG",
    nousresearch: "NousResearch",
    xiaomimimo: "Xiaomi MiMo",
    unsloth: "Unsloth",
    "rednote-hilab": "Rednote",
  };

  const pDisplay = providerDisplay[provider] ?? provider;
  const nDisplay = name
    .replace(/-/g, " ")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

  return `${pDisplay} ${nDisplay}`;
}

// =============================================================================
// Known models with full metadata (sourced from Chutes AI API)
// =============================================================================

const KNOWN_MODELS: ChutesModelEntry[] = [
  // ── DeepSeek ──────────────────────────────────────────────────────────────
  {
    id: "deepseek-ai/DeepSeek-V3.2-TEE",
    name: "DeepSeek V3.2",
    reasoning: true,
    input: ["text"],
    contextWindow: 131072,
    maxTokens: 65536,
    cost: { input: 0.28, output: 0.42, cacheRead: 0.14, cacheWrite: 0.28 },
  },
  {
    id: "deepseek-ai/DeepSeek-V3.1-TEE",
    name: "DeepSeek V3.1",
    reasoning: true,
    input: ["text"],
    contextWindow: 163840,
    maxTokens: 65536,
    cost: { input: 0.27, output: 1.0, cacheRead: 0.135, cacheWrite: 0.27 },
  },
  {
    id: "deepseek-ai/DeepSeek-V3-0324-TEE",
    name: "DeepSeek V3 0324",
    reasoning: false,
    input: ["text"],
    contextWindow: 163840,
    maxTokens: 65536,
    cost: { input: 0.25, output: 1.0, cacheRead: 0.125, cacheWrite: 0.25 },
  },
  {
    id: "deepseek-ai/DeepSeek-R1-0528-TEE",
    name: "DeepSeek R1 0528",
    reasoning: true,
    input: ["text"],
    contextWindow: 163840,
    maxTokens: 65536,
    cost: { input: 0.45, output: 2.15, cacheRead: 0.225, cacheWrite: 0.45 },
  },
  {
    id: "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    name: "DeepSeek R1 Distill Llama 70B",
    reasoning: true,
    input: ["text"],
    contextWindow: 131072,
    maxTokens: 131072,
    cost: {
      input: 0.0272,
      output: 0.1087,
      cacheRead: 0.0136,
      cacheWrite: 0.0272,
    },
  },
  {
    id: "tngtech/DeepSeek-TNG-R1T2-Chimera-TEE",
    name: "TNG DeepSeek R1T2 Chimera",
    reasoning: true,
    input: ["text"],
    contextWindow: 163840,
    maxTokens: 163840,
    cost: { input: 0.3, output: 1.1, cacheRead: 0.15, cacheWrite: 0.3 },
  },

  // ── Qwen ───────────────────────────────────────────────────────────────────
  {
    id: "Qwen/Qwen3.5-397B-A17B-TEE",
    name: "Qwen 3.5 397B A17B",
    reasoning: true,
    input: ["text", "image"],
    contextWindow: 262144,
    maxTokens: 65536,
    cost: { input: 0.39, output: 2.34, cacheRead: 0.195, cacheWrite: 0.39 },
  },
  {
    id: "Qwen/Qwen3-235B-A22B-Thinking-2507",
    name: "Qwen 3 235B A22B Thinking 2507",
    reasoning: true,
    input: ["text"],
    contextWindow: 262144,
    maxTokens: 262144,
    cost: { input: 0.11, output: 0.6, cacheRead: 0.055, cacheWrite: 0.11 },
  },
  {
    id: "Qwen/Qwen3-235B-A22B-Instruct-2507-TEE",
    name: "Qwen 3 235B A22B Instruct 2507",
    reasoning: false,
    input: ["text"],
    contextWindow: 262144,
    maxTokens: 65536,
    cost: { input: 0.1, output: 0.6, cacheRead: 0.05, cacheWrite: 0.1 },
  },
  {
    id: "Qwen/Qwen3-Coder-Next-TEE",
    name: "Qwen 3 Coder Next",
    reasoning: false,
    input: ["text"],
    contextWindow: 262144,
    maxTokens: 65536,
    cost: { input: 0.12, output: 0.75, cacheRead: 0.06, cacheWrite: 0.12 },
  },
  {
    id: "Qwen/Qwen3-Next-80B-A3B-Instruct",
    name: "Qwen 3 Next 80B A3B Instruct",
    reasoning: false,
    input: ["text"],
    contextWindow: 262144,
    maxTokens: 262144,
    cost: { input: 0.1, output: 0.8, cacheRead: 0.05, cacheWrite: 0.1 },
  },
  {
    id: "Qwen/Qwen3-32B-TEE",
    name: "Qwen 3 32B",
    reasoning: true,
    input: ["text"],
    contextWindow: 40960,
    maxTokens: 40960,
    cost: { input: 0.08, output: 0.24, cacheRead: 0.04, cacheWrite: 0.08 },
  },
  {
    id: "Qwen/Qwen3-30B-A3B",
    name: "Qwen 3 30B A3B",
    reasoning: true,
    input: ["text"],
    contextWindow: 40960,
    maxTokens: 40960,
    cost: { input: 0.06, output: 0.22, cacheRead: 0.03, cacheWrite: 0.06 },
  },
  {
    id: "Qwen/Qwen2.5-72B-Instruct",
    name: "Qwen 2.5 72B Instruct",
    reasoning: false,
    input: ["text"],
    contextWindow: 32768,
    maxTokens: 32768,
    cost: {
      input: 0.2989,
      output: 1.1957,
      cacheRead: 0.14945,
      cacheWrite: 0.2989,
    },
  },
  {
    id: "Qwen/Qwen2.5-Coder-32B-Instruct",
    name: "Qwen 2.5 Coder 32B Instruct",
    reasoning: false,
    input: ["text"],
    contextWindow: 32768,
    maxTokens: 32768,
    cost: {
      input: 0.0272,
      output: 0.1087,
      cacheRead: 0.0136,
      cacheWrite: 0.0272,
    },
  },
  {
    id: "Qwen/Qwen2.5-VL-32B-Instruct",
    name: "Qwen 2.5 VL 32B Instruct",
    reasoning: false,
    input: ["text", "image"],
    contextWindow: 16384,
    maxTokens: 16384,
    cost: {
      input: 0.0543,
      output: 0.2174,
      cacheRead: 0.02715,
      cacheWrite: 0.0543,
    },
  },
  {
    id: "Qwen/Qwen3Guard-Gen-0.6B",
    name: "Qwen 3 Guard Gen 0.6B",
    reasoning: false,
    input: ["text"],
    contextWindow: 32768,
    maxTokens: 32768,
    cost: { input: 0.01, output: 0.0109, cacheRead: 0.005, cacheWrite: 0.01 },
  },

  // ── Moonshot AI ────────────────────────────────────────────────────────────
  {
    id: "moonshotai/Kimi-K2.5-TEE",
    name: "Moonshot AI Kimi K2.5",
    reasoning: true,
    input: ["text", "image"],
    contextWindow: 262144,
    maxTokens: 65535,
    cost: {
      input: 0.3827,
      output: 1.72,
      cacheRead: 0.19135,
      cacheWrite: 0.3827,
    },
  },
  {
    id: "moonshotai/Kimi-K2.6-TEE",
    name: "Moonshot AI Kimi K2.6",
    reasoning: true,
    input: ["text", "image"],
    contextWindow: 262144,
    maxTokens: 65535,
    cost: { input: 0.95, output: 4.0, cacheRead: 0.475, cacheWrite: 0.95 },
  },

  // ── OpenAI ─────────────────────────────────────────────────────────────────
  {
    id: "openai/gpt-oss-120b-TEE",
    name: "OpenAI GPT OSS 120B",
    reasoning: true,
    input: ["text"],
    contextWindow: 131072,
    maxTokens: 65536,
    cost: { input: 0.09, output: 0.36, cacheRead: 0.045, cacheWrite: 0.09 },
  },

  // ── Zhipu AI (GLM) ─────────────────────────────────────────────────────────
  {
    id: "zai-org/GLM-5.1-TEE",
    name: "Zhipu AI GLM 5.1",
    reasoning: true,
    input: ["text"],
    contextWindow: 202752,
    maxTokens: 65535,
    cost: { input: 0.95, output: 3.15, cacheRead: 0.475, cacheWrite: 0.95 },
  },
  {
    id: "zai-org/GLM-5-TEE",
    name: "Zhipu AI GLM 5",
    reasoning: true,
    input: ["text"],
    contextWindow: 202752,
    maxTokens: 65535,
    cost: { input: 0.95, output: 3.15, cacheRead: 0.475, cacheWrite: 0.95 },
  },
  {
    id: "zai-org/GLM-5-Turbo",
    name: "Zhipu AI GLM 5 Turbo",
    reasoning: true,
    input: ["text"],
    contextWindow: 202752,
    maxTokens: 65535,
    cost: {
      input: 0.4891,
      output: 1.9565,
      cacheRead: 0.24455,
      cacheWrite: 0.4891,
    },
  },
  {
    id: "zai-org/GLM-4.7-TEE",
    name: "Zhipu AI GLM 4.7",
    reasoning: true,
    input: ["text"],
    contextWindow: 202752,
    maxTokens: 65535,
    cost: { input: 0.39, output: 1.75, cacheRead: 0.195, cacheWrite: 0.39 },
  },
  {
    id: "zai-org/GLM-4.7-FP8",
    name: "Zhipu AI GLM 4.7 FP8",
    reasoning: true,
    input: ["text"],
    contextWindow: 202752,
    maxTokens: 65535,
    cost: {
      input: 0.2989,
      output: 1.1957,
      cacheRead: 0.14945,
      cacheWrite: 0.2989,
    },
  },
  {
    id: "zai-org/GLM-4.6V",
    name: "Zhipu AI GLM 4.6V",
    reasoning: true,
    input: ["text", "image"],
    contextWindow: 131072,
    maxTokens: 65536,
    cost: { input: 0.3, output: 0.9, cacheRead: 0.15, cacheWrite: 0.3 },
  },

  // ── MiniMax ────────────────────────────────────────────────────────────────
  {
    id: "MiniMaxAI/MiniMax-M2.5-TEE",
    name: "MiniMax M2.5",
    reasoning: true,
    input: ["text"],
    contextWindow: 196608,
    maxTokens: 65536,
    cost: { input: 0.118, output: 0.99, cacheRead: 0.059, cacheWrite: 0.118 },
  },

  // ── Xiaomi MiMo ────────────────────────────────────────────────────────────
  {
    id: "XiaomiMiMo/MiMo-V2-Flash-TEE",
    name: "Xiaomi MiMo V2 Flash",
    reasoning: false,
    input: ["text"],
    contextWindow: 262144,
    maxTokens: 65536,
    cost: { input: 0.09, output: 0.29, cacheRead: 0.045, cacheWrite: 0.09 },
  },

  // ── NousResearch ───────────────────────────────────────────────────────────
  {
    id: "NousResearch/DeepHermes-3-Mistral-24B-Preview",
    name: "NousResearch DeepHermes 3 Mistral 24B",
    reasoning: false,
    input: ["text"],
    contextWindow: 32768,
    maxTokens: 32768,
    cost: {
      input: 0.0245,
      output: 0.0978,
      cacheRead: 0.01225,
      cacheWrite: 0.0245,
    },
  },
  {
    id: "NousResearch/Hermes-4-14B",
    name: "NousResearch Hermes 4 14B",
    reasoning: true,
    input: ["text"],
    contextWindow: 40960,
    maxTokens: 40960,
    cost: {
      input: 0.0136,
      output: 0.0543,
      cacheRead: 0.0068,
      cacheWrite: 0.0136,
    },
  },

  // ── Google Gemma (via Unsloth) ──────────────────────────────────────────────
  {
    id: "unsloth/gemma-3-27b-it",
    name: "Unsloth Gemma 3 27B IT",
    reasoning: false,
    input: ["text", "image"],
    contextWindow: 128000,
    maxTokens: 65536,
    cost: {
      input: 0.0272,
      output: 0.1087,
      cacheRead: 0.0136,
      cacheWrite: 0.0272,
    },
  },
  {
    id: "unsloth/gemma-3-12b-it",
    name: "Unsloth Gemma 3 12B IT",
    reasoning: false,
    input: ["text", "image"],
    contextWindow: 131072,
    maxTokens: 131072,
    cost: { input: 0.03, output: 0.1, cacheRead: 0.015, cacheWrite: 0.03 },
  },
  {
    id: "unsloth/gemma-3-4b-it",
    name: "Unsloth Gemma 3 4B IT",
    reasoning: false,
    input: ["text", "image"],
    contextWindow: 96000,
    maxTokens: 96000,
    cost: { input: 0.01, output: 0.0272, cacheRead: 0.005, cacheWrite: 0.01 },
  },

  // ── Mistral (via Unsloth) ─────────────────────────────────────────────────
  {
    id: "unsloth/Mistral-Nemo-Instruct-2407",
    name: "Unsloth Mistral Nemo Instruct 2407",
    reasoning: false,
    input: ["text"],
    contextWindow: 131072,
    maxTokens: 131072,
    cost: { input: 0.02, output: 0.04, cacheRead: 0.01, cacheWrite: 0.02 },
  },

  // ── Llama (via Unsloth) ────────────────────────────────────────────────────
  {
    id: "unsloth/Llama-3.2-3B-Instruct",
    name: "Unsloth Llama 3.2 3B Instruct",
    reasoning: false,
    input: ["text"],
    contextWindow: 16384,
    maxTokens: 16384,
    cost: { input: 0.01, output: 0.0136, cacheRead: 0.005, cacheWrite: 0.01 },
  },
  {
    id: "unsloth/Llama-3.2-1B-Instruct",
    name: "Unsloth Llama 3.2 1B Instruct",
    reasoning: false,
    input: ["text"],
    contextWindow: 16384,
    maxTokens: 16384,
    cost: { input: 0.01, output: 0.0109, cacheRead: 0.005, cacheWrite: 0.01 },
  },

  // ── Rednote ────────────────────────────────────────────────────────────────
  {
    id: "rednote-hilab/dots.ocr",
    name: "Rednote Dots OCR",
    reasoning: false,
    input: ["text", "image"],
    contextWindow: 131072,
    maxTokens: 131072,
    cost: { input: 0.01, output: 0.0109, cacheRead: 0.005, cacheWrite: 0.01 },
  },
];

// =============================================================================
// Curated "featured" models - listed first in the model selector
// =============================================================================
const FEATURED_MODEL_IDS = [
  // Frontier / Flagship
  "moonshotai/Kimi-K2.5-TEE",
  "deepseek-ai/DeepSeek-R1-0528-TEE",
  "deepseek-ai/DeepSeek-V3.2-TEE",
  "Qwen/Qwen3.5-397B-A17B-TEE",
  "Qwen/Qwen3-235B-A22B-Thinking-2507",
  "zai-org/GLM-5.1-TEE",
  "deepseek-ai/DeepSeek-V3.1-TEE",
  "tngtech/DeepSeek-TNG-R1T2-Chimera-TEE",
  // Strong general-purpose
  "MiniMaxAI/MiniMax-M2.5-TEE",
  "openai/gpt-oss-120b-TEE",
  "Qwen/Qwen3-Coder-Next-TEE",
  "XiaomiMiMo/MiMo-V2-Flash-TEE",
  // Cost-effective
  "Qwen/Qwen3-32B-TEE",
  "Qwen/Qwen3-30B-A3B",
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
  // Vision models
  "zai-org/GLM-4.6V",
  "Qwen/Qwen2.5-VL-32B-Instruct",
  "unsloth/gemma-3-27b-it",
];

// =============================================================================
// Dynamic model discovery helpers
// =============================================================================

// Build a lookup from known model IDs to their full entries
const knownModelMap = new Map<string, ChutesModelEntry>();
for (const entry of KNOWN_MODELS) {
  entry.compat = {
    supportsReasoningEffort: entry.reasoning,
    supportsDeveloperRole: true,
    maxTokensField: "max_tokens",
  };
  knownModelMap.set(entry.id, entry);
}

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
      console.warn(
        `CHUTES AI: Failed to fetch models: ${response.status} ${response.statusText}`,
      );
      return [];
    }

    const data = (await response.json()) as { data: ChutesApiModel[] };
    return data.data?.map((m) => m.id) ?? [];
  } catch (error) {
    console.warn("CHUTES AI: Failed to fetch models:", error);
    return [];
  }
}

/**
 * Build a minimal model entry for a dynamically discovered model
 * that isn't in our KNOWN_MODELS list. We use conservative defaults.
 */
function buildFallbackEntry(modelId: string): ChutesModelEntry {
  const entry: ChutesModelEntry = {
    id: modelId,
    name: makeDisplayName(modelId),
    reasoning: false,
    input: ["text"],
    contextWindow: 40960,
    maxTokens: 4096,
    cost: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
    },
  };

  entry.compat = {
    supportsReasoningEffort: false,
    supportsDeveloperRole: true,
    maxTokensField: "max_tokens",
  };

  return entry;
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
        `export ${CHUTES_API_KEY_ENV}=your-api-key`,
    );
  }

  const modifiedOptions: SimpleStreamOptions = {
    ...options,
    apiKey: chutesApiKey,
  };

  return streamSimpleOpenAICompletions(
    model as Model<"openai-completions">,
    context,
    modifiedOptions,
  );
}

// =============================================================================
// Extension Entry Point
// =============================================================================

export default function (pi: ExtensionAPI) {
  // Build the curated model list: featured models first, then remaining known models
  const modelMap = new Map<string, ChutesModelEntry>();

  // Add featured models first (preserves order in selector)
  for (const id of FEATURED_MODEL_IDS) {
    const entry = knownModelMap.get(id);
    if (entry) modelMap.set(id, entry);
  }

  // Add remaining known models
  for (const entry of KNOWN_MODELS) {
    if (!modelMap.has(entry.id)) {
      modelMap.set(entry.id, entry);
    }
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

      // Check if we have full metadata for this model
      const known = knownModelMap.get(id);
      if (known) {
        modelMap.set(id, known);
      } else {
        // Build a fallback entry with conservative defaults
        modelMap.set(id, buildFallbackEntry(id));
      }
      newModelsAdded++;
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
