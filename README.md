# Pi CHUTES AI Extension

A Pi extension that provides access to AI models via the CHUTES AI platform through their OpenAI-compatible API.

## Features

- Access models from Anthropic, OpenAI, Google, Meta, DeepSeek, Qwen, Mistral, and more
- OpenAI-compatible API endpoint
- Dynamic model discovery
- Reasoning/thinking support for compatible models
- Vision support for multimodal models

## Setup

1. Get an API key from [https://chutes.ai](https://chutes.ai)

2. Export your API key:
   ```bash
   export CHUTES_API_KEY=your-api-key
   ```

3. Load the extension:
   ```bash
   # Direct loading
   pi -e ./path/to/pi-chutes-ai

   # Or install as a package
   pi install git:github.com/user/pi-chutes-ai
   ```

4. Use `/model` in Pi and search for models prefixed with "chutes-ai/"

## Featured Models

- **Anthropic**: Claude 3.7 Sonnet, Claude 3.5 Sonnet, Claude 3 Opus
- **OpenAI**: GPT-4o, GPT-4o-mini
- **DeepSeek**: DeepSeek-V3, DeepSeek-R1
- **Google**: Gemini 2.5 Pro, Gemini 2.5 Flash
- **Meta**: Llama 4 Maverick, Llama 4 Scout, Llama 3.3 70B
- **Qwen**: Qwen3 235B, Qwen3 Coder, QwQ-32B

## Configuration

The extension uses the following environment variable:

| Variable | Description |
|----------|-------------|
| `CHUTES_API_KEY` | Your CHUTES AI API key |

## Compatibility

This extension works alongside the NVIDIA NIM extension and other Pi providers. Use `/model` to switch between them.

## License

MIT
