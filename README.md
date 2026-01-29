# LiteLLM Config Generator

Automatically fetches available models from multiple AI providers and generates a LiteLLM config file.

## Supported Providers

- Anthropic
- OpenAI
- Gemini
- xAI
- Perplexity
- ElevenLabs
- Ollama
- vLLM

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your API keys:

```
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
XAI_API_KEY=your_key
PERPLEXITYAI_API_KEY=your_key
ELEVEN_LABS_API_KEY=your_key
OLLAMA_API_BASE=http://localhost:11434
HOSTED_VLLM_API_BASE=http://localhost:8000
```

## Usage

```bash
# Fetch all models and start LiteLLM
python run_litellm.py

# Fetch specific provider(s) only
python run_litellm.py -p openai anthropic

# List supported providers
python run_litellm.py -l

# Debug mode
python run_litellm.py -d
```

LiteLLM will start on port 4000.

