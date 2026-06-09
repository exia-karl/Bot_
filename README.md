# Botzone Mahjong LLM Example Repository

This repository is a teaching/submission template for building an LLM-based Mahjong bot that can run with Botzone LocalAI.

## What is included

- A runnable Mahjong LLM bot example (`llm_bot.py`, `llm_bot_cn.py`)
- A LocalAI adapter for Botzone polling/forwarding (`local_ai/local_ai.py`)
- A unified OpenAI-compatible API client layer (`api_config/conf.py`)
- Deployment/tutorial docs for local model serving and full bot setup

## Project structure

- `local_bots/mahjong/llm_bot.py`: English prompt bot
- `local_bots/mahjong/llm_bot_cn.py`: Chinese prompt bot
- `local_ai/local_ai.py`: LocalAI adapter loop
- `local_bots/mahjong/policy_llm.py`: shared policy helpers (retry + answer parsing)
- `api_config/conf.py`: unified OpenAI-compatible client wrapper
- `api_config/llm_config.json`: model endpoint config
- `docs/LOCAL_MODEL_VLLM_GUIDE.md`: local model serving guide (vLLM)
- `docs/LLM_BOT_BUILD_TUTORIAL.md`: full LLM bot build tutorial

## Installation

```bash
pip install -r requirements.txt
```

## API config

Edit `api_config/llm_config.json`:

```json
{
  "llm_name": "your-model-name",
  "api_base": "https://your-openai-compatible-endpoint/v1/chat/completions",
  "api_key": "your-api-key"
}
```

All backends are switched only by config changes (`llm_name`, `api_base`, `api_key`).

Config loading order:
1. `LLM_CONFIG_PATH` env var
2. `/data/llm_config.json` (Botzone userfile mount)
3. `api_config/llm_config.json` (local development)

## LocalAI run example

Windows (PowerShell, same style as verified local run):

```powershell
python local_ai\local_ai.py `
  --localai-url "https://botzone.org.cn/api/<bot_uid>/<bot_secret>/localai" `
  --bot-cwd . `
  --bot-cmd python local_bots\mahjong\llm_bot.py
```

Windows (Chinese prompt bot):

```powershell
python local_ai\local_ai.py `
  --localai-url "https://botzone.org.cn/api/<bot_uid>/<bot_secret>/localai" `
  --bot-cwd . `
  --bot-cmd python local_bots\mahjong\llm_bot_cn.py
```

Parameter notes:
- `<bot_uid>`: your Botzone bot id
- `<bot_secret>`: your LocalAI access secret/token
- `--bot-cwd .`: run bot from repository root (recommended)
- `--bot-cmd ...`: target bot entry script

Cross-platform example:

```bash
python local_ai/local_ai.py \
  --localai-url "https://www.botzone.org.cn/api/<uid>/<secret>/localai" \
  --bot-cmd python local_bots/mahjong/llm_bot.py \
  --bot-cwd .
```

For Chinese prompt bot:

```bash
python local_ai/local_ai.py \
  --localai-url "https://www.botzone.org.cn/api/<uid>/<secret>/localai" \
  --bot-cmd python local_bots/mahjong/llm_bot_cn.py \
  --bot-cwd .
```

## Documentation

- Build tutorial: `docs/LLM_BOT_BUILD_TUTORIAL.md`
- Local model deployment (vLLM): `docs/LOCAL_MODEL_VLLM_GUIDE.md`
- LocalAI protocol details: `local_ai/README.md`

## About Botzone-ALE

This repository does not include Botzone-ALE itself.
Please install and read from:
- https://github.com/AMysteriousBeing/Botzone-ALE
