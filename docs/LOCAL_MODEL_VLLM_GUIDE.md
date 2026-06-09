# Local Model Serving Guide (vLLM, OpenAI-Compatible)

This guide explains how to run a local model service that is compatible with OpenAI Chat Completions, so it can be used directly by this Mahjong bot repository.

## 1. Goal

The bot code only assumes one thing:
- a `POST` endpoint that follows OpenAI-compatible `/v1/chat/completions` protocol.

So both of these are supported in the same way:
- remote API provider
- local model server (e.g., vLLM)

## 2. Environment (example)

You need:
- Python 3.9+
- CUDA GPU runtime if serving large models with GPU
- enough GPU memory for your target model

Install vLLM (example):

```bash
pip install vllm
```

## 3. Start a local OpenAI-compatible server

Example command (Qwen model as example):

```bash
python -m vllm.entrypoints.openai.api_server   --model Qwen/Qwen2.5-7B-Instruct   --host 0.0.0.0   --port 8000
```

After startup, your API endpoint is usually:

```text
http://127.0.0.1:8000/v1/chat/completions
```

If you configure an API key on your serving side, keep it consistent with `llm_config.json`.

## 4. Configure this repository to use local vLLM

Edit `api_config/llm_config.json`:

```json
{
  "llm_name": "Qwen/Qwen2.5-7B-Instruct",
  "api_base": "http://127.0.0.1:8000/v1/chat/completions",
  "api_key": "EMPTY_OR_YOUR_KEY"
}
```

No other code changes are needed.

## 5. Quick connectivity check

Use curl (or any HTTP client):

```bash
curl http://127.0.0.1:8000/v1/chat/completions   -H "Content-Type: application/json"   -H "Authorization: Bearer EMPTY_OR_YOUR_KEY"   -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": [{"type":"text","text":"You are a helpful assistant."}]},
      {"role": "user", "content": [{"type":"text","text":"Say hello."}]}
    ],
    "stream": false,
    "response_format": {"type": "text"}
  }'
```

If you receive a JSON response with `choices`, the bot can use this endpoint.

## 6. Typical issues

1. Connection refused:
- server not started
- wrong host/port in `api_base`

2. Model mismatch error:
- `llm_name` in `llm_config.json` differs from served model id

3. Timeout:
- model too large / first token too slow
- increase hardware resources or switch to smaller model

4. OOM on startup or inference:
- use smaller model
- reduce context length
- adjust serving parameters

## 7. Notes for this repo

- `api_config/conf.py` already supports unified OpenAI-compatible calls.
- config loading order:
  1. `LLM_CONFIG_PATH` env var
  2. `/data/llm_config.json` (Botzone userfile mount)
  3. `api_config/llm_config.json` (local dev)

This means you can develop locally and deploy to Botzone with the same code.
