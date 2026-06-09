# LLM Mahjong Bot Build Tutorial

This tutorial describes how to build and configure a full LLM-based Botzone Mahjong bot in this repository structure.

## 1. Architecture split

For maintainability, split the solution into two layers:

1. Botzone interaction layer
- receives Botzone protocol input
- updates game observation/state
- calls LLM service with prompt + valid actions
- parses LLM output to a valid action
- prints action back to Botzone

Current files:
- `local_bots/mahjong/llm_bot.py`
- `local_bots/mahjong/llm_bot_cn.py`
- `local_bots/mahjong/botzone_engine.py` (shared protocol loop)
- `local_bots/mahjong/policy_llm.py` (shared LLM retry + answer parsing utilities)

2. Model service layer
- provides OpenAI-compatible `/v1/chat/completions`
- can be remote API provider or local model server

Current files:
- `api_config/conf.py`
- `api_config/llm_config.json`

Language note:
- This repository documentation is mainly in English for encoding stability across environments.
- Chinese prompt gameplay logic is available in `local_bots/mahjong/llm_bot_cn.py`.

## 2. Minimal workflow

1. Configure model endpoint in `api_config/llm_config.json`.
2. Start LocalAI adapter:

```bash
python local_ai/local_ai.py   --localai-url "https://www.botzone.org.cn/api/<uid>/<secret>/localai"   --bot-cmd python local_bots/mahjong/llm_bot.py   --bot-cwd .
```

3. Observe logs and verify legal actions are returned.

## 3. API configuration model

Current config format:

```json
{
  "llm_name": "your-model-name",
  "api_base": "https://your-openai-compatible-endpoint/v1/chat/completions",
  "api_key": "your-api-key"
}
```

All model backends should be switched only by config change.

## 4. How response parsing works

In `llm_bot*.py`, the action parser:
- tries to extract `Answer:` / `??:` section first
- falls back to fuzzy matching against valid action list
- retries up to a small number of rounds when invalid answer appears

Recommended model output format:

```text
Evaluation: ...
Reason: ...
Answer: Play W5
```

or Chinese variant.

## 5. How to customize for your own bot

1. Keep protocol loop unchanged (`if __name__ == "__main__"` block).
2. Replace only prompt strategy and output parsing policy.
3. Keep strict final action normalization against legal action mask.
4. Keep fallback action (usually `PASS`) for robustness.

In the current refactor, protocol loop is extracted into:
- `local_bots/mahjong/botzone_engine.py`
- Shared policy helpers are extracted into:
- `local_bots/mahjong/policy_llm.py`

So the recommended change surface is:
- `obs2response(...)` in `llm_bot.py` / `llm_bot_cn.py`
- prompt template and how `infer_action_with_retry(...)` is called

## 6. Using local model service

See:
- `docs/LOCAL_MODEL_VLLM_GUIDE.md`

That document explains how to run a vLLM OpenAI-compatible local endpoint and bind it with this bot config.

## 7. Suggested refactor direction (optional)

If you want cleaner packaging for teaching/demo, split bot files into modules:

- `bot_runtime/engine.py`: protocol loop
- `bot_runtime/state_adapter.py`: request -> observation updates
- `bot_runtime/policy_llm.py`: prompt + llm call + parse
- `bot_runtime/action_codec.py`: action canonicalization

Keep `local_bots/mahjong/llm_bot.py` as a thin startup wrapper.

## 8. Checklist before submission

1. No hardcoded secret key in repository.
2. `llm_config.json` uses placeholders.
3. At least one successful local dry run with LocalAI adapter.
4. README and docs reflect your actual file paths and commands.

## 9. Bot Development Guide (What to Change / What to Keep)

This section is a practical guide for students who want to improve the bot.

### 9.1 Component responsibilities

- `local_bots/mahjong/botzone_engine.py`
  - Handles Botzone stdin/stdout protocol loop.
  - Receives requests and sends final action text back.
  - Usually **keep unchanged** unless you are debugging protocol-level issues.

- `local_bots/mahjong/llm_bot.py` / `llm_bot_cn.py`
  - Main strategy entry (`obs2response`).
  - Builds prompt and state text for LLM decision.
  - This is the **primary place to optimize decision quality**.

- `local_bots/mahjong/policy_llm.py`
  - Retry logic and answer/action parsing.
  - Converts raw LLM output to legal Botzone action candidate.
  - Good place to improve robustness.

- `api_config/conf.py`
  - Unified OpenAI-compatible API call path.
  - Use this layer to switch between remote API and local model service.

### 9.2 Recommended optimization priority

1. Prompt quality in `obs2response`
  - clarify objectives (win-rate, safety, fan potential)
  - enforce strict answer format

2. Observation text quality
  - reorganize `observation_llm` summary
  - remove low-value noise and keep key tactical signals

3. Output robustness
  - improve retry policy and illegal-action fallback in `policy_llm.py`

### 9.3 What should not be changed casually

- Botzone protocol output format (`PASS`, `PLAY X`, `HU`, etc.)
- Core state transition logic inside `request2obs` (unless fully understood)
- LocalAI adapter protocol handshake/sentinel behavior

### 9.4 Safe iteration workflow

1. Start with prompt changes only.
2. Run local LocalAI debugging for several rounds.
3. If stable, then adjust parsing/retry rules.
4. Keep fallback action path (usually `PASS`) for safety.
