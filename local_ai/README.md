# Offline LocalAI Adapter Guide (For Bot Developers)

This document describes a generic `localai` interaction interface for Botzone LocalAI.
It is intended for developers who already have their own bots (in any form) and want
to connect them to Botzone for real-time `localai` interaction.

This README focuses on:

- Adapter protocol and lifecycle
- Bot integration contract
- How to connect different bot implementations

## 1. Architecture

```text
Botzone /localai  <-->  LocalAI Adapter  <-->  Your Bot
```

Responsibilities:

- Botzone `/localai`: provides match requests and accepts match responses
- LocalAI Adapter: network loop + match state + process/service glue
- Your Bot: decision engine (input game state -> output action)

## 2. Botzone LocalAI Protocol (Adapter Side)

For each poll:

1. Adapter sends one HTTP GET to `/localai`.
2. Adapter may include response headers:
   - `X-Match-<matchid>: <action>`
3. Server returns text:
   - first line: `m n`
   - next `2*m` lines: request pairs (`matchid`, `request`)
   - next `n` lines: finished match info

Adapter loop:

1. Parse new requests.
2. Route each request to the corresponding bot instance.
3. Cache returned action as pending response for next poll.
4. Remove bot instance when match is finished.

## 3. Bot Integration Contract

Your bot only needs to satisfy this contract:

- Input: one request string (for one game step)
- Output: one action string in valid game format

For Mahjong-like Botzone bots, examples are:

- `PASS`
- `PLAY W5`
- `HU`
- `PENG W3 W7`
- `CHI W2 W4`

If your bot cannot produce a valid action, adapter fallback is usually `PASS`.
Note: fallback `PASS` may be invalid for some states and can lose the match.

## 4. Supported Bot Shapes

The adapter can work with different bot shapes by adding a thin bridge layer.

### 4.1 CLI subprocess bot (recommended)

Use a command to launch your bot process. Adapter writes request lines to stdin and
reads action from stdout.

Use when your bot already has a command-line IO protocol.

### 4.2 In-process function/class bot

Add a small wrapper process:

1. read request from stdin
2. call `decide(request)` in your code
3. print action to stdout

Use when your bot is currently a Python module.

### 4.3 HTTP service bot

Add a local bridge:

1. adapter receives Botzone request
2. bridge POSTs request to your local HTTP bot service
3. bridge returns service action back to adapter

Use when your bot is already deployed as a service.

## 5. Minimal Runtime Requirements

Your package should include:

- Adapter script
- This interface README
- Your bot runtime and dependencies
- A startup command for developers

Recommended startup command template:

You can adjust retry interval with `--retry-seconds` (default: 5).

```powershell
python <adapter_script>.py `
  --localai-url "https://www.botzone.org.cn/api/<uid>/<secret>/localai" `
  --bot-cwd <bot_workdir> `
  --bot-cmd <your bot launch command tokens>
```

## 5.1 Bot Process Protocol Requirements (Important)

When using this adapter with a CLI bot process, your bot must follow these IO details:

1. Startup handshake:
   - Adapter writes one line `1` to bot stdin right after process launch.
   - Bot should stay alive after receiving it.

2. Per-request input:
   - Adapter writes one command line (request text) to bot stdin.

3. Per-request output:
   - Bot may print debug lines.
   - Bot must print at least one action line.
   - Bot must print a sentinel line exactly:
     - `>>>BOTZONE_REQUEST_KEEP_RUNNING<<<`

4. Action selection rule in adapter:
   - Adapter scans output lines before the sentinel.
   - First line matching an action regex is used.
   - If none match, first non-empty line is used.
   - If no output lines exist, fallback is `PASS`.

5. Recommended behavior for bots:
   - Emit exactly one clear action line, then sentinel.
   - Flush stdout promptly.

## 6. Action Validity and Safety

The adapter does not know game rules deeply; it only relays strings.
So correctness depends on your bot output.

Best practices:

1. Always emit exactly one valid action line per request.
2. Ensure format is stable and deterministic.
3. Keep strict mapping from request -> action.
4. Add your own validation before returning action.

## 7. Observability and Debugging

For production use, keep these logs in adapter:

1. request received (`matchid`, request preview)
2. action returned (exact action string)
3. transport errors (`/localai` retry)
4. match finished info line from server

When a match ends unexpectedly:

1. check whether action was valid in that state
2. check whether your bot emitted empty output
3. check whether bridge converted request/action formats correctly

## 8. Typical Failure Patterns

1. Bot process starts but returns no action line:
   - Usually protocol mismatch between adapter and bot.
2. Adapter keeps sending `PASS` and match ends quickly:
   - Usually bot failed to produce valid action.
3. Import/config errors at startup:
   - Usually wrong working directory or missing dependencies.
4. LLM-backed bot fails only on decision turns:
   - Usually model endpoint or response parsing mismatch.

## 9. Integration Checklist

Before release, verify:

1. Adapter can run for multiple matches continuously.
2. One match has isolated bot state.
3. Finished matches are cleaned up correctly.
4. Bot outputs valid action format for all request types.
5. Startup command works on a clean machine.
