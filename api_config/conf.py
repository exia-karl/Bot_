import json
import os
import re
import time

import requests


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_llm_config_path():
    candidates = []
    env_path = os.environ.get("LLM_CONFIG_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.append("/data/llm_config.json")
    candidates.append(os.path.join(os.path.dirname(__file__), "llm_config.json"))

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def load_llm_config():
    """
    Load OpenAI-compatible API config.

    Config search order:
      1) LLM_CONFIG_PATH env var
      2) /data/llm_config.json (Botzone userfile mount path)
      3) api_config/llm_config.json (local development path)
    """
    config_path = _resolve_llm_config_path()
    if not config_path:
        return None, None, None
    data = _read_json(config_path)
    return data.get("llm_name"), data.get("api_base"), data.get("api_key")


def extract_think_content(text):
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else ""


def remove_think_content(text):
    pattern = r"<think>.*?</think>"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
    return cleaned_text.strip()


def query_openai_compatible(
    api_base,
    api_key,
    llm_name,
    system_prompt,
    user_prompt,
    temperature=0.0,
    max_tokens=256,
):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": llm_name,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "text"},
    }

    max_try = 5
    last_err = None
    for _ in range(max_try):
        try:
            resp = requests.post(api_base, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            response_json = resp.json()
            if "choices" in response_json:
                responsed_text = response_json["choices"][0]["message"]["content"]
                reasoning_text = response_json["choices"][0]["message"].get(
                    "reasoning_content", ""
                )
                if not reasoning_text:
                    reasoning_text = extract_think_content(responsed_text)
                    if responsed_text:
                        responsed_text = remove_think_content(responsed_text)
                return responsed_text, reasoning_text
            last_err = ValueError("No 'choices' in response JSON")
        except Exception as e:
            last_err = e
        time.sleep(2)

    raise RuntimeError(f"OpenAI-compatible query failed: {last_err}")
