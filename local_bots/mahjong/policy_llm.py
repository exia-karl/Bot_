import re
from typing import Callable, Dict, List, Optional, Tuple


def _normalize_action(text: str) -> str:
    """Normalize common model output variations into standard action format."""
    # Collapse multiple whitespace to single space
    text = re.sub(r"\s+", " ", text)
    # Strip punctuation, parentheses, brackets
    text = text.strip().rstrip(".;,?!:;\u3002\uff1b\uff0c\uff1f\uff01\uff1a\uff08\uff09()[]\u3010\u3011\"'").strip()

    # Handle "Discard X" -> "Play X"
    m = re.match(r"Discard\s+(\w\d)", text, re.IGNORECASE)
    if m:
        return "Play " + m.group(1)

    # Handle Chinese "\u6253\u51fa X" / "\u51fa X" / "\u6253 X" -> "Play X"
    m = re.match(r"\u6253\u51fa\s*(\w\d)", text)
    if m:
        return "Play " + m.group(1)
    m = re.match(r"\u51fa\s*(\w\d)", text)
    if m:
        return "Play " + m.group(1)
    m = re.match(r"\u6253\s*(\w\d)", text)
    if m:
        return "Play " + m.group(1)

    # Handle "\u80e1" / "\u548c" -> "Hu"
    if text in ("\u80e1", "\u548c", "\u548c\u724c", "\u80e1\u724c"):
        return "Hu"

    # Handle "\u8fc7" -> "Pass"
    if text in ("\u8fc7", "\u8df3\u8fc7", "\u653e\u8fc7", "\u4e0d\u8981"):
        return "Pass"

    # Handle "\u78b0 X" -> "Peng X"
    m = re.match(r"\u78b0\s*(\w\d)", text)
    if m:
        return "Peng " + m.group(1)

    # Handle "\u5403 X" -> "Chi X"
    m = re.match(r"\u5403\s*(\w\d)", text)
    if m:
        return "Chi " + m.group(1)

    # Handle "\u660e\u6760 X" -> "Gang X", "\u6697\u6760 X" -> "AnGang X", "\u6760 X" -> "Gang X"
    m = re.match(r"\u660e\u6760\s*(\w\d)", text)
    if m:
        return "Gang " + m.group(1)
    m = re.match(r"\u6697\u6760\s*(\w\d)", text)
    if m:
        return "AnGang " + m.group(1)
    m = re.match(r"\u8865\u6760\s*(\w\d)", text)
    if m:
        return "BuGang " + m.group(1)
    m = re.match(r"\u6760\s*(\w\d)", text)
    if m:
        return "Gang " + m.group(1)

    # Handle bare "Peng" / "\u78b0" (no tile, but valid when only one option)
    if text == "Peng" or text == "\u78b0":
        return "Peng"

    # Collapse whitespace again after normalization
    text = re.sub(r"\s+", " ", text)
    return text


def _tile_from_action(action: str) -> str:
    """Extract tile from action string like 'Play W3' -> 'W3'."""
    parts = action.split()
    return parts[-1] if len(parts) > 1 else ""


def _scan_all_lines_for_action(text: str, valid_action_list: List[str]) -> Tuple[int, str]:
    """Scan every line of text for a valid action substring. More robust fallback."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Normalize the line
        normalized = _normalize_action(line)
        if normalized in valid_action_list:
            return 1, _canonicalize_action(normalized)
        # Check if any valid action is a substring of this line
        for action in valid_action_list:
            if action in normalized:
                return 1, _canonicalize_action(action)
    return 0, ""


def extract_move(text: str, valid_action_list: List[str]) -> Tuple[int, str]:
    answer_keywords = [
        "\u7b54\u6848\u662f:", "\u7b54\u6848\u662f\uff1a", "\u7b54\u6848:", "\u7b54\u6848\uff1a", "\u7b54\u6848:",
        "\u52a8\u4f5c:", "\u52a8\u4f5c\uff1a", "\u9009\u62e9:", "\u9009\u62e9\uff1a", "\u8f93\u51fa:",
        "Answer:", "answer:", "Answer\uff1a", "answer\uff1a",
        "\u7b54\u6848?", "Answer?",
    ]

    answer_text = text.strip()
    for keyword in answer_keywords:
        idx = answer_text.find(keyword)
        if idx != -1:
            answer_text = answer_text[idx + len(keyword):].strip()
            break

    # Take first non-empty line after keyword
    lines = [ln.strip() for ln in answer_text.split("\n") if ln.strip()]
    if lines:
        first_line = lines[0]
    else:
        first_line = answer_text

    # Collapse whitespace and strip punctuation/brackets
    clean_action = re.sub(r"\s+", " ", first_line)
    clean_action = clean_action.strip().rstrip(".;,?!:;\u3002\uff1b\uff0c\uff1f\uff01\uff1a\uff08\uff09()[]\u3010\u3011\"'").strip()

    # Try normalization first
    normalized = _normalize_action(clean_action)
    if normalized in valid_action_list:
        return 1, _canonicalize_action(normalized)

    # Direct match
    if clean_action in valid_action_list:
        return 1, _canonicalize_action(clean_action)

    # Use normalized version for fuzzy matching if different from original
    match_text = normalized if normalized != clean_action else clean_action

    # Fuzzy match: exact substring containment (try both directions)
    for action in valid_action_list:
        if not match_text:
            continue
        # match_text is substring of valid action
        if match_text in action:
            return 1, _canonicalize_action(action)
        # valid action is substring of match_text
        if action in match_text:
            # Prefer longest/most specific match
            if "BuGang" in match_text and "BuGang" in action:
                tile = _tile_from_action(match_text)
                return 1, f"BuGang {tile}" if tile else action
            if "AnGang" in match_text:
                tile = _tile_from_action(match_text)
                return 1, f"Gang {tile}" if tile else "Gang"
            if "Peng" in match_text and "Peng" in action:
                return 1, "Peng"
            if "Gang" in match_text and "Gang" in action:
                tile = _tile_from_action(match_text)
                if tile:
                    return 1, f"Gang {tile}"
                valid_tile = _tile_from_action(action)
                return 1, f"Gang {valid_tile}" if valid_tile else "Gang"
            if "Chi" in match_text and "Chi" in action:
                return 1, action
            if "Hu" in match_text and action == "Hu":
                return 1, "Hu"
            if "Pass" in match_text and action == "Pass":
                return 1, "Pass"
            return 1, action

    # Tile-only match for Play/Chi actions
    for action in valid_action_list:
        if action.startswith("Play ") or action.startswith("Chi "):
            tile = action.split()[-1]
            if tile == clean_action or clean_action.endswith(tile):
                return 1, action

    # Last resort: scan ALL lines of original text for any valid action
    return _scan_all_lines_for_action(text, valid_action_list)


def normalize_text(text: str) -> str:
    return re.sub(r"\r?\n|\r", " ", text)


def extract_answer(response_text: str) -> str:
    if not response_text:
        return ""

    text = response_text.strip()

    # Strip \uff1cthink\uff1e...\uff1c/think\uff1e tags if present (DeepSeek thinking mode)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    markers = [
        "\u7b54\u6848:", "\u7b54\u6848\uff1a",
        "Answer:", "answer:", "Answer\uff1a", "answer\uff1a",
        "\u7b54\u6848?", "Answer?",
    ]
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker) :].strip()

    # Fallback: use last non-empty line as answer candidate.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines[-1]

    return text


def infer_action_with_retry(
    *,
    obs: Dict,
    rule_prompt: str,
    state_prompt: str,
    valid_move_prompt: str,
    invalid_answer_prompt_builder: Callable[[List[str]], str],
    llm_query_fn: Callable[[str, str], Tuple[str, str]],
    answer_extractor: Optional[Callable[[str], str]] = None,
    max_retry: int = 3,
) -> Tuple[str, Optional[Dict[str, str]]]:
    llm_valid_move_list = obs["action_mask_llm"]
    if len(llm_valid_move_list) == 1:
        return llm_valid_move_list[0], None

    trial_count = 0
    previous_answers: List[str] = []
    valid_signal = 0
    response = "Pass"
    reasoning_text = ""
    responsed_text = ""
    user_prompt = ""

    while trial_count < max_retry:
        trial_count += 1
        user_prompt = (
            state_prompt
            + valid_move_prompt
            + invalid_answer_prompt_builder(previous_answers)
        )

        responsed_text, reasoning_text = llm_query_fn(rule_prompt, user_prompt)

        if answer_extractor is not None:
            answer_text = answer_extractor(responsed_text)
            valid_signal, response = extract_move(
                "Answer:" + answer_text, llm_valid_move_list
            )
            if not valid_signal:
                previous_answers.append(answer_text)
        else:
            valid_signal, response = extract_move(responsed_text, llm_valid_move_list)
            if not valid_signal:
                previous_answers.append(response)

        if valid_signal:
            break

    # FALLBACK: pick a safe action based on context
    if not valid_signal:
        response = safe_fallback_action(llm_valid_move_list)

    debug_info = {
        "system_prompt": rule_prompt,
        "user_prompt": user_prompt,
        "reasoning": reasoning_text,
        "output": responsed_text,
    }
    return response, debug_info


def safe_fallback_action(valid_action_list):
    """Pick a safe action when LLM is unavailable (config missing or API failure).

    Priority: Hu > Play (prefer middle tiles for safety) > Pass
    Never returns an illegal action.
    """
    if not valid_action_list:
        return "Pass"
    if len(valid_action_list) == 1:
        return valid_action_list[0]
    if "Hu" in valid_action_list:
        return "Hu"
    play_actions = [a for a in valid_action_list if a.startswith("Play ")]
    if play_actions:
        def _tile_safety_score(action):
            tile = action.split()[-1]
            num = int(tile[1]) if len(tile) > 1 and tile[1].isdigit() else 5
            return abs(num - 5)
        play_actions.sort(key=_tile_safety_score)
        return play_actions[0]
    if "Pass" in valid_action_list:
        return "Pass"
    return valid_action_list[0]


def _canonicalize_action(clean_action: str) -> str:
    """Convert action to Botzone protocol format.

    Botzone protocol expects:
    - "Hu" / "Pass" / "Peng" (no tile needed)
    - "Play X" / "Chi X" / "Gang X" / "BuGang X" / "AnGang X" (with tile)
    - AnGang during draw turn: engine converts response[0]=="Gang" + tile to GANG X
    """
    parts = clean_action.split()
    if "AnGang" in clean_action:
        # Convert "AnGang X" -> "Gang X" for botzone protocol
        tile = parts[1] if len(parts) > 1 else ""
        return f"Gang {tile}" if tile else "Gang"
    if "BuGang" in clean_action:
        return clean_action
    if "Peng" in clean_action:
        # Peng doesn't need tile in botzone protocol
        return "Peng"
    if "Gang" in clean_action:
        # Keep "Gang X" with tile (draw turn needs it, opponent turn ignores it)
        return clean_action
    return clean_action
