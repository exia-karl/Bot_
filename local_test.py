#!/usr/bin/env python3
"""
本地离线测试工具 —— 不需要 Botzone 账号，不需要 LLM API。

模拟 Botzone 协议，将请求序列喂给 bot，检测每个响应是否合法。

用法：
  # 测试解析逻辑（不需要API，秒级完成）
  python local_test.py --mode parse

  # 测试完整协议流程（不需要API，秒级完成）
  python local_test.py --mode protocol

  # 测试真实 LLM 调用（需要配置 api_config/llm_config.json）
  python local_test.py --mode llm
"""

import sys
from pathlib import Path

# Ensure we can import from project root
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from local_bots.mahjong.policy_llm import extract_move, _normalize_action, _canonicalize_action, safe_fallback_action


# ── Test data from the user's online game ──
REQUESTS = [
    "0 3 1",
    "1 0 0 0 0 W7 W4 W8 W2 B2 B4 B9 W1 W9 T8 W5 B2 J1",
    "3 0 DRAW",
    "3 0 PLAY F2",
    "3 1 DRAW",
    "3 1 PLAY F2",
    "3 2 DRAW",
    "3 2 PLAY B1",
    "2 B8",
    "3 3 PLAY W5",
    "3 0 DRAW",
    "3 0 PLAY F1",
    "3 1 DRAW",
    "3 1 PLAY F4",
    "3 0 PENG T1",
    "3 1 CHI T2 J1",
    "3 2 DRAW",
    "3 2 PLAY F4",
    "2 W5",
    "3 3 PLAY J1",
    "3 0 DRAW",
    "3 0 PLAY F3",
    "3 2 PENG B5",
    "2 W9",
]

# Expected request types:
# "0 X Y" = init, "1 ..." = deal, "2 X" = our draw, "3 ..." = other player action


def test_parsing():
    """测试解析逻辑 —— 不需要 API，秒级完成"""
    print("=" * 60)
    print("📋 测试1: 解析逻辑单元测试")
    print("=" * 60)

    tests = [
        # (llm_output, valid_actions, expected_result)
        # 正常情况
        ("答案：Play W3", ["Play W1", "Play W3", "Play B5"], "Play W3"),
        ("答案：Hu", ["Hu"], "Hu"),
        ("答案：Pass", ["Peng W1", "Pass"], "Pass"),
        # 中文动作名
        ("答案：打出 W3", ["Play W1", "Play W3", "Play B5"], "Play W3"),
        ("答案：打 W3", ["Play W1", "Play W3", "Play B5"], "Play W3"),
        ("答案：出 W3", ["Play W1", "Play W3", "Play B5"], "Play W3"),
        ("答案：胡", ["Hu", "Pass"], "Hu"),
        ("答案：过", ["Peng W1", "Pass"], "Pass"),
        ("答案：碰 W1", ["Peng W1", "Pass"], "Peng"),
        ("答案：吃 W2", ["Chi W2", "Pass"], "Chi W2"),
        ("答案：杠 W1", ["Gang W1", "Pass"], "Gang W1"),
        ("答案：暗杠 W1", ["AnGang W1", "Pass"], "Gang W1"),
        ("答案：补杠 W1", ["BuGang W1", "Pass"], "BuGang W1"),
        # fuzzy match: "Discard W9" normalizes to "Play W9", not in list, no match
        ("分析：前期\n答案：Discard W9", ["Play W1", "Play W3", "Play B5"], None),
        # think tag
        ("<think>reasoning</think>\n分析：前期\n答案：Play W3", ["Play W1", "Play W3"], "Play W3"),
        # 无答案标记时扫描全行
        ("分析：前期\nPlay W5 is good", ["Play W1", "Play W5", "Pass"], "Play W5"),
        # 非法输出应返回失败
        ("答案：xyz", ["Play W1", "Play W3"], None),
    ]

    passed = 0
    failed = 0
    for llm_out, valid, expected in tests:
        status, result = extract_move(llm_out, valid)
        if expected is None:
            if status == 0:
                passed += 1
                print(f"  ✅ 正确拒绝非法输入: '{llm_out[:40]}...'")
            else:
                failed += 1
                print(f"  ❌ 应拒绝但返回了: {result} | 输入: '{llm_out[:40]}...'")
        elif result == expected:
            passed += 1
            print(f"  ✅ '{llm_out[:40]}...' -> {result}")
        else:
            failed += 1
            print(f"  ❌ '{llm_out[:40]}...' -> {result} (expected {expected})")

    print(f"\n  结果: {passed} 通过, {failed} 失败 / {len(tests)} 总计")
    return failed == 0


def test_fallback():
    """测试安全回退逻辑"""
    print("\n" + "=" * 60)
    print("📋 测试2: 安全回退逻辑 (_safe_fallback_action)")
    print("=" * 60)

    tests = [
        # (valid_actions, description, must_not_be_pass)
        (["Play W1", "Play W3", "Play B5", "Play W9"], "摸牌回合", True),
        (["Hu"], "胡牌回合", False),
        (["Peng W1", "Pass"], "对手回合（碰/过）", False),
        (["Play W9", "Play W1", "Play W5", "Play T8"], "摸牌回合-应选中张", True),
        ([], "空列表", False),
    ]

    passed = 0
    for valid, desc, must_not_be_pass in tests:
        obs = {"action_mask_llm": valid}
        result = safe_fallback_action(obs["action_mask_llm"])
        if must_not_be_pass and result == "Pass":
            print(f"  ❌ {desc}: 不应返回 Pass! valid={valid}")
            continue
        if not must_not_be_pass and result == "Pass":
            print(f"  ✅ {desc}: 正确返回 Pass")
            passed += 1
        elif result in valid or (not valid and result == "Pass"):
            middle_check = ""
            if "Play W5" in valid and result == "Play W5":
                middle_check = " [中张偏好✅]"
            elif "Play" in str(valid) and result.startswith("Play"):
                middle_check = f" [选中{result}]"
            print(f"  ✅ {desc}: {result}{middle_check}")
            passed += 1
        else:
            print(f"  ❌ {desc}: 非法动作 {result} not in {valid}")

    print(f"\n  结果: {passed}/{len(tests)} 通过")
    return passed == len(tests)


def test_protocol():
    """测试完整 Botzone 协议流程 —— 不需要 API"""
    print("\n" + "=" * 60)
    print("📋 测试3: 完整协议流程模拟")
    print("=" * 60)

    # Ensure correct import paths
    import sys
    _project = Path(__file__).resolve().parent
    if str(_project) not in sys.path:
        sys.path.insert(0, str(_project))
    if str(_project / "local_bots" / "mahjong") not in sys.path:
        sys.path.insert(0, str(_project / "local_bots" / "mahjong"))

    # Import agent class directly
    from llm_bot_cn import FeatureAgent2Adapted

    agent = None
    seatWind = None

    for i, request in enumerate(REQUESTS):
        parts = request.split()
        req_type = parts[0]
        label = f"[{i}] {request}"

        try:
            if req_type == "0":
                seatWind = int(parts[1])
                agent = FeatureAgent2Adapted(seatWind)
                agent.request2obs(f"Wind {parts[2]}")
                print(f"  {label} -> 初始化 seatWind={seatWind}")

            elif req_type == "1":
                agent.request2obs(" ".join(["Deal", *parts[5:]]))
                print(f"  {label} -> 发牌: hand={parts[5:]}")

            elif req_type == "2":
                # Our draw turn
                obs = agent.request2obs(f"Draw {parts[1]}")
                valid = obs["action_mask_llm"]

                # Simulate what would happen if LLM is unavailable (fallback)
                fallback = safe_fallback_action(obs["action_mask_llm"])
                is_pass_illegal = "Pass" not in valid

                if is_pass_illegal:
                    assert fallback != "Pass", f"BUG: 摸牌回合的fallback返回了Pass!"
                    print(f"  {label} -> 摸 {parts[1]}, fallback={fallback} ✅ (非Pass)")
                else:
                    print(f"  {label} -> 摸 {parts[1]}, fallback={fallback}")

                # Verify every valid action is well-formed
                for action in valid:
                    assert action.startswith("Play ") or action in ("Hu",), \
                        f"非法action格式: {action}"

            elif req_type == "3":
                p = int(parts[1])
                if parts[2] == "DRAW":
                    agent.request2obs(f"Player {p} Draw")
                elif parts[2] == "PLAY":
                    agent.request2obs(f"Player {p} Play {parts[3]}")
                elif parts[2] == "CHI":
                    if len(parts) >= 5:
                        agent.request2obs(f"Player {p} Chi {parts[3]}")
                        # Chi followed by play
                        obs = agent.request2obs(f"Player {p} Play {parts[4]}")
                        if obs is not None:
                            valid = obs["action_mask_llm"]
                            print(f"  {label} -> 对手吃+出牌, 合法动作={valid}")
                    else:
                        agent.request2obs(f"Player {p} Chi {parts[3]}")
                elif parts[2] == "PENG":
                    agent.request2obs(f"Player {p} Peng")
                    # Peng followed by play
                    if len(parts) >= 4:
                        agent.request2obs(f"Player {p} Play {parts[3]}")
                elif parts[2] == "GANG":
                    agent.request2obs(f"Player {p} Gang")
                elif parts[2] == "BUGANG":
                    obs = agent.request2obs(f"Player {p} BuGang {parts[3]}")
                    if obs is not None:
                        valid = obs["action_mask_llm"]
                        print(f"  {label} -> 对手补杠, 合法动作={valid}")
                print(f"  {label} -> OK")

        except Exception as e:
            print(f"  ❌ {label} -> 异常: {e}")
            import traceback
            traceback.print_exc()
            return False

    print(f"\n  ✅ 全部 {len(REQUESTS)} 个请求处理完成，无异常")
    return True


def test_llm_single_turn():
    """测试单次 LLM 调用 —— 需要 API 配置"""
    print("\n" + "=" * 60)
    print("📋 测试4: 单次 LLM 决策测试")
    print("=" * 60)

    from api_config.conf import load_llm_config
    model_name, api_base, api_key = load_llm_config()

    if not all([model_name, api_base, api_key]):
        print("  ⚠️  未配置 API (api_config/llm_config.json)，跳过 LLM 测试")
        print("  如需测试，请编辑 api_config/llm_config.json 填入配置")
        return True

    from llm_bot_cn import FeatureAgent2Adapted
    from policy_llm import safe_fallback_action

    # Simulate a simple draw-and-play scenario
    agent = FeatureAgent2Adapted(0)
    agent.request2obs("Wind 0")
    agent.request2obs("Deal W1 W2 W3 B5 B6 B7 T1 T2 T3 F1 F2 J1 J2")
    agent.request2obs("Player 1 Play F3")
    agent.request2obs("Player 2 Play F4")
    agent.request2obs("Player 3 Play J3")
    obs = agent.request2obs("Draw W5")

    valid = obs["action_mask_llm"]
    print(f"  手牌: W1,W2,W3,B5,B6,B7,T1,T2,T3,F1,F2,J1,J2 + 摸 W5")
    print(f"  合法动作: {valid}")

    from local_bots.mahjong.llm_bot_cn import obs2response
    try:
        response = obs2response(None, obs)
        print(f"  LLM 响应: {response}")
        if response in valid:
            print(f"  ✅ 响应是合法动作")
        else:
            print(f"  ❌ 响应不在合法动作列表中!")
    except Exception as e:
        print(f"  ❌ LLM 调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="本地离线测试工具")
    parser.add_argument("--mode", choices=["parse", "protocol", "llm", "all"],
                        default="all", help="测试模式")
    args = parser.parse_args()

    all_ok = True

    if args.mode in ("parse", "all"):
        all_ok &= test_parsing()
        all_ok &= test_fallback()

    if args.mode in ("protocol", "all"):
        all_ok &= test_protocol()

    if args.mode in ("llm", "all"):
        all_ok &= test_llm_single_turn()

    print("\n" + "=" * 60)
    if all_ok:
        print("🎉 所有测试通过!")
    else:
        print("⚠️  部分测试失败，请检查上面的 ❌ 标记")
    print("=" * 60)
