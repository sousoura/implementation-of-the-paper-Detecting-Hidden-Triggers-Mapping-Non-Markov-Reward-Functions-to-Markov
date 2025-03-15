# build_rm.py

import json


def build_mealy_machine_json(reward_machine, output_file):
    """
    将 ILP 求解器输出的 reward_machine 转换为 Mealy Machine 格式，
    最终写入 output_file（如 "mealy_rm.json"）。

    输入 reward_machine 的格式：
      {
         "states": [0, 1, ..., K-1],
         "initial_state": 0,
         "transitions": { (i, s, a, s_next): { "next_state": j, "reward": r, "steps": [...] } }
      }

    输出格式：
      {
         "initial_state": "q0",
         "states": {
           "q0": {
              "fingerprint": { "s,a": "r", ... },
              "transitions": { "s,a": "qj", ... }
           },
           "q1": { ... },
           ...
         },
         "state_count": K
      }
    """
    # 根据 reward_machine 中 states 的个数确定 K
    if "states" in reward_machine:
        K = len(reward_machine["states"])
    else:
        # 根据 transitions 中出现的状态推断 K
        states_set = set()
        for key in reward_machine["transitions"]:
            i, s, a, s_next = key
            states_set.add(i)
            states_set.add(reward_machine["transitions"][key]["next_state"])
        K = max(states_set) + 1

    mealy_rm = {
        "initial_state": "q0",
        "states": {},
        "state_count": K
    }

    # 初始化每个状态的 fingerprint 和 transitions 字典
    for i in range(K):
        state_name = f"q{i}"
        mealy_rm["states"][state_name] = {"fingerprint": {}, "transitions": {}}

    # 遍历 reward_machine["transitions"] 中的每一条转移规则
    # 键的格式为 (i, s, a, s_next)，值为 { "next_state": j, "reward": r, "steps": [...] }
    for key, info in reward_machine["transitions"].items():
        i, s, a, s_next = key
        next_state = info["next_state"]
        reward_val = info["reward"]
        # 以 (s, a) 作为 fingerprint 与 transitions 的 key
        fingerprint_key = f"{s},{a}"
        mealy_rm["states"][f"q{i}"]["fingerprint"][fingerprint_key] = str(reward_val)
        mealy_rm["states"][f"q{i}"]["transitions"][fingerprint_key] = f"q{next_state}"

    with open(output_file, 'w') as f:
        json.dump(mealy_rm, f, indent=2)
    print(f"Mealy RM saved to {output_file}")


if __name__ == "__main__":
    # 测试示例
    sample_rm = {
        "states": [0, 1],
        "initial_state": 0,
        "transitions": {
            (0, "755", 3, "756"): {
                "next_state": 1,
                "reward": 1,
                "steps": [("755", 3, 1, "756")]
            },
            (1, "756", 3, "757"): {
                "next_state": 1,
                "reward": 0,
                "steps": [("756", 3, 0, "757")]
            }
        }
    }
    build_mealy_machine_json(sample_rm, "mealy_rm_test.json")
