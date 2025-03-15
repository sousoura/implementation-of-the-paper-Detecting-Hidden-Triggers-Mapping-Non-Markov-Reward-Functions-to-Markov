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
         "edges": {
             (i,j): {
                 "transitions": [ (s, a, r, s_next), ... ],
                 "reward": r
             },
             ...
         }
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
    K = len(reward_machine["states"]) if "states" in reward_machine else None
    if K is None:
        # 根据 edges 中出现的状态推断 K
        states_set = set()
        for (i, j) in reward_machine["edges"]:
            states_set.add(i)
            states_set.add(j)
        K = max(states_set) + 1
        reward_machine["states"] = list(range(K))

    mealy_rm = {
        "initial_state": "q0",
        "states": {},
        "state_count": K
    }

    # 为每个状态建立 fingerprint 和 transitions 字典
    for i in range(K):
        state_name = f"q{i}"
        mealy_rm["states"][state_name] = {"fingerprint": {}, "transitions": {}}

    # 遍历每个边，将对应的 (s,a) 信息加入起始状态的 fingerprint 和 transitions 中
    for (i, j), edge_info in reward_machine["edges"].items():
        transitions = edge_info.get("transitions", [])
        reward_val = edge_info.get("reward", None)
        for (s, a, r, s_next) in transitions:
            key = f"{s},{a}"
            mealy_rm["states"][f"q{i}"]["fingerprint"][key] = str(reward_val)
            mealy_rm["states"][f"q{i}"]["transitions"][key] = f"q{j}"

    with open(output_file, 'w') as f:
        json.dump(mealy_rm, f, indent=2)
    print(f"Mealy RM saved to {output_file}")


if __name__ == "__main__":
    # 测试示例
    sample_rm = {
        "states": [0, 1],
        "initial_state": 0,
        "edges": {
            (0, 1): {
                "transitions": [(755, 3, 1, 756)],
                "reward": 1
            },
            (1, 1): {
                "transitions": [(756, 3, 0, 757)],
                "reward": 0
            }
        }
    }
    build_mealy_machine_json(sample_rm, "mealy_rm_test.json")
