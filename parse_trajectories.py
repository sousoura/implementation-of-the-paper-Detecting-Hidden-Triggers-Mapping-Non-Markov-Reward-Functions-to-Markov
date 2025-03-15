# parse_trajectories.py

import ast


def parse_trajectories(file_path):
    """
    读取 trajectories.json（非标准 JSON 文件，每行一个 trajectory），
    将每条 trajectory 转换为一系列 (s, a, r, s_next) 四元组，
    其中 s 和 s_next 分别为相邻步骤中状态的数字编码，
    a 为动作，r 为奖励（取自下一步）。

    注意：对于一个长度为 N 的 trajectory，
    会生成 N-1 个转移（即每个转移由当前步和下一步构成）。
    """
    trajectories = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 使用 ast.literal_eval 解析每行（不是标准 JSON 格式）
            traj = ast.literal_eval(line)
            if len(traj) < 2:
                # 长度不足，无法形成转移
                continue
            processed_traj = []
            for i in range(len(traj) - 1):
                curr_step = traj[i]
                next_step = traj[i + 1]
                # 提取当前状态的数字编码（忽略 label）
                s = curr_step[0][1]
                a = curr_step[1]
                # 根据论文格式，转移的奖励取下一步中的奖励
                r = curr_step[2]
                s_next = next_step[0][1]
                processed_traj.append((s, a, r, s_next))
            trajectories.append(processed_traj)
    return trajectories


if __name__ == "__main__":
    file_path = "trajectories.json"
    trajs = parse_trajectories(file_path)
    print("Parsed trajectories:")
    for t in trajs:
        print(t)
