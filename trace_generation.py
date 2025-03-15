import json
import random


# ---------- 环境与任务定义 ----------

# 3x3网格：坐标 (x, y)，x, y ∈ {0,1,2}
# 为方便，与状态编号对应: state_id = x*3 + y
# 例如 (0,0)->0, (0,1)->1, (0,2)->2, (1,0)->3, ..., (2,2)->8
def pos_to_id(x, y):
    return x * 3 + y


def id_to_pos(state_id):
    return divmod(state_id, 3)  # 返回(x, y)


# 标签规则： (0,2)->"a", (2,0)->"b", 其余 "None"
def get_label(x, y):
    if (x, y) == (0, 2):
        return "a"
    elif (x, y) == (2, 0):
        return "b"
    elif (x, y) == (2, 2):
        return "c"
    else:
        return "None"


# 动作编码: 0=up, 1=down, 2=left, 3=right
ACTIONS = [0, 1, 2, 3]


def next_position(x, y, action):
    """根据动作得到下一时刻坐标，超出边界则保持不动。"""
    if action == 0:  # up
        new_x = max(x - 1, 0)
        new_y = y
    elif action == 1:  # down
        new_x = min(x + 1, 2)
        new_y = y
    elif action == 2:  # left
        new_x = x
        new_y = max(y - 1, 0)
    elif action == 3:  # right
        new_x = x
        new_y = min(y + 1, 2)
    else:
        new_x, new_y = x, y
    return new_x, new_y


def run_one_episode(max_steps=20):
    """
    运行一次回合，返回一条轨迹 (list of steps)。
    每个 step 记录为 [ [label, state_id], action, reward ]。
    任务逻辑：
      - 初始位置(0,0)
      - 必须先到(0,2)(label=a)，再到(2,0)(label=b) 才获得reward=1并结束
      - 如果先到达b,则没有奖励也不结束
      - 其余步reward=0,回合不结束
      - 最多 max_steps 步后强制结束
    """
    trajectory = []
    x, y = 0, 0  # 初始位置
    have_visited_a = False  # 是否访问过 a
    have_visited_c = False  # 是否访问过 c
    done = False

    for _ in range(max_steps):
        # 当前label, state_id
        label = get_label(x, y)
        s_id = pos_to_id(x, y)

        # 随机动作
        action = random.choice(ACTIONS)

        # 执行动作
        nx, ny = next_position(x, y, action)

        # 判断奖励
        reward = 0.0
        # 如果已经访问过 a，并且当前这一步新位置就是 b => 给 1 并结束
        if have_visited_a and have_visited_c and (nx, ny) == (2, 0):
            reward = 1.0
            done = True

        # 更新“是否访问过 a”
        if (x, y) == (0, 2):
            have_visited_a = True

        # 更新“是否访问过 a”
        if (x, y) == (2, 2):
            have_visited_c = True

        # 更新 agent 位置
        x, y = nx, ny

        # 更新轨迹记录
        # 这里的 reward 是这一步动作后得到的奖励
        trajectory.append([[label, s_id], action, reward])

        # 如果拿到奖励就结束
        if done:
            final_s_id = pos_to_id(x, y)
            trajectory.append([[label, final_s_id], 0, 0.0])
            break

    return trajectory


def main():
    random.seed(0)  # 固定随机种子，方便复现。可根据需要修改或去掉。
    # 收集 50 条轨迹
    all_trajectories = []
    num = 50
    for _ in range(num):
        traj = run_one_episode(max_steps=1000)
        all_trajectories.append(traj)

    # 将它们写入 trajectories.json
    # 由于题目中说：每一条轨迹写成一行 JSON，所以我们一行一行写
    with open("trajectories.json", "w", encoding="utf-8") as f:
        for traj in all_trajectories:
            line_str = json.dumps(traj, ensure_ascii=False)
            f.write(line_str + "\n")

    print("Done! " + str(num) + " trajectories have been saved to trajectories.json")


if __name__ == "__main__":
    main()
