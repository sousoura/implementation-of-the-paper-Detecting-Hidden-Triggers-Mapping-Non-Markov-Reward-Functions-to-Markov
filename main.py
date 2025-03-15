# main.py

from parse_trajectories import parse_trajectories
# from solve_rm_ilp import solve_reward_machine_ILP
from gurobi_solve_rm_ilp import gurobi_solve_reward_machine_ILP
from original_solve_rm_ilp import original_solve_reward_machine_ILP
from build_rm import build_mealy_machine_json


def main():
    # 1. 解析 trajectories.json
    traj_file = "trajectories.json"
    print(f"Parsing trajectories from {traj_file} ...")
    trajectories = parse_trajectories(traj_file)
    if not trajectories:
        print("未解析到有效的 trajectories 数据。")
        return

    # 2. 调用 ILP 求解器
    # 这里设定 RM 状态数 K（例如初始设为 2，若不满足再调大）
    K = 3
    print("Solving ILP for Reward Machine with K =", K)
    sol, reward_machine = original_solve_reward_machine_ILP(trajectories, K)
    # sol, reward_machine = gurobi_solve_reward_machine_ILP(trajectories, K)
    # sol, reward_machine = solve_reward_machine_ILP(trajectories, K)
    if reward_machine is None:
        print("ILP 无法求解 Reward Machine，请尝试增加 K 的值。")
        return
    print("ILP 求解成功。")

    # 3. 根据 ILP 求解结果构造 mealy_rm.json
    output_file = "mealy_rm.json"
    build_mealy_machine_json(reward_machine, output_file)


if __name__ == "__main__":
    main()
