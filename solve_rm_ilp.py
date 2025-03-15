import pulp
import gurobipy as gp
from gurobipy import GRB


def gurobi_solve_reward_machine_ILP(traces, K):
    """
    使用 Gurobi 求解器构造并求解 Reward Machine ILP 问题。

    输入:
      - traces: List of trajectories，每条轨迹是一个 [(s, a, r, s_next), ...] 的列表
      - K: 假设的 RM 状态数（整数），状态编号为 0,1,...,K-1

    输出:
      - solution: 每条轨迹中每一步映射到 RM 边 (u_i -> u_j) 的取值
      - reward_machine: 根据 ILP 解构造出的 Reward Machine 结构，
                        格式为 { "states": [...], "initial_state": 0, "edges": { (i,j): { "transitions": [...], "reward": r } } }
    """
    # 创建 Gurobi 模型
    model = gp.Model("RewardMachineILP")
    # 开启日志输出，可根据需要设置 OutputFlag 参数（1：输出；0：关闭）
    model.setParam("OutputFlag", 1)

    # 定义决策变量 O[(m, n, i, j)]
    # m: 轨迹索引, n: 轨迹步索引, i,j: RM 状态索引（0 到 K-1）
    O = {}
    for m, traj in enumerate(traces):
        for n, (s, a, r, s_next) in enumerate(traj):
            for i in range(K):
                for j in range(K):
                    var_name = f"O_{m}_{n}_{i}_{j}"
                    O[(m, n, i, j)] = model.addVar(vtype=GRB.BINARY, name=var_name)
    model.update()

    # ----------------------
    # 1. 唯一性约束：每条轨迹的每一步必须恰好选择一个 (i,j) 映射
    # ----------------------
    for m, traj in enumerate(traces):
        for n in range(len(traj)):
            model.addConstr(gp.quicksum(O[(m, n, i, j)] for i in range(K) for j in range(K)) == 1,
                            name=f"uniqueness_{m}_{n}")

    # ----------------------
    # 2. 连续性约束：同一条轨迹中，第 n 步映射为 (u_i -> u_j) 后，
    #    第 n+1 步的起始 RM 状态必须为 u_j，即：对每个 j, sum_i O[m,n,i,j] == sum_k O[m,n+1,j,k]
    # ----------------------
    for m, traj in enumerate(traces):
        for n in range(len(traj) - 1):
            for j in range(K):
                model.addConstr(gp.quicksum(O[(m, n, i, j)] for i in range(K)) ==
                                gp.quicksum(O[(m, n + 1, j, k)] for k in range(K)),
                                name=f"continuity_{m}_{n}_{j}")

    # ----------------------
    # 3. 确定性和奖励一致性约束
    # 对所有出现过的 (s, a, s_next) 进行分组
    # ----------------------
    sa_sprime_steps = {}
    for m, traj in enumerate(traces):
        for n, (s, a, r, s_next) in enumerate(traj):
            key = (s, a, s_next)
            if key not in sa_sprime_steps:
                sa_sprime_steps[key] = []
            sa_sprime_steps[key].append((m, n, r))

    # (a) 确定性约束：
    # 同一 (s,a,s_next) 的两次出现，如果它们都被分配到同一 RM 起始状态 i，
    # 那么它们选择的目标状态必须不同，具体通过：
    # 对于任意不同的 j 和 j'，有 O[m1,n1,i,j] + O[m2,n2,i,j'] <= 1
    for key, steps in sa_sprime_steps.items():
        for idx1 in range(len(steps)):
            for idx2 in range(idx1 + 1, len(steps)):
                m1, n1, r1 = steps[idx1]
                m2, n2, r2 = steps[idx2]
                for i in range(K):
                    for j in range(K):
                        for jp in range(K):
                            if j == jp:
                                continue
                            model.addConstr(O[(m1, n1, i, j)] + O[(m2, n2, i, jp)] <= 1,
                                            name=f"determinism_{key}_{i}_{m1}_{n1}_{m2}_{n2}_{j}_{jp}")

    # (b) 奖励一致性约束：
    # 对于同一 (s,a,s_next) 的两次出现，如果奖励不同，则不能分配到同一 RM 起始状态 i
    for key, steps in sa_sprime_steps.items():
        for idx1 in range(len(steps)):
            for idx2 in range(idx1 + 1, len(steps)):
                m1, n1, r1 = steps[idx1]
                m2, n2, r2 = steps[idx2]
                if r1 != r2:
                    for i in range(K):
                        model.addConstr(gp.quicksum(O[(m1, n1, i, j)] for j in range(K)) +
                                        gp.quicksum(O[(m2, n2, i, j)] for j in range(K)) <= 1,
                                        name=f"reward_consistency_{key}_{i}_{m1}_{n1}_{m2}_{n2}")

    # ----------------------
    # 4. 目标函数：最小化非自环转移数，即所有 i != j 的 O 变量之和
    # ----------------------
    obj = gp.quicksum(O[(m, n, i, j)]
                      for m, traj in enumerate(traces)
                      for n in range(len(traj))
                      for i in range(K)
                      for j in range(K) if i != j)
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()

    # ----------------------
    # 求解 ILP 问题
    # ----------------------
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print("Gurobi 未找到最优解。")
        return None, None

    # ----------------------
    # 从求解结果中提取每条轨迹每一步的 RM 映射
    # ----------------------
    solution = {}
    for m, traj in enumerate(traces):
        solution[m] = []
        for n in range(len(traj)):
            assigned = None
            for i in range(K):
                for j in range(K):
                    if O[(m, n, i, j)].X > 0.5:
                        assigned = (i, j)
                        break
                if assigned is not None:
                    break
            solution[m].append({"assignment": assigned, "trace_step": traj[n]})

    # ----------------------
    # 根据映射构造 Reward Machine 的边信息
    # ----------------------
    rm_transitions = {}
    for m in solution:
        for item in solution[m]:
            i, j = item["assignment"]
            s, a, r, s_next = item["trace_step"]
            if (i, j) not in rm_transitions:
                rm_transitions[(i, j)] = []
            rm_transitions[(i, j)].append((s, a, r, s_next))

    reward_machine = {
        "states": list(range(K)),  # RM 状态为 0, 1, ..., K-1
        "initial_state": 0,  # 初始状态设为 0
        "edges": {}
    }
    for (i, j), transitions in rm_transitions.items():
        rewards = set([t[2] for t in transitions])
        if len(rewards) > 1:
            print(f"警告：RM 边 ({i}->{j}) 上出现不一致奖励：{rewards}")
        reward = rewards.pop() if rewards else None
        reward_machine["edges"][(i, j)] = {
            "transitions": transitions,
            "reward": reward
        }

    return solution, reward_machine


def solve_reward_machine_ILP(traces, K):
    """
    输入:
      - traces: List of trajectories. 每条轨迹是一个 [(s, a, r, s_next), ...] 的列表.
      - K: 假设的 Reward Machine 状态个数，即 U = {u0, u1, ..., u{K-1}}.

    输出:
      - solution: 每条轨迹每步对应的 RM 映射（即 (u_i -> u_j) 的分配情况）。
      - reward_machine: 从 ILP 解中提取的 Reward Machine 结构，包含状态集合、初始状态和边信息（边上包括产生的奖励）。
    """

    # 建立 ILP 模型，目标是最小化非自环转移数
    prob = pulp.LpProblem("RewardMachineILP", pulp.LpMinimize)

    # 定义决策变量 O[(m, n, i, j)]
    # 其中 m: 轨迹索引, n: 轨迹步索引, i,j: RM 状态索引
    O = {}
    for m, traj in enumerate(traces):
        for n, (s, a, r, s_next) in enumerate(traj):
            for i in range(K):
                for j in range(K):
                    var_name = f"O_{m}_{n}_{i}_{j}"
                    O[(m, n, i, j)] = pulp.LpVariable(var_name, cat='Binary')

    # ----------------------
    # 1. 唯一性约束：
    # 每条轨迹的每一步必须恰好选择一个 (i,j) 映射
    # ----------------------
    for m, traj in enumerate(traces):
        for n in range(len(traj)):
            prob += pulp.lpSum(O[(m, n, i, j)] for i in range(K) for j in range(K)) == 1, f"uniqueness_{m}_{n}"

    # ----------------------
    # 2. 连续性约束：
    # 同一条轨迹中，假设第 n 步映射为 (u_i -> u_j)，那么第 n+1 步的起始 RM 状态必须为 u_j。
    # 可用：对每个 j，有 sum_i O[m,n,i,j] = sum_k O[m,n+1,j,k]
    # ----------------------
    for m, traj in enumerate(traces):
        for n in range(len(traj) - 1):
            for j in range(K):
                prob += (pulp.lpSum(O[(m, n, i, j)] for i in range(K)) ==
                         pulp.lpSum(O[(m, n + 1, j, k)] for k in range(K))), f"continuity_{m}_{n}_{j}"

    # ----------------------
    # 3. 确定性和奖励一致性约束
    # 针对在轨迹中出现的每个 (s, a, s_next)，我们需要保证：
    # (a) 同一 (s, a, s′) 出现的不同步中，若分配到同一起始 RM 状态，则映射的目标状态必须一致。
    # (b) 若 (s, a, s′) 出现时奖励不同，则它们不能被分配到同一 RM 起始状态.
    # 为此，我们首先对所有出现的 (s, a, s_next) 分组。
    # ----------------------
    sa_sprime_steps = {}
    for m, traj in enumerate(traces):
        for n, (s, a, r, s_next) in enumerate(traj):
            key = (s, a, s_next)
            if key not in sa_sprime_steps:
                sa_sprime_steps[key] = []
            sa_sprime_steps[key].append((m, n, r))

    # (a) 确定性约束：对于同一 (s,a,s_next) 内任意两次出现，
    # 如果它们都分配到 RM 状态 i 的起始位置，则不能选择不同的目标状态 j.
    for key, steps in sa_sprime_steps.items():
        for idx1 in range(len(steps)):
            for idx2 in range(idx1 + 1, len(steps)):
                m1, n1, r1 = steps[idx1]
                m2, n2, r2 = steps[idx2]
                for i in range(K):
                    # 对于不同的 j 和 j'（j != j'）不能同时取 1
                    for j in range(K):
                        for jp in range(K):
                            if j == jp:
                                continue
                            prob += O[(m1, n1, i, j)] + O[(m2, n2, i, jp)] <= 1, \
                                    f"determinism_{key}_{i}_{m1}_{n1}_{m2}_{n2}_{j}_{jp}"

    # (b) 奖励一致性约束：对于同一 (s,a,s_next)，若两次出现的奖励不同，
    # 则它们不能同时被分配到同一 RM 起始状态 i.
    for key, steps in sa_sprime_steps.items():
        for idx1 in range(len(steps)):
            for idx2 in range(idx1 + 1, len(steps)):
                m1, n1, r1 = steps[idx1]
                m2, n2, r2 = steps[idx2]
                if r1 != r2:
                    for i in range(K):
                        prob += (pulp.lpSum(O[(m1, n1, i, j)] for j in range(K)) +
                                 pulp.lpSum(O[(m2, n2, i, j)] for j in range(K))) <= 1, \
                                f"reward_consistency_{key}_{i}_{m1}_{n1}_{m2}_{n2}"

    # ----------------------
    # 4. 目标函数：最小化非自环转移数，即 i != j 的情况
    # ----------------------
    objective = pulp.lpSum(O[(m, n, i, j)]
                           for m, traj in enumerate(traces)
                           for n in range(len(traj))
                           for i in range(K)
                           for j in range(K) if i != j)
    prob += objective, "MinimizeNonSelfTransitions"

    # 求解 ILP
    solver = pulp.PULP_CBC_CMD(msg=True)
    print("开始求解...")
    result = prob.solve(solver)
    print("Solver Status:", pulp.LpStatus[result])

    if pulp.LpStatus[result] != "Optimal":
        print("未找到最优解！")
        return None, None

    # ----------------------
    # 5. 从 ILP 解中提取结果：
    # 对于每条轨迹每一步，记录选择了哪条 RM 转移 (u_i -> u_j)
    # ----------------------
    solution = {}  # solution[m] 为第 m 条轨迹的步映射列表
    for m, traj in enumerate(traces):
        solution[m] = []
        for n in range(len(traj)):
            assigned = None
            for i in range(K):
                for j in range(K):
                    if pulp.value(O[(m, n, i, j)]) > 0.5:
                        assigned = (i, j)
                        break
                if assigned is not None:
                    break
            solution[m].append({"assignment": assigned, "trace_step": traj[n]})

    # 根据每步映射，汇总 RM 中各边上对应的 (s,a,r,s_next)
    rm_transitions = {}  # key: (i,j) -> list of (s, a, r, s_next)
    for m in solution:
        for item in solution[m]:
            i, j = item["assignment"]
            s, a, r, s_next = item["trace_step"]
            if (i, j) not in rm_transitions:
                rm_transitions[(i, j)] = []
            rm_transitions[(i, j)].append((s, a, r, s_next))

    # 组装 Reward Machine 结构：状态集合、初始状态及边信息
    reward_machine = {
        "states": list(range(K)),  # RM 状态 u0, u1, …, u{K-1}
        "initial_state": 0,  # 设定 u0 为初始状态
        "edges": {}  # edges: (i,j) -> { "transitions": [...], "reward": r }
    }
    for (i, j), transitions in rm_transitions.items():
        # 这里简单起见，如果同一边上有多个奖励且不一致，则报警；实际可根据需求进一步处理
        rewards = set([t[2] for t in transitions])
        if len(rewards) > 1:
            print(f"警告：RM 边 ({i}->{j}) 上出现不一致奖励：{rewards}")
        reward = rewards.pop() if rewards else None
        reward_machine["edges"][(i, j)] = {
            "transitions": transitions,
            "reward": reward
        }

    return solution, reward_machine


if __name__ == "__main__":
    # 构造示例轨迹数据
    # 每个轨迹由多个 (s, a, r, s_next) 四元组构成
    traces = [
        # 轨迹 0
        [
            ("s1", "a1", 1, "s2"),
            ("s2", "a2", 0, "s3"),
            ("s3", "a3", 1, "s4")
        ],
        # 轨迹 1
        [
            ("s1", "a1", 1, "s2"),
            ("s2", "a2", 1, "s3"),  # 此处奖励与轨迹 0 不同，用于测试奖励一致性约束
            ("s3", "a3", 1, "s4")
        ]
    ]

    K = 2  # 初始假设 RM 状态数
    sol, rm = solve_reward_machine_ILP(traces, K)

    if sol is not None:
        print("\n每条轨迹各步对应的 RM 映射：")
        for m, mapping in sol.items():
            print(f"轨迹 {m}:")
            for step in mapping:
                print(step)

        print("\n提取的 Reward Machine 结构：")
        print("状态集合：", rm["states"])
        print("初始状态：", rm["initial_state"])
        print("边信息：")
        for (i, j), edge in rm["edges"].items():
            print(f"  边 u{i} -> u{j}: 奖励 = {edge['reward']}, 对应原始转移 = {edge['transitions']}")
