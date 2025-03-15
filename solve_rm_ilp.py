# # solve_rm_ilp.py
#
# import pulp
#
#
# def solve_reward_machine_ILP(traces, K):
#     """
#     输入:
#       - traces: List of trajectories. 每条轨迹是一个 [(s, a, r, s_next), ...] 的列表.
#       - K: 假设的 Reward Machine 状态个数，即 U = {u0, u1, ..., u{K-1}}.
#
#     输出:
#       - solution: 每条轨迹每步对应的 RM 映射（即 (u_i -> u_j) 的分配情况）。
#       - reward_machine: 从 ILP 解中提取的 Reward Machine 结构，
#           其中对于每个 <u_i, s, a, s_next> 都有唯一的下一个状态 u_j 和奖励 r，
#           表示为： δᵤ(u_i, (s,a,s_next)) = u_j, δᵣ(u_i, (s,a,s_next)) = r.
#     """
#
#     # 建立 ILP 模型，目标函数为最小化非自环转移数（鼓励RM简洁）
#     prob = pulp.LpProblem("RewardMachineILP", pulp.LpMinimize)
#
#     # 定义决策变量 O[(m, n, i, j)]
#     # m: 轨迹索引, n: 轨迹步索引, i,j: RM 状态索引
#     O = {}
#     for m, traj in enumerate(traces):
#         for n, (s, a, r, s_next) in enumerate(traj):
#             for i in range(K):
#                 for j in range(K):
#                     var_name = f"O_{m}_{n}_{i}_{j}"
#                     O[(m, n, i, j)] = pulp.LpVariable(var_name, cat='Binary')
#
#     # ----------------------
#     # 1. 唯一性约束：每一步必须唯一地分配到 (u_i, u_j)
#     # ----------------------
#     for m, traj in enumerate(traces):
#         for n in range(len(traj)):
#             prob += pulp.lpSum(O[(m, n, i, j)] for i in range(K) for j in range(K)) == 1, f"uniqueness_{m}_{n}"
#
#     # ----------------------
#     # 2. 连续性约束：对于同一条轨迹，
#     # 如果第 n 步分配为 (u_i -> u_j)，则第 n+1 步的分配的起始状态必须为 u_j
#     # ----------------------
#     for m, traj in enumerate(traces):
#         for n in range(len(traj) - 1):
#             for j in range(K):
#                 prob += (pulp.lpSum(O[(m, n, i, j)] for i in range(K)) ==
#                          pulp.lpSum(O[(m, n + 1, j, k)] for k in range(K))), f"continuity_{m}_{n}_{j}"
#
#     # ----------------------
#     # 3. 确定性约束：
#     # 对于全局中任意相同环境转移 (s, a, s_next)，若两步均分配到相同起始状态 u_i，
#     # 则它们的目标状态必须一致。
#     # ----------------------
#     # 先按 (s,a,s_next) 分组
#     sa_sprime_steps = {}
#     for m, traj in enumerate(traces):
#         for n, (s, a, r, s_next) in enumerate(traj):
#             key = (s, a, s_next)
#             if key not in sa_sprime_steps:
#                 sa_sprime_steps[key] = []
#             sa_sprime_steps[key].append((m, n, r))
#
#     for key, steps in sa_sprime_steps.items():
#         for idx1 in range(len(steps)):
#             for idx2 in range(idx1 + 1, len(steps)):
#                 m1, n1, r1 = steps[idx1]
#                 m2, n2, r2 = steps[idx2]
#                 for i in range(K):
#                     # 如果两步均在 u_i 上选取了不同的目标状态，则不允许同时为1
#                     for j in range(K):
#                         for jp in range(K):
#                             if j == jp:
#                                 continue
#                             prob += O[(m1, n1, i, j)] + O[(m2, n2, i, jp)] <= 1, \
#                                     f"determinism_{key}_{i}_{m1}_{n1}_{m2}_{n2}_{j}_{jp}"
#
#     # ----------------------
#     # 4. 奖励一致性约束：
#     # 对于全局中相同 (s, a, s_next) 的两步，
#     # 如果它们的奖励不同，则它们不能被分配到相同的起始状态 u_i。
#     # ----------------------
#     for key, steps in sa_sprime_steps.items():
#         for idx1 in range(len(steps)):
#             for idx2 in range(idx1 + 1, len(steps)):
#                 m1, n1, r1 = steps[idx1]
#                 m2, n2, r2 = steps[idx2]
#                 if r1 != r2:
#                     for i in range(K):
#                         prob += (pulp.lpSum(O[(m1, n1, i, j)] for j in range(K)) +
#                                  pulp.lpSum(O[(m2, n2, i, j)] for j in range(K))) <= 1, \
#                                 f"reward_consistency_{key}_{i}_{m1}_{n1}_{m2}_{n2}"
#
#     # ----------------------
#     # 5. 目标函数：最小化非自环转移数（即 i != j 的情况）
#     # ----------------------
#     objective = pulp.lpSum(O[(m, n, i, j)]
#                            for m, traj in enumerate(traces)
#                            for n in range(len(traj))
#                            for i in range(K)
#                            for j in range(K) if i != j)
#     prob += objective, "MinimizeNonSelfTransitions"
#
#     # 求解 ILP
#     solver = pulp.PULP_CBC_CMD(msg=True)
#     result = prob.solve(solver)
#     print("Solver Status:", pulp.LpStatus[result])
#
#     if pulp.LpStatus[result] != "Optimal":
#         print("未找到最优解！")
#         return None
#
#     # ----------------------
#     # 6. 提取 ILP 求解结果：
#     # 对于每条轨迹的每一步，记录分配的 (u_i, u_j)
#     # ----------------------
#     solution = {}  # solution[m] 为第 m 条轨迹的步映射列表
#     for m, traj in enumerate(traces):
#         solution[m] = []
#         for n in range(len(traj)):
#             assigned = None
#             for i in range(K):
#                 for j in range(K):
#                     if pulp.value(O[(m, n, i, j)]) > 0.5:
#                         assigned = (i, j)
#                         break
#                 if assigned is not None:
#                     break
#             solution[m].append({"assignment": assigned, "trace_step": traj[n]})
#
#     # ----------------------
#     # 7. 根据分配结果构造 Reward Machine 结构：
#     # 对于每个步骤，按 (u_i, s, a, s_next) 分组，
#     # 并记录对应的目标状态 u_j 以及奖励 r，
#     # 从而定义 δᵤ(u_i, (s,a,s_next)) = u_j 和 δᵣ(u_i, (s,a,s_next)) = r.
#     # ----------------------
#     rm_transitions = {}  # key: (i, s, a, s_next) -> (u_j, r) 以及出现该环境转移的列表
#     for m, traj in enumerate(traces):
#         for n, step in enumerate(traj):
#             s, a, r, s_next = step
#             # 取该步分配的 (u_i, u_j)
#             i, j = solution[m][n]["assignment"]
#             key = (i, s, a, s_next)
#             if key in rm_transitions:
#                 # 若已存在，则检查一致性
#                 prev_j, prev_r, _ = rm_transitions[key]
#                 if prev_j != j or prev_r != r:
#                     print(
#                         f"错误：对于 (u_{i}, {s}, {a}, {s_next}) 出现不一致分配: 之前(u_{i}->{prev_j}, r={prev_r}), 当前(u_{i}->{j}, r={r})")
#             else:
#                 rm_transitions[key] = (j, r, [])
#             # 记录此条转移对应的原始步（便于后续查看）
#             rm_transitions[key][2].append((s, a, r, s_next))
#
#     # 组装 Reward Machine 结构
#     # 状态集合为 0,1,...,K-1；初始状态设为 0
#     reward_machine = {
#         "states": list(range(K)),  # RM 状态 u0, u1, …, u{K-1}
#         "initial_state": 0,
#         "transitions": {}  # transitions: key (u_i, s, a, s_next) -> { "next_state": u_j, "reward": r, "steps": [...] }
#     }
#     for key, (j, r, steps) in rm_transitions.items():
#         i, s, a, s_next = key
#         reward_machine["transitions"][(i, s, a, s_next)] = {
#             "next_state": j,
#             "reward": r,
#             "steps": steps
#         }
#
#     return solution, reward_machine
