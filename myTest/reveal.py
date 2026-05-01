import pickle
import pprint

# 1. 替换为你实际的 pkl 文件路径 (注意修改为你真实的文件夹时间戳)
pkl_path = "Opt_Results_7MZM_8DET_20260409_185719/Top100_Optimized_Params.pkl"

# 2. 以二进制读取模式 ('rb') 打开文件并加载
with open(pkl_path, 'rb') as f:
    top_100_data = pickle.load(f)

print(f"🎉 成功解冻数据！总共包含 {len(top_100_data)} 个个体的详细参数。\n")

# 3. 提取排名第一（Rank 1）的个体看看
best_topo = top_100_data[0]

print("=========================================")
print(f"🥇 全局最优拓扑 (Rank 1, ID: {best_topo['id']})")
print(f"   最终得分 Cost: {best_topo['score']:.6f}")
print("=========================================\n")

print("【1. 拓扑连接矩阵】:")
print(best_topo['matrix'])

print("\n【2. 节点参数 (如 MZM 的偏置电压、射频信号等)】:")
# 使用 pprint (pretty print) 可以让字典打印得更整齐漂亮
pprint.pprint(best_topo['node_params'])

print("\n【3. 连线参数 (Delay Line 延迟时间)】:")
pprint.pprint(best_topo['edge_params'])

# 如果你想看第二名，就把 top_100_data[0] 改成 top_100_data[1]，以此类推。