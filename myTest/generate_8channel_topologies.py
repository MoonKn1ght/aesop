import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path
from collections import deque, defaultdict
import random
import time
import os

# ==========================================
# 0. 绘图相关辅助函数
# ==========================================
PORT_MAP = {1: ("out_top", "in_top"), 2: ("out_top", "in_bottom"),
            3: ("out_bottom", "in_top"), 4: ("out_bottom", "in_bottom")}


def compute_levels(A, source=0):
    n = A.shape[0]
    indeg = A.astype(bool).sum(axis=0)
    q = deque([source])
    level = [-1] * n
    level[source] = 0
    while q:
        u = q.popleft()
        for v in range(n):
            if A[u, v] > 0:
                level[v] = max(level[v], level[u] + 1)
                indeg[v] -= 1
                if indeg[v] == 0: q.append(v)
    return level


def draw_mzm(ax, center, label):
    x, y = center
    w, h = 1.2, 0.6
    rect = Rectangle((x - w / 2, y - h / 2), w, h, fc="#d2691e", ec="brown", lw=1.5, zorder=20)
    ax.add_patch(rect)
    ports = {"in_top": (x - w / 2, y + h / 4), "in_bottom": (x - w / 2, y - h / 4),
             "out_top": (x + w / 2, y + h / 4), "out_bottom": (x + w / 2, y - h / 4)}
    for p in ports.values(): ax.plot(*p, "ko", ms=3, zorder=21)
    ax.text(x, y, label, ha="center", va="center", fontsize=8, zorder=22, color='white', fontweight='bold')
    return ports


def draw_edge(ax, p1, p2, bend=0.0, style='solid', alpha=0.6, color="black", lw=1.2):
    arrow = FancyArrowPatch(
        posA=p1, posB=p2, connectionstyle=f"arc3,rad={bend}",
        arrowstyle='-|>', mutation_scale=12, color=color, lw=lw, alpha=alpha, linestyle=style, zorder=1
    )
    ax.add_patch(arrow)


def visualize_topology(ax, A, N_mzm, N_det, title):
    levels = compute_levels(A.copy())
    layers = defaultdict(list)
    for i, lv in enumerate(levels): layers[lv].append(i)

    pos = {};
    port_pos = {}
    sorted_levels = sorted(layers.keys())

    V_SPACING, H_SPACING = 3.5, 5.5
    all_ys = []

    for lv in sorted_levels:
        nodes = layers[lv]
        n_nodes = len(nodes)
        total_span = (n_nodes - 1) * V_SPACING
        ys = np.linspace(total_span / 2, -total_span / 2, n_nodes) if n_nodes > 1 else [0]
        for i, node in enumerate(nodes):
            pos[node] = (lv * H_SPACING, ys[i])
            all_ys.append(ys[i])
            if node == 0:
                ax.text(*pos[node], "Src", ha="center", va="center", fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='#FFD700', alpha=0.9, edgecolor='#DAA520', boxstyle='round,pad=0.4',
                                  zorder=20))
            elif node <= N_mzm:
                port_pos[node] = draw_mzm(ax, pos[node], f"M{node}")
            else:
                ax.text(*pos[node], f"D{node - N_mzm}", ha="center", va="center", fontsize=9, fontweight='bold',
                        bbox=dict(facecolor='#98FB98', alpha=0.9, edgecolor='#2E8B57', boxstyle='round,pad=0.4',
                                  zorder=20))

    edge_idx = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if A[i, j] > 0:
                p1 = port_pos[i][PORT_MAP[A[i, j]][0]] if i in port_pos else pos[i]
                p2 = port_pos[j][PORT_MAP[A[i, j]][1]] if j in port_pos else pos[j]
                level_diff = levels[j] - levels[i]
                y_diff = p2[1] - p1[1]

                if level_diff == 1:
                    bend = 0.0 if abs(y_diff) < 0.1 else (0.15 if y_diff > 0 else -0.15)
                    draw_edge(ax, p1, p2, bend=bend, style='-', alpha=0.85, color="#2F4F4F", lw=1.2)
                else:
                    bend = 0.3 * level_diff * (1 if edge_idx % 2 == 0 else -1)
                    draw_edge(ax, p1, p2, bend=bend, style='--', alpha=0.35, color="#4682B4", lw=1.0)
                edge_idx += 1

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.axis("off")
    ax.autoscale_view()
    ax.margins(0.1)


# ==========================================
# 1. 严格物理规则验证器 (1端口 = 1波导)
# ==========================================
def check_connectivity(A):
    n = A.shape[0]
    visited = {0}
    queue = deque([0])
    while queue:
        u = queue.popleft()
        neighbors = np.where(A[u, :] > 0)[0]
        for v in neighbors:
            if v not in visited:
                visited.add(v)
                queue.append(v)
    return len(visited) == n


# ==========================================
# 1. 严格物理规则验证器 (Source可多分发，MZM/DET严守1端1线)
# ==========================================
def is_physically_valid_strict(A, N_mzm, N_det):
    n = A.shape[0]

    # 规则 1: Source 必须至少输出 1 根波导 (不能是断头路)，且绝对不能直接连到探测器
    if np.count_nonzero(A[0, :]) == 0: return False
    if np.any(A[0, 1 + N_mzm:] > 0): return False

    for i in range(1, 1 + N_mzm):
        row = A[i, :]
        col = A[:, i]

        # 规则 2: MZM 的每个输出端口 (Top/Bottom) 物理上最多只能接 1 根波导
        if np.sum((row == 1) | (row == 2)) > 1: return False  # Top 口接了多条线
        if np.sum((row == 3) | (row == 4)) > 1: return False  # Bottom 口接了多条线

        # 规则 3: MZM 的每个输入端口 (Top/Bottom) 物理上最多只能进 1 根波导
        if np.sum((col == 1) | (col == 3)) > 1: return False
        if np.sum((col == 2) | (col == 4)) > 1: return False

        # 规则 4: 不允许存在“断头路”元件
        if np.count_nonzero(col) == 0: return False
        if np.count_nonzero(row) == 0: return False

    for j in range(1 + N_mzm, n):
        col = A[:, j]
        # 规则 5: 探测器物理上只能接 1 根波导
        if np.count_nonzero(col) != 1: return False

    return True

def generate_traditional_padc_tree(N_mzm, N_det):
    """
    生成传统的满二叉树 PADC 拓扑
    严格按照 1分2, 2分4, 4分8 的端口映射关系连线
    """
    n = 1 + N_mzm + N_det
    A = np.zeros((n, n), dtype=int)

    # 1. Source (0) 连接到第一级 MZM (1) 的 Top Input
    A[0, 1] = 1  # (out_top -> in_top)

    # 2. 构造 MZM 的二叉树级联 (针对 1 到 N_mzm 的非叶子节点)
    for i in range(1, N_mzm // 2 + 1):
        left_child = 2 * i
        right_child = 2 * i + 1

        if left_child <= N_mzm:
            # 左路：当前 MZM 的 out_top 接下一级 MZM 的 in_top
            A[i, left_child] = 1
        if right_child <= N_mzm:
            # 右路：当前 MZM 的 out_bottom 接下一级 MZM 的 in_top
            A[i, right_child] = 3

    # 3. 叶子节点 MZM 连接到探测器 DET
    det_idx = 1 + N_mzm
    for i in range(N_mzm // 2 + 1, N_mzm + 1):
        if det_idx < n:
            A[i, det_idx] = 1  # Top out -> DET
            det_idx += 1
        if det_idx < n:
            A[i, det_idx] = 3  # Bottom out -> DET
            det_idx += 1

    return A
# ==========================================
# 2. 严格受限的随机图生成器 (光速构造版 Fast)
# ==========================================
def generate_strict_dag_matrix(N_mzm, N_det):
    n = 1 + N_mzm + N_det
    while True:
        A = np.zeros((n, n), dtype=int)

        # 记录管脚占用
        out_counts = {i: [0, 0] for i in range(1, 1 + N_mzm)}
        in_counts = {i: [0, 0] for i in range(1, 1 + N_mzm)}

        success = True

        # 步骤 1：从左到右定向施工，确保每个元件都有且只有 1 个保底输入
        for v in range(1, n):
            # MZM 可以接 Source 或前面的 MZM；探测器只能接 MZM
            possible_u = list(range(0, v)) if v <= N_mzm else list(range(1, 1 + N_mzm))
            random.shuffle(possible_u)

            connected = False
            for u in possible_u:
                # 找源头的空闲输出口
                if u == 0:
                    out_p = 0  # Source 不限量
                else:
                    free_outs = [p for p in [0, 1] if out_counts[u][p] == 0]
                    if not free_outs: continue
                    out_p = random.choice(free_outs)

                # 接收端的空闲输入口
                in_p = random.choice([0, 1]) if v <= N_mzm else 0

                # 连线定稿
                A[u, v] = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}[(out_p, in_p)]
                if u != 0: out_counts[u][out_p] += 1
                if v <= N_mzm: in_counts[v][in_p] += 1

                connected = True
                break

            # 如果某个元件实在找不到上游接盘，说明这张图废了，提前重开
            if not connected:
                success = False
                break

        if not success: continue

        # 步骤 2：给 MZM 补充 0~3 条额外的交叉线，增加光学干涉的复杂度
        extra_edges = random.randint(0, 3)
        for _ in range(extra_edges):
            # 【完美修复】：起点 u 最多只能抽到 N_mzm - 1
            # 这样保证了它后面永远有下游 MZM (v) 可以连接
            u = random.randint(0, N_mzm - 1)
            v = random.randint(max(1, u + 1), N_mzm)  # 额外线只连给 MZM
            if A[u, v] > 0: continue

            if u == 0:
                out_p = 0
            else:
                free_outs = [p for p in [0, 1] if out_counts[u][p] == 0]
                if not free_outs: continue
                out_p = random.choice(free_outs)

            free_ins = [p for p in [0, 1] if in_counts[v][p] == 0]
            if not free_ins: continue
            in_p = random.choice(free_ins)

            A[u, v] = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}[(out_p, in_p)]
            if u != 0: out_counts[u][out_p] += 1
            in_counts[v][in_p] += 1
            
        # 步骤 3：最后过一遍体检（此时绝大部分图都能一次性过关）
        if is_physically_valid_strict(A, N_mzm, N_det) and check_connectivity(A):
            return A
# ==========================================
# 3. 实时图同构去重器
# ==========================================
class IsomorphismFilter:
    def __init__(self, N_mzm, N_det):
        self.N_mzm = N_mzm
        self.N_det = N_det
        self.node_match = nx.algorithms.isomorphism.categorical_node_match('type', -1)
        self.edge_match = nx.algorithms.isomorphism.categorical_edge_match('port', 0)
        self.buckets = defaultdict(list)

    def is_unique(self, A):
        n = A.shape[0]
        G = nx.DiGraph()
        for i in range(n):
            ntype = 0 if i == 0 else (1 if i <= self.N_mzm else 2)
            G.add_node(i, type=ntype)
        for i in range(n):
            for j in range(n):
                if A[i, j] > 0: G.add_edge(i, j, port=A[i, j])

        in_degrees = dict(G.in_degree());
        out_degrees = dict(G.out_degree())
        src_sig = (in_degrees[0], out_degrees[0])
        mzm_sig = tuple(sorted([(in_degrees[i], out_degrees[i]) for i in range(1, 1 + self.N_mzm)]))
        det_sig = tuple(sorted([(in_degrees[i], out_degrees[i]) for i in range(1 + self.N_mzm, n)]))
        signature = (src_sig, mzm_sig, det_sig)

        for UG in self.buckets[signature]:
            if nx.is_isomorphic(G, UG, node_match=self.node_match, edge_match=self.edge_match):
                return False

        self.buckets[signature].append(G)
        return True


# ==========================================
# 4. 主程序：收集纯净拓扑并画图
# ==========================================
if __name__ == "__main__":
    N_MZM, N_DET = 7, 8
    TARGET_UNIQUE = 1000

    SAVE_BASE_DIR = "/root/autodl-tmp/aesop/asope_data"
    os.makedirs(SAVE_BASE_DIR, exist_ok=True)

    unique_topologies = []
    iso_filter = IsomorphismFilter(N_MZM, N_DET)

    print(f"🚀 开始进行 {N_DET} 通道 ({N_MZM} MZM, {N_DET} DET) 严格物理拓扑盲搜...")

    # ================= 新增：预置传统架构 =================
    traditional_A = generate_traditional_padc_tree(N_MZM, N_DET)
    if is_physically_valid_strict(traditional_A, N_MZM, N_DET) and check_connectivity(traditional_A):
        unique_topologies.append(traditional_A)
        iso_filter.is_unique(traditional_A)  # 注册到同构器，防止后续随机生成重复的传统架构
        print("  -> ✅ 已成功固定注入传统 1-2-4-8 PADC 架构作为第一个拓扑 (ID: 1)")
    else:
        print("  -> ⚠️ 传统架构生成未通过物理检查，请核对端口规则。")
    # ======================================================

    attempts = 0
    start_time = time.time()

    # 后续继续用蒙特卡洛方法补齐剩下的数量
    while len(unique_topologies) < TARGET_UNIQUE:
        attempts += 1
        A = generate_strict_dag_matrix(N_MZM, N_DET)
        if iso_filter.is_unique(A):
            unique_topologies.append(A)
            if len(unique_topologies) % 10 == 0:
                print(f"  -> 已收集 {len(unique_topologies)} / {TARGET_UNIQUE} 个纯净拓扑... (尝试生成: {attempts} 次)")

    elapsed = time.time() - start_time
    print(f"\n✅ 矩阵收集完成！耗时 {elapsed:.2f} 秒。")

    # 【修改】：拼接绝对路径来保存 numpy 文件
    npy_filename = os.path.join(SAVE_BASE_DIR, f"topologies_{N_MZM}MZM_{N_DET}DET_unique_all.npy")
    np.save(npy_filename, unique_topologies)
    print(f"💾 数据已安全保存至：'{npy_filename}'。")

    # ==========================================
    # 5. 批量绘制拓扑图并归档
    # ==========================================
    # 【修改】：拼接绝对路径来创建图片保存文件夹
    img_dir = os.path.join(SAVE_BASE_DIR, f"Generated_Topologies_{N_MZM}MZM_{N_DET}DET_Strict_Images")
    os.makedirs(img_dir, exist_ok=True)
    print(f"\n🎨 正在绘制极其纯净的拓扑图纸，并保存至 '{img_dir}' (请稍候)...")

    for idx, A in enumerate(unique_topologies):
        fig, ax = plt.subplots(figsize=(12, 8))
        title = f"Strict Physical Topology ID: {idx + 1}"
        visualize_topology(ax, A, N_MZM, N_DET, title)

        img_path = os.path.join(img_dir, f"Topology_{idx + 1:04d}.png")
        plt.savefig(img_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        if (idx + 1) % 50 == 0:
            print(f"  -> 图纸绘制进度: {idx + 1} / {TARGET_UNIQUE} 张")

    print(f"\n🎉 所有纯净图纸绘制完毕！你可以直接去 {SAVE_BASE_DIR} 文件夹里查看结果啦！")