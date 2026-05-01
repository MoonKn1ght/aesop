import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path
from collections import defaultdict, deque
import networkx as nx

### ==========================================
### 1. 基础配置与辅助函数
### ==========================================

PORT_MAP = {
    1: ("out_top", "in_top"),
    2: ("out_top", "in_bottom"),
    3: ("out_bottom", "in_top"),
    4: ("out_bottom", "in_bottom"),
}

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
                if indeg[v] == 0:
                    q.append(v)
    return level


### ==========================================
### 2. 可视化绘图模块
### ==========================================

def draw_mzm(ax, center, label):
    x, y = center
    w, h = 1.2, 0.6
    rect = Rectangle((x - w / 2, y - h / 2), w, h, fc="#d2691e", ec="brown", lw=1.5, zorder=20)
    ax.add_patch(rect)
    ports = {
        "in_top": (x - w / 2, y + h / 4), "in_bottom": (x - w / 2, y - h / 4),
        "out_top": (x + w / 2, y + h / 4), "out_bottom": (x + w / 2, y - h / 4),
    }
    for p in ports.values(): ax.plot(*p, "ko", ms=3, zorder=21)
    ax.text(x, y, label, ha="center", va="center", fontsize=8, zorder=22, color='white', fontweight='bold')
    return ports

def draw_edge(ax, p1, p2, bend, style='solid', alpha=0.6):
    x1, y1 = p1
    x2, y2 = p2
    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2 + bend
    path = Path([(x1, y1), (xm, ym), (x2, y2)], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    arrow = FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=12,
                            color="black", lw=1.1, alpha=alpha, linestyle=style, zorder=1)
    ax.add_patch(arrow)

def visualize_topology(ax, A, N_mzm, N_det, title):
    levels = compute_levels(A.copy())
    layers = defaultdict(list)
    for i, lv in enumerate(levels):
        layers[lv].append(i)

    pos = {}
    port_pos = {}
    sorted_levels = sorted(layers.keys())
    V_SPACING = 2.5
    H_SPACING = 4.0

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
                ax.text(*pos[node], "Src", ha="center", va="center", fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='yellow', alpha=1.0, edgecolor='orange', boxstyle='round,pad=0.8', zorder=20))
            elif node <= N_mzm:
                port_pos[node] = draw_mzm(ax, pos[node], f"M{node}")
            else:
                ax.text(*pos[node], f"D{node - N_mzm}", ha="center", va="center", fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='#90EE90', alpha=1.0, edgecolor='green', boxstyle='round,pad=0.8', zorder=20))

    max_y, min_y = (max(all_ys), min(all_ys)) if all_ys else (2, -2)
    edge_idx = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if A[i, j] > 0:
                p1 = port_pos[i][PORT_MAP[A[i, j]][0]] if i in port_pos else pos[i]
                p2 = port_pos[j][PORT_MAP[A[i, j]][1]] if j in port_pos else pos[j]

                level_diff = levels[j] - levels[i]
                if level_diff > 1:
                    side = 1 if edge_idx % 2 == 0 else -1
                    bend = (max_y + 1.5 - (p1[1] + p2[1]) / 2) if side == 1 else (min_y - 1.5 - (p1[1] + p2[1]) / 2)
                    draw_edge(ax, p1, p2, bend=bend, style='--', alpha=0.4)
                else:
                    bend = 0.6 * (((edge_idx % 5) / 2.0) - 1.0)
                    draw_edge(ax, p1, p2, bend=bend, style='-', alpha=0.8)
                edge_idx += 1
    ax.set_title(title)
    ax.axis("off")
    ax.autoscale_view()


### ==========================================
### 3. 物理检验与拓扑生成模块
### ==========================================

def check_connectivity(A, N_mzm, N_det):
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
    for i in range(1, n):
        if i not in visited: return False
    return True

def is_physically_valid(A, N_mzm, N_det):
    indeg = A.astype(bool).sum(axis=0)
    outdeg = A.astype(bool).sum(axis=1)

    if outdeg[0] == 0: return False
    if np.any(A[0, 1 + N_mzm:] > 0): return False

    for i in range(1, 1 + N_mzm):
        if indeg[i] == 0 or outdeg[i] == 0: return False

    for i in range(1 + N_mzm, A.shape[0]):
        if indeg[i] == 0: return False

    return True

def generate_all_topologies_exhaustive(N_mzm, N_det):
    n = 1 + N_mzm + N_det
    out_ports = [(0, 0)] * N_mzm
    for i in range(1, 1 + N_mzm):
        out_ports.extend([(i, 0), (i, 1)])

    in_ports = []
    for i in range(1, 1 + N_mzm):
        in_ports.extend([(i, 0), (i, 1)])
    for i in range(1 + N_mzm, n):
        in_ports.append((i, 0))

    valid_matrices = set()
    results = []

    def dfs(out_idx, current_in_ports, current_A):
        if out_idx == len(out_ports):
            if is_physically_valid(current_A, N_mzm, N_det) and check_connectivity(current_A, N_mzm, N_det):
                a_tuple = tuple(current_A.flatten())
                if a_tuple not in valid_matrices:
                    valid_matrices.add(a_tuple)
                    results.append(current_A.copy())
            return

        u, out_p = out_ports[out_idx]
        dfs(out_idx + 1, current_in_ports, current_A)

        for i, (v, in_p) in enumerate(current_in_ports):
            if u >= v: continue
            if u == 0 and v >= 1 + N_mzm: continue
            if current_A[u, v] > 0: continue

            mapping = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
            current_A[u, v] = mapping[(out_p, in_p)]

            next_in_ports = current_in_ports[:i] + current_in_ports[i + 1:]
            dfs(out_idx + 1, next_in_ports, current_A)
            current_A[u, v] = 0

    initial_A = np.zeros((n, n), dtype=int)
    dfs(0, in_ports, initial_A)
    return results


### ==========================================
### 4. 同构去重与辅助打印模块
### ==========================================

def filter_isomorphic_topologies(topologies, N_mzm, N_det):
    print("开始进行物理同构去重（利用特征签名与图论等价性筛选）...")
    node_match = nx.algorithms.isomorphism.categorical_node_match('type', -1)
    edge_match = nx.algorithms.isomorphism.categorical_edge_match('port', 0)

    unique_topologies = []
    buckets = defaultdict(list)

    for A in topologies:
        n = A.shape[0]
        G = nx.DiGraph()

        for i in range(n):
            ntype = 0 if i == 0 else (1 if i <= N_mzm else 2)
            G.add_node(i, type=ntype)

        for i in range(n):
            for j in range(n):
                if A[i, j] > 0:
                    G.add_edge(i, j, port=A[i, j])

        in_degrees, out_degrees = dict(G.in_degree()), dict(G.out_degree())
        src_sig = (in_degrees[0], out_degrees[0])
        mzm_sig = tuple(sorted([(in_degrees[i], out_degrees[i]) for i in range(1, 1 + N_mzm)]))
        det_sig = tuple(sorted([(in_degrees[i], out_degrees[i]) for i in range(1 + N_mzm, n)]))
        signature = (src_sig, mzm_sig, det_sig)

        is_duplicate = False
        for UG in buckets[signature]:
            if nx.is_isomorphic(G, UG, node_match=node_match, edge_match=edge_match):
                is_duplicate = True
                break

        if not is_duplicate:
            buckets[signature].append(G)
            unique_topologies.append(A.copy())

    return unique_topologies

def print_formatted_matrix(A, N_mzm, N_det, topo_idx):
    n = A.shape[0]
    headers = ["Src(0)"] + [f"M{i}({i})" for i in range(1, N_mzm + 1)] + \
              [f"D{i}({i + N_mzm})" for i in range(1, N_det + 1)]

    print(f"\n--- Topology {topo_idx} Adjacency Matrix ---")
    header_str = " " * 10 + "".join([f"{h:>10}" for h in headers])
    print(header_str)
    print("-" * len(header_str))

    for i in range(n):
        row_str = f"{headers[i]:<10}"
        for j in range(n):
            val = A[i, j]
            display_val = f"{val:>10}" if val != 0 else f"{'.':>10}"
            row_str += display_val
        print(row_str)
    print("-" * len(header_str))


### ==========================================
### 5. 主程序流 (已梳理整合)
### ==========================================

if __name__ == "__main__":
    # --- 核心参数配置区 ---
    N_MZM = 3  # DO-MZM 数量 (4通道系统通常配置为3)
    N_DET = 4  # 探测器 数量 (4通道系统配置为4)
    ENABLE_ISOMORPHISM_FILTER = True  # 开关：True 表示执行同构去重，False 表示保留所有穷举结果
    # ----------------------

    print(f"阶段 1：开始穷举搜索 {N_MZM} MZM + {N_DET} DET 的严格连通数学拓扑（请耐心等待）...")
    raw_topologies = generate_all_topologies_exhaustive(N_MZM, N_DET)
    print(f"穷举完成！共找到 {len(raw_topologies)} 种数学合法矩阵。")

    # 阶段 2：根据配置执行或跳过去重
    if ENABLE_ISOMORPHISM_FILTER:
        print("\n阶段 2：执行网络图同构算法，剔除物理等价的重复结构...")
        final_topologies = filter_isomorphic_topologies(raw_topologies, N_MZM, N_DET)
        status_str = "【物理唯一】"
    else:
        print("\n阶段 2：已跳过去重步骤！保留原始穷举数据...")
        final_topologies = raw_topologies
        status_str = "【全部穷举】"

    n_total = len(final_topologies)
    print(f"处理完成！最终生成的{status_str}拓扑数量为：{n_total} 种。")

    # 阶段 3：数据保存与可视化
    if n_total > 0:
        # 1. 保存 Numpy 数据文件
        npy_filename = f"topologies_{N_MZM}MZM_{N_DET}DET_{'unique' if ENABLE_ISOMORPHISM_FILTER else 'all'}.npy"
        np.save(npy_filename, final_topologies)
        print(f"【数据保存】矩阵已完整保存至 '{npy_filename}'。")

        # 2. 绘制图片并保存
        save_dir = f"topology_images_{'unique' if ENABLE_ISOMORPHISM_FILTER else 'all'}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        n_per_fig = 9
        n_figs = math.ceil(n_total / n_per_fig)

        print(f"正在生成 {n_total} 个拓扑的图像并分批保存至 '{save_dir}' 文件夹...")
        for f_idx in range(n_figs):
            fig, axes = plt.subplots(3, 3, figsize=(18, 12))
            axes = axes.flatten()

            start_idx = f_idx * n_per_fig
            for i in range(n_per_fig):
                topo_idx = start_idx + i
                ax = axes[i]

                if topo_idx < n_total:
                    A = final_topologies[topo_idx]
                    visualize_topology(ax, A, N_MZM, N_DET, title=f"Topology {topo_idx + 1}")
                else:
                    ax.axis('off') # 空白的子图隐藏坐标轴

            plt.tight_layout()
            file_path = os.path.join(save_dir, f"batch_view_{f_idx + 1}.png")
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        print("【图片保存】所有拓扑图像均已成功生成！")