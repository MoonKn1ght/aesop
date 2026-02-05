import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
from collections import defaultdict, deque
import random

### 端口编码
PORT_MAP = {
    1: ("out_top", "in_top"),
    2: ("out_top", "in_bottom"),
    3: ("out_bottom", "in_top"),
    4: ("out_bottom", "in_bottom"),
}


# 辅助函数：DAG分层 (用于画图)
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


# DAG 检查函数：确保没有闭环
def is_dag(A):
    n = A.shape[0]
    indeg = (A > 0).sum(axis=0)
    queue = deque([i for i in range(n) if indeg[i] == 0])
    visited_count = 0
    while queue:
        u = queue.popleft()
        visited_count += 1
        for v in range(n):
            if A[u, v] > 0:
                indeg[v] -= 1
                if indeg[v] == 0: queue.append(v)
    return visited_count == n


### 绘图相关函数
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
    x1, y1 = p1;
    x2, y2 = p2
    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2 + bend
    path = Path([(x1, y1), (xm, ym), (x2, y2)], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    ax.add_patch(PathPatch(path, ec="black", lw=1.1, fc="none", zorder=1, alpha=alpha, linestyle=style))


def visualize_topology(ax, A, N_mzm, N_det, title):
    levels = compute_levels(A.copy())
    layers = defaultdict(list)
    for i, lv in enumerate(levels):
        layers[lv].append(i)

    pos = {};
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
                ax.text(*pos[node], "Src", ha="center", va="center", fontsize=9, fontweight='bold',
                        bbox=dict(facecolor='yellow', alpha=1.0, edgecolor='orange', boxstyle='round,pad=0.3',
                                  zorder=20))
            elif node <= N_mzm:
                port_pos[node] = draw_mzm(ax, pos[node], f"M{node}")
            else:
                ax.text(*pos[node], f"D{node - N_mzm}", ha="center", va="center", fontsize=8, fontweight='bold',
                        bbox=dict(facecolor='#90EE90', alpha=1.0, edgecolor='green', boxstyle='round,pad=0.3',
                                  zorder=20))

    max_y, min_y = (max(all_ys), min(all_ys)) if all_ys else (2, -2)
    edge_idx = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if A[i, j] > 0:
                p1 = port_pos[i][PORT_MAP[A[i, j]][0]] if i in port_pos else pos[i]
                p2 = port_pos[j][PORT_MAP[A[i, j]][1]] if j in port_pos else pos[j]

                # 智能路由：跨层用拱桥，相邻用微弯
                level_diff = levels[j] - levels[i]
                if level_diff > 1:
                    side = 1 if edge_idx % 2 == 0 else -1
                    bend = (max_y + 1.5 - (p1[1] + p2[1]) / 2) if side == 1 else (min_y - 1.5 - (p1[1] + p2[1]) / 2)
                    draw_edge(ax, p1, p2, bend=bend, style='--', alpha=0.4)
                else:
                    bend = 0.6 * (((edge_idx % 5) / 2.0) - 1.0)
                    draw_edge(ax, p1, p2, bend=bend, style='-', alpha=0.8)
                edge_idx += 1
    ax.set_title(title);
    ax.axis("off");
    ax.autoscale_view()


# 连通性检查
def check_connectivity(A, N_mzm, N_det):
    n = A.shape[0];
    visited = {0};
    queue = deque([0])
    while queue:
        u = queue.popleft()
        neighbors = np.where(A[u, :] > 0)[0]
        for v in neighbors:
            if v not in visited: visited.add(v); queue.append(v)
    for i in range(1, n):  # 检查所有节点是否都被 Source 覆盖
        if i not in visited: return False
    return True


# 物理合法性检查
def is_physically_valid(A, N_mzm, N_det):
    indeg = A.astype(bool).sum(axis=0);
    outdeg = A.astype(bool).sum(axis=1)

    # 1. Source: 必须有出度
    if outdeg[0] == 0: return False

    # 【修改点】重新禁止 Source 直接连接到 Detector (Index范围: 1+N_mzm 到 n-1)
    if np.any(A[0, 1 + N_mzm:] > 0):
        return False

    # 2. MZM: 必须有入度和出度
    for i in range(1, 1 + N_mzm):
        if indeg[i] == 0 or outdeg[i] == 0: return False

    # 3. Detector: 必须有入度
    for i in range(1 + N_mzm, A.shape[0]):
        if indeg[i] == 0: return False

    return True


# 拓扑生成器
def generate_topology(N_mzm, N_det):
    n = 1 + N_mzm + N_det;
    A = np.zeros((n, n), dtype=int)

    # Source 拥有 1~N 的出度潜力 (通过给它分配足够的端口)
    free_out_ports = {0: [0] * (N_mzm)}
    for i in range(1, 1 + N_mzm): free_out_ports[i] = [0, 1]

    free_in_ports = {i: [0, 1] for i in range(1, 1 + N_mzm)}
    for i in range(1 + N_mzm, n): free_in_ports[i] = [0]

    def connect(u, v, out_p, in_p):
        mapping = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
        A[u, v] = mapping[(out_p, in_p)]
        free_out_ports[u].remove(out_p);
        free_in_ports[v].remove(in_p)

    # 强制第一跳到 MZM
    target = random.choice(range(1, 1 + N_mzm))
    connect(0, target, 0, random.choice(free_in_ports[target]))

    possible_sources = list(free_out_ports.keys())
    random.shuffle(possible_sources)
    for u in possible_sources:
        while free_out_ports[u]:
            if random.random() > 0.7: break  # 控制密度
            out_p = free_out_ports[u][0]

            # 筛选候选目标
            candidates = []
            for v in free_in_ports:
                if u == v or not free_in_ports[v] or A[u, v] > 0: continue

                # 【修改点】生成逻辑中直接排除 Source -> Detector 的可能
                if u == 0 and v >= 1 + N_mzm: continue

                candidates.append(v)

            if not candidates: break
            v = random.choice(candidates);
            in_p = random.choice(free_in_ports[v])
            connect(u, v, out_p, in_p)
    return A


### 主程序
if __name__ == "__main__":
    N_MZM, N_DET = 3, 4
    n_sample = 10
    topologies = []
    attempts = 0

    print("Generating strictly connected topologies (No Source-Detector direct links)...")
    while len(topologies) < n_sample and attempts < 100000:
        attempts += 1
        A = generate_topology(N_MZM, N_DET)

        if not is_dag(A.copy()): continue
        if not is_physically_valid(A, N_MZM, N_DET): continue
        if not check_connectivity(A, N_MZM, N_DET): continue

        topologies.append(A.copy())

    print(f"Generated {len(topologies)} valid topologies.")
    fig, axes = plt.subplots(2, 5, figsize=(30, 11))
    axes = axes.flatten()
    for i, A in enumerate(topologies):
        visualize_topology(axes[i], A, N_MZM, N_DET, title=f"Sample {i + 1}")
    plt.tight_layout();
    plt.show()