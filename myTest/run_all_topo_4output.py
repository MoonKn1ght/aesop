import sys
print(f"--- 正在使用的 Python 路径: {sys.executable} ---")
import pathlib
import os
import platform
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path
from collections import defaultdict, deque
import autograd.numpy as np
import copy
import time
import math  # 【新增】：用于全家福的分页计算
import pickle
import multiprocessing as mp
from functools import partial

# ==========================================
# 0. 环境路径设置
# ==========================================
parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

from lib.functions import InputOutput, parse_command_line_args
from lib.graph import Graph
from simulator.fiber.node_types import MultiPath, SinglePath
from simulator.fiber.assets.propagator import Propagator
from simulator.fiber.assets.functions import power_
from simulator.fiber.node_types_subclasses.inputs import PulsedLaser
from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink
from algorithms.parameter_optimization import parameters_optimize
from simulator.fiber.evaluator import Evaluator

plt.close('all')

# ==========================================
# 1. 绘图相关辅助函数
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


def draw_edge(ax, p1, p2, bend, style='solid', alpha=0.6):
    x1, y1 = p1;
    x2, y2 = p2
    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2 + bend
    path = Path([(x1, y1), (xm, ym), (x2, y2)], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    arrow = FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=12,
                            color="black", lw=1.1, alpha=alpha, linestyle=style, zorder=1)
    ax.add_patch(arrow)


def visualize_topology(ax, A, N_mzm, N_det, title):
    levels = compute_levels(A.copy())
    layers = defaultdict(list)
    for i, lv in enumerate(levels): layers[lv].append(i)

    pos = {};
    port_pos = {}
    sorted_levels = sorted(layers.keys())
    V_SPACING, H_SPACING = 2.5, 4.0
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
                level_diff = levels[j] - levels[i]
                if level_diff > 1:
                    side = 1 if edge_idx % 2 == 0 else -1
                    bend = (max_y + 1.5 - (p1[1] + p2[1]) / 2) if side == 1 else (min_y - 1.5 - (p1[1] + p2[1]) / 2)
                    draw_edge(ax, p1, p2, bend=bend, style='--', alpha=0.4)
                else:
                    bend = 0.6 * (((edge_idx % 5) / 2.0) - 1.0)
                    draw_edge(ax, p1, p2, bend=bend, style='-', alpha=0.8)
                edge_idx += 1
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis("off")
    ax.autoscale_view()


# ==========================================
# 2. 本地组件 (加入防弹容错机制)
# ==========================================
class Local_SafeDelayLine(SinglePath):
    node_acronym = 'DL'
    number_of_parameters = 1
    node_lock = True

    def __init__(self, **kwargs):
        self.default_parameters = [0.0]
        self.upper_bounds = [10e-9]
        self.lower_bounds = [0.0]
        self.data_types = ['float']
        self.step_sizes = [None]
        self.parameter_imprecisions = [0.0]
        self.parameter_units = ['s']
        self.parameter_locks = [False]
        self.parameter_names = ['delay']
        self.parameter_symbols = [r"$\tau$"]
        self._n = 1.444
        super().__init__(**kwargs)

    def propagate(self, state, propagator, save_transforms=False):
        if state is None:
            state = np.zeros_like(propagator.t)
        elif isinstance(state, list):
            if len(state) > 0 and state[0] is not None:
                state = state[0]
            else:
                state = np.zeros_like(propagator.t)

        state = np.squeeze(state)
        delay = self.parameters[0]
        dt = propagator.dt
        n = len(state)
        spectrum = np.fft.fft(state)
        freqs = np.fft.fftfreq(n, d=dt)
        phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay)
        return np.squeeze(np.fft.ifft(spectrum * phase_shift))


class Local_MZM_Modulated(MultiPath):
    node_acronym = 'MZM_MOD'
    node_lock = False

    def __init__(self, **kwargs):
        self.number_of_parameters = 6
        self.parameter_names = ['v_pi', 'v_bias', 'v_rf', 'f_rf', 'phase_rf', 'loss']
        self.default_parameters = [3.5, 1.75, 1.0, 2.5e9, 0.0, 0.1]
        self.upper_bounds = [10.0, 10.0, 20.0, 50.0e9, 2 * np.pi, 1.0]
        self.lower_bounds = [1.0, -10.0, 0.0, 0.0, -2 * np.pi, 0.0]
        self.data_types = ['float'] * 6
        self.step_sizes = [0.1] * 6
        self.parameter_imprecisions = [0.01] * 6
        self.parameter_units = ['V'] * 3 + ['Hz', 'rad', 'dB']
        self.parameter_locks = [True, False, False, True, False, True]
        self.parameter_symbols = [r"$V_{\pi}$", r"$V_{bias}$", r"$V_{RF}$", r"$f_{RF}$", r"$\theta_{RF}$", "IL"]
        super().__init__(**kwargs)
        self._range_input_edges = [1, 2]
        self.num_inputs = 2
        self.num_outputs = 2

    def update_attributes(self, num_inputs, num_outputs):
        self.num_inputs = 2
        self.num_outputs = 2
        if not hasattr(self, 'parameters') or len(self.parameters) != 6:
            self.parameters = self.default_parameters

    def propagate(self, states, propagator, num_inputs, num_outputs, save_transforms=False):
        v_pi, v_bias, v_rf, f_rf, phase_rf, loss = self.parameters
        t_len = len(np.squeeze(propagator.t))
        clean_states = []

        if states is None:
            states = []
        elif not isinstance(states, list):
            states = [states]

        for s in states:
            if s is None:
                clean_states.append(np.zeros(t_len, dtype=complex))
            else:
                clean_states.append(np.squeeze(s))

        while len(clean_states) < 2:
            clean_states.append(np.zeros(t_len, dtype=complex))

        in_mat = np.stack(clean_states[:2], axis=0)

        t = np.squeeze(propagator.t)
        rf_signal = v_rf * np.cos(2 * np.pi * f_rf * t + phase_rf)
        phi_t = (np.pi * (v_bias + rf_signal)) / (2.0 * v_pi)
        cos_phi = np.cos(phi_t)
        sin_phi = np.sin(phi_t)

        E_in1 = in_mat[0, :]
        E_in2 = in_mat[1, :]
        factor = (1.0 - loss)

        out1 = (E_in1 * cos_phi - 1j * E_in2 * sin_phi) * factor
        out2 = (-1j * E_in1 * sin_phi + E_in2 * cos_phi) * factor
        return [out1, out2]


class Local_TerminalSource(TerminalSource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._range_output_edges = (1, 10)  # 放开输出端口限制

    def update_attributes(self, num_inputs, num_outputs):
        self.num_outputs = num_outputs

    def propagate(self, states, propagator, num_inputs=0, num_outputs=1, save_transforms=False):
        if isinstance(states, list) and len(states) > 0:
            base_state = states[0]
        else:
            base_state = states
        return [base_state] * num_outputs


class Local_StrictEvaluator(Evaluator):
    def __init__(self, propagator, targets, evaluation_nodes):
        super().__init__()
        self.propagator = propagator
        self.targets = targets
        self.evaluation_nodes = evaluation_nodes
        self.masks = {}
        self.target_norms = {}
        # 预先计算目标波形的归一化形态和噪声掩码
        for nid, target_field in self.targets.items():
            p_tgt = power_(target_field)
            max_p = np.max(p_tgt) + 1e-15
            norm_tgt = np.squeeze(p_tgt / max_p)
            self.target_norms[nid] = norm_tgt
            # 定义非脉冲区域（低于最大值5%的区域），用于惩罚旁瓣和噪声
            self.masks[nid] = np.squeeze((norm_tgt < 0.05).astype(float))

    def evaluate_graph(self, graph, propagator=None):
        return self.evaluate(graph)

    def evaluate(self, graph):
        total_cost = 0.0
        for node in self.evaluation_nodes:
            signal = graph.measure_propagator(node)

            if isinstance(signal, list):
                if len(signal) > 0 and signal[0] is not None:
                    signal = signal[0]
                else:
                    signal = None

            if signal is None:
                total_cost += 5000.0
                continue

            # 计算当前探测器接收到的归一化光功率
            p_signal = np.abs(signal) ** 2
            max_sig = np.max(p_signal) + 1e-15
            p_sig_norm = np.squeeze(p_signal / max_sig)

            # 获取固定位置的目标波形和掩码
            norm_tgt = self.target_norms[node]
            mask = self.masks[node]

            # 【核心修改】：直接计算点对点的均方误差 (MSE)，不进行任何平移寻找最大相关性
            shape_error = np.mean((p_sig_norm - norm_tgt) ** 2)

            # 计算噪声惩罚项（强烈压制非脉冲窗口内的能量漏出）
            noise_content = p_sig_norm * mask
            penalty = np.mean(noise_content ** 2) * 20.0

            total_cost += (shape_error + penalty)
        return total_cost


class Local_XCorrEvaluator(Evaluator):
    def __init__(self, propagator, targets, evaluation_nodes):
        super().__init__()
        self.propagator = propagator
        self.targets = targets
        self.evaluation_nodes = evaluation_nodes
        self.masks = {}
        self.target_norms = {}
        for nid, target_field in self.targets.items():
            p_tgt = power_(target_field)
            max_p = np.max(p_tgt) + 1e-15
            norm_tgt = np.squeeze(p_tgt / max_p)
            self.target_norms[nid] = norm_tgt
            self.masks[nid] = np.squeeze((norm_tgt < 0.05).astype(float))

    def evaluate_graph(self, graph, propagator=None):
        return self.evaluate(graph)

    def evaluate(self, graph):
        total_cost = 0.0
        for node in self.evaluation_nodes:
            signal = graph.measure_propagator(node)

            if isinstance(signal, list):
                if len(signal) > 0 and signal[0] is not None:
                    signal = signal[0]
                else:
                    signal = None

            if signal is None:
                total_cost += 5000.0
                continue

            p_signal = np.abs(signal) ** 2
            max_sig = np.max(p_signal) + 1e-15
            p_sig_norm = np.squeeze(p_signal / max_sig)
            norm_tgt = self.target_norms[node]
            xcorr = np.correlate(p_sig_norm, norm_tgt, mode='full')
            best_shift_idx = np.argmax(xcorr) - (len(norm_tgt) - 1)
            shifted_tgt = np.roll(norm_tgt, best_shift_idx)
            shifted_mask = np.roll(self.masks[node], best_shift_idx)
            shape_error = np.mean((p_sig_norm - shifted_tgt) ** 2)
            noise_content = p_sig_norm * shifted_mask
            penalty = np.mean(noise_content ** 2) * 20.0
            total_cost += (shape_error + penalty)
        return total_cost


# ==========================================
# 3. 动态拓扑图构建引擎
# ==========================================
def build_graph_from_matrix(A, input_laser_template, N_mzm, N_det):
    nodes = {}
    edges = {}
    n = A.shape[0]

    nodes[0] = Local_TerminalSource()  # 使用我们自定义的安全 Source
    nodes[0].node_acronym = 'SRC'

    for i in range(1, 1 + N_mzm):
        mzm = Local_MZM_Modulated()
        mzm.node_acronym = f'MZM_{i}'
        mzm.parameter_locks[3] = False
        nodes[i] = mzm

    for i in range(1 + N_mzm, n):
        ch_idx = i - N_mzm
        nodes[i] = TerminalSink(node_name=f'sink_{ch_idx}')
        nodes[i].node_acronym = f'CH_{ch_idx}'

    port_idx_map = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}
    current_in_port = {i: 0 for i in range(n)}

    for u in range(n):
        for v in range(n):
            if A[u, v] > 0:
                out_p, original_in_p = port_idx_map[A[u, v]]
                safe_in_p = current_in_port[v]
                current_in_port[v] += 1

                if u == 0:
                    edge_obj = copy.deepcopy(input_laser_template)
                    edge_obj.src_port_idx = out_p
                    edge_obj.dst_port_idx = safe_in_p
                    edges[(u, v)] = edge_obj
                else:
                    dl = Local_SafeDelayLine()
                    dl.parameters = [0.0]
                    dl.node_lock = True
                    dl.src_port_idx = out_p
                    dl.dst_port_idx = safe_in_p
                    edges[(u, v)] = dl

    return Graph.init_graph(nodes, edges)


# 1. 抽离出一个“打工函数”，专门处理单个拓扑
def optimize_single_topology(idx_and_matrix, input_laser_tpl, n_mzm, n_det, prop, eval_obj):
    idx, A = idx_and_matrix
    print(f"[Core {mp.current_process().name}] 开始优化 Topology {idx + 1}")

    # 重新构建 Graph（避免进程间的内存冲突）
    graph = build_graph_from_matrix(A, input_laser_tpl, n_mzm, n_det)
    graph.initialize_func_grad_hess(prop, eval_obj, exclude_locked=True)

    try:
        start_time = time.time()
        opt_graph, final_params, final_score, _ = parameters_optimize(
            graph, method='CMA', verbose=False
        )
        elapsed = time.time() - start_time
        print(f"   [成功] Topology {idx + 1} 耗时: {elapsed:.1f}s | Cost: {final_score:.6f}")

        # 【修改这里】：不要返回 'graph': opt_graph，而是返回纯数据 'params': final_params
        return {'id': idx + 1, 'matrix': A, 'score': final_score, 'params': final_params}

    except Exception as e:
        print(f"   [失败] Topology {idx + 1} 报错: {e}")
        return None

# ==========================================
# 4. 批量优化主循环与归档
# ==========================================
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])
    io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io.init_save_dir(sub_path=None, unique_id=True)

    N_MZM, N_DET = 3, 4
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"Opt_Results_{N_MZM}MZM_{N_DET}DET_{run_timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n[工作流] 本次实验的所有结果将保存在独立文件夹: {run_dir}")

    input_freq, output_freq = 10.0e9, 2.5e9
    pulse_width, peak_power = (3e-12, 1.0)
    propagator = Propagator(window_t=4e-9, n_samples=2 ** 13, central_wl=1.55e-6)

    input_laser = PulsedLaser(parameters_from_name={
        'pulse_width': pulse_width, 'peak_power': peak_power,
        't_rep': 1.0 / input_freq, 'pulse_shape': 'gaussian',
        'central_wl': 1.55e-6, 'train': True
    })
    input_laser.node_lock = True

    targets = {};
    target_ids_map = {};
    eval_ids = []
    target_shifts = [0.00, 0.50, 0.25, 0.75]

    for i in range(N_DET):
        nid = 1 + N_MZM + i
        eval_ids.append(nid)
        targets[nid] = input_laser.get_pulse_train(
            propagator.t, pulse_width, 1.0 / output_freq, peak_power / 4, phase_shift=target_shifts[i]
        )
        target_ids_map[nid] = targets[nid]

    # evaluator = Local_XCorrEvaluator(propagator, targets=target_ids_map, evaluation_nodes=eval_ids)
    evaluator = Local_StrictEvaluator(propagator, targets=target_ids_map, evaluation_nodes=eval_ids)

    npy_file = f"topologies_{N_MZM}MZM_{N_DET}DET_unique_all.npy"
    if not os.path.exists(npy_file):
        raise FileNotFoundError(f"找不到拓扑文件 {npy_file}！")

    all_topologies = np.load(npy_file)
    n_topos = len(all_topologies)
    print(f"\n[Info] 加载了 {n_topos} 个矩阵。即将使用 CMA 算法进行物理参数全空间搜索...")

    MAX_TO_OPTIMIZE = n_topos
    ################# 增加使用的CPU核心数 ####################
    tasks = [(idx, A) for idx, A in enumerate(all_topologies[:MAX_TO_OPTIMIZE])]

    # 获取当前服务器的 CPU 核心数，留 2 个给系统，防止卡死
    cpu_cores = max(1, os.cpu_count() - 2)
    # 如果你租了那台 32 核的机器，这里会自动变成 30
    print(f"\n🚀 启动多进程加速，分配了 {cpu_cores} 个 CPU 核心同时工作！")

    leaderboard = []

    # 使用 partial 固定住不变的参数
    worker_func = partial(optimize_single_topology,
                          input_laser_tpl=input_laser,
                          n_mzm=N_MZM, n_det=N_DET,
                          prop=propagator, eval_obj=evaluator)

    # 开启进程池
    with mp.Pool(processes=cpu_cores) as pool:
        # map 函数会自动把 100 个任务分发给这些核心
        results = pool.map(worker_func, tasks)

    # 收集成功的结果
    for res in results:
        if res is not None:
            leaderboard.append(res)


    # leaderboard = []
    #
    # for idx, A in enumerate(all_topologies[:MAX_TO_OPTIMIZE]):
    #     print(f"\n>> 正在优化 Topology {idx + 1} / {MAX_TO_OPTIMIZE}")
    #     graph = build_graph_from_matrix(A, input_laser, N_MZM, N_DET)
    #     graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)
    #
    #     try:
    #         start_time = time.time()
    #         opt_graph, final_params, final_score, _ = parameters_optimize(
    #             graph, method='CMA',
    #             n_generations=20, population_size=20,
    #             verbose=False
    #         )
    #         elapsed = time.time() - start_time
    #         print(f"   [完成] 耗时: {elapsed:.1f}s | 最终 Cost: {final_score:.6f}")
    #         leaderboard.append({'id': idx + 1, 'matrix': A, 'score': final_score, 'graph': opt_graph})
    #     except Exception as e:
    #         print(f"   [失败] 拓扑 {idx + 1} 报错: {e}")
    #
    # if not leaderboard: sys.exit("所有优化均失败。")

    leaderboard.sort(key=lambda x: x['score'])

    txt_path = os.path.join(run_dir, "leaderboard_scores.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Optimization Run: {run_timestamp}\n")
        f.write("=========================================\n")
        for i, res in enumerate(leaderboard):
            f.write(f"Rank {i + 1:03d} | Topology ID: {res['id']:03d} | Cost: {res['score']:.6f}\n")
    print(f"\n[数据保存] 完整的排行榜得分已保存至: {txt_path}")

    # ==========================================
    # 5. 可视化阶段 I：展示前 100 个最优个体的“分页全家福”
    # ==========================================
    # 【核心调整】：将上限扩展到 100
    top_k = min(100, len(leaderboard))
    print(f"\n=========================================")
    print(f"🏆 准备生成 Top {top_k} 的可视化数据...")

    n_per_page = 10
    num_pages = math.ceil(top_k / n_per_page)

    first_page_fig = None  # 用于记录第一页，最后展示在屏幕上

    for page in range(num_pages):
        fig_top, axes_top = plt.subplots(2, 5, figsize=(25, 10))
        axes_top = axes_top.flatten()

        for i in range(n_per_page):
            idx = page * n_per_page + i
            ax = axes_top[i]
            if idx < top_k:
                res = leaderboard[idx]
                title = f"Rank {idx + 1} (ID: {res['id']})\nCost: {res['score']:.4f}"
                visualize_topology(ax, res['matrix'], N_MZM, N_DET, title=title)
            else:
                ax.axis('off')

        start_rank = page * 10 + 1
        end_rank = min((page + 1) * 10, top_k)
        fig_top.suptitle(f"Top {top_k} Best Topologies Discovered (Rank {start_rank}-{end_rank})", fontsize=18,
                         fontweight='bold')
        plt.tight_layout()

        fig_top_path = os.path.join(run_dir, f"Overview_Page_{page + 1:02d}.png")
        plt.savefig(fig_top_path, dpi=150, bbox_inches='tight')

        if page == 0:
            first_page_fig = fig_top  # 留住第一页
        else:
            plt.close(fig_top)  # 释放后面的页的内存

    # ==========================================
    # 6. 可视化阶段 II：批量详细呈现 Top 100 个体的 [拓扑 + 波形] 对比图
    # ==========================================
    save_dir_top = os.path.join(run_dir, "Detailed_Waveforms_Top100")
    os.makedirs(save_dir_top, exist_ok=True)

    print(f"\n正在生成 Top {top_k} 的详细波形与拓扑对应图表 (共 {top_k} 张，后台静默保存中)...")

    for rank in range(top_k):
        res = leaderboard[rank]

        # 【修改这里】：把下面这两行删掉
        # opt_graph = res['graph']
        # opt_graph.propagate(propagator)

        # 【替换为下面这四行】：在主进程中秒速重建模型，并塞入优化好的参数
        opt_graph = build_graph_from_matrix(res['matrix'], input_laser, N_MZM, N_DET)
        opt_graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)
        _, models, param_idx, _, _ = opt_graph.extract_parameters_to_list()
        opt_graph.distribute_parameters_from_list(res['params'], models, param_idx)
        opt_graph.propagate(propagator)

        fig_detail = plt.figure(figsize=(16, 10))
        gs = fig_detail.add_gridspec(4, 2, width_ratios=[1, 1.2])

        ax_topo = fig_detail.add_subplot(gs[:, 0])
        title_str = f"Rank {rank + 1} Topology (ID: {res['id']})\nFinal Cost: {res['score']:.4f}"
        if rank == 0: title_str = "🥇 Global Best " + title_str
        visualize_topology(ax_topo, res['matrix'], N_MZM, N_DET, title=title_str)

        # ==========================================
        # 【新增】：提取参数并在拓扑图的左下角绘制参数信息框
        # ==========================================
        param_str_lines = ["【Optimized Parameters】"]

        # 1. 提取 MZM 节点参数 (忽略锁定的输入输出节点)
        for nid, node in opt_graph.nodes.items():
            if hasattr(node, 'node_acronym') and 'MZM' in node.node_acronym:
                p = node.parameters
                # 根据你的 Local_MZM_Modulated 定义，p的顺序为:
                # [v_pi, v_bias, v_rf, f_rf, phase_rf, loss]
                # 这里我们展示几个会参与优化的关键参数
                param_str_lines.append(
                    f"{node.node_acronym}: V_bias={p[1]:.2f}V, V_rf={p[2]:.2f}V, Phase={p[4]:.2f}rad"
                )

        # 2. 提取边上的延迟线(Delay Line)参数
        for eid, edge in opt_graph.edges.items():
            if hasattr(edge, 'parameters') and len(edge.parameters) > 0:
                delay_ps = edge.parameters[0] * 1e12  # 转换为皮秒方便阅读
                if delay_ps > 0.001:  # 只显示有实际延迟的边
                    param_str_lines.append(f"Delay {eid}: {delay_ps:.2f} ps")

        # 3. 将文本拼合并放置在图表上
        param_text = "\n".join(param_str_lines)

        # 使用 transAxes 将坐标定位在子图的左下角 (x=0.02, y=0.02)
        # bbox 用于给文字加一个半透明的背景框，防止被拓扑图的连线挡住
        ax_topo.text(0.02, 0.02, param_text, transform=ax_topo.transAxes,
                     fontsize=10, verticalalignment='bottom', horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
        # ==========================================

        center_t = propagator.t[propagator.n_samples // 2]
        t_span = 800e-12
        labels = ["Ch1 (0.00)", "Ch2 (0.50)", "Ch3 (0.25)", "Ch4 (0.75)"]
        axes_wave = []

        # # 结果归一化展示
        # for i, nid in enumerate(eval_ids):
        #     ax = fig_detail.add_subplot(gs[i, 1], sharex=axes_wave[0] if i > 0 else None)
        #     axes_wave.append(ax)
        #
        #     out = opt_graph.measure_propagator(nid)
        #     if out is None: out = np.zeros_like(propagator.t)
        #
        #     p_out = power_(out)
        #     p_tgt = power_(targets[nid])
        #     p_out /= (np.max(p_out) + 1e-12)
        #     p_tgt /= (np.max(p_tgt) + 1e-12)
        #
        #     ax.plot(propagator.t * 1e12, p_out, 'r-', linewidth=2.0, label='Demuxed')
        #     ax.plot(propagator.t * 1e12, p_tgt, 'k--', alpha=0.4, label='Target')
        #     ax.set_ylabel('Norm Power')
        #     ax.set_title(f'{labels[i]} - Node CH_{i + 1}')
        #     ax.legend(loc='upper right')
        #     ax.set_xlim((center_t - t_span) * 1e12, (center_t + t_span) * 1e12)
        #     ax.grid(True, alpha=0.3)
        #     if i < 3: ax.tick_params(labelbottom=False)

        # 结果不归一化展示
        for i, nid in enumerate(eval_ids):
            ax = fig_detail.add_subplot(gs[i, 1], sharex=axes_wave[0] if i > 0 else None)
            axes_wave.append(ax)

            out = opt_graph.measure_propagator(nid)
            if out is None: out = np.zeros_like(propagator.t)

            p_out = power_(out)
            p_tgt = power_(targets[nid])

            # 【修改点 1】：注释或删除下面这两行归一化代码
            # p_out /= (np.max(p_out) + 1e-12)
            # p_tgt /= (np.max(p_tgt) + 1e-12)

            ax.plot(propagator.t * 1e12, p_out, 'r-', linewidth=2.0, label='Demuxed')
            ax.plot(propagator.t * 1e12, p_tgt, 'k--', alpha=0.4, label='Target')

            # 【修改点 2】：更新 Y 轴标签，因为现在不是归一化的功率了
            ax.set_ylabel('Power (W)')  # 假设你的单位是瓦特，或者改为 'Absolute Power'

            ax.set_title(f'{labels[i]} - Node CH_{i + 1}')
            ax.legend(loc='upper right')
            ax.set_xlim((center_t - t_span) * 1e12, (center_t + t_span) * 1e12)
            ax.grid(True, alpha=0.3)
            if i < 3: ax.tick_params(labelbottom=False)

        axes_wave[-1].set_xlabel('Time (ps)')
        plt.tight_layout()

        file_path = os.path.join(save_dir_top, f"Rank_{rank + 1:03d}_TopoID_{res['id']:03d}.png")
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close(fig_detail)  # 必须写在这里释放内存，否则 100 张高清图会吃光 RAM

        # 每 10 张汇报一下进度
        if (rank + 1) % 10 == 0:
            print(f"  -> 已完成 {rank + 1} / {top_k} 张图片的渲染和保存...")

    print(f"\n【图表保存】大功告成！Top {top_k} 的 100 张详细分析图均已安全存入 '{save_dir_top}' 文件夹中！")

    # 屏幕上只留一张 Rank 1-10 的概览图作为提示，不卡电脑
    if first_page_fig:
        plt.show()

    # ==========================================
    # 提取并保存 Top 100 个体的拓扑与参数
    # ==========================================
    top_k = min(100, len(leaderboard))
    top_100_export_data = []

    for rank in range(top_k):
        res = leaderboard[rank]

        opt_graph = build_graph_from_matrix(res['matrix'], input_laser, N_MZM, N_DET)

        # 【同样加上这行！】
        opt_graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

        _, models, param_idx, _, _ = opt_graph.extract_parameters_to_list()
        opt_graph.distribute_parameters_from_list(res['params'], models, param_idx)

        # ====== DEBUG 拦截开始 ======
        if rank == 0:
            print(f"\n[DEBUG] 正在检查 Rank 1 (ID: {res['id']}) 的参数提取情况:")
            print(f"--- 原始参数列表长度: {len(res['params'])}")
            print(f"--- 原始参数内容: {res['params'][:3]} ...")  # 只印前三个
        # ====== DEBUG 拦截结束 ======

        # ==========================================
        # 提取参数 (防弹版：无视底层封装字典，自动寻址)
        # ==========================================

        # 1. 提取节点参数
        node_params = {}
        for nid, node_data in opt_graph.nodes.items():
            node_obj = None
            if isinstance(node_data, dict):
                # 遍历字典里的所有内容，把真正的物理节点揪出来
                for key, val in node_data.items():
                    if hasattr(val, 'parameters'):
                        node_obj = val
                        break
            else:
                node_obj = node_data

            if node_obj and hasattr(node_obj, 'parameters'):
                node_params[nid] = copy.deepcopy(node_obj.parameters)

        # 2. 提取边参数
        edge_params = {}
        for eid, edge_data in opt_graph.edges.items():
            edge_obj = None
            if isinstance(edge_data, dict):
                for key, val in edge_data.items():
                    if hasattr(val, 'parameters'):
                        edge_obj = val
                        break
            else:
                edge_obj = edge_data

            if edge_obj and hasattr(edge_obj, 'parameters'):
                edge_params[eid] = copy.deepcopy(edge_obj.parameters)

        # 3. 打包导出
        top_100_export_data.append({
            'rank': rank + 1,
            'id': res['id'],
            'score': res['score'],
            'matrix': res['matrix'],
            'raw_params': res['params'],  # 物理数组原片，最稳妥的备份
            'node_params': node_params,
            'edge_params': edge_params
        })

    # 4. 保存为 pkl 文件
    export_file_path = os.path.join(run_dir, f"Top{top_k}_Optimized_Params.pkl")
    with open(export_file_path, 'wb') as f:
        pickle.dump(top_100_export_data, f)
    print(f"\n[参数归档] Top {top_k} 的拓扑矩阵与参数已安全导出至: {export_file_path}")