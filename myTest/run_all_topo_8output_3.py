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
import math
import pickle
import multiprocessing as mp
from functools import partial
import matplotlib.gridspec as gridspec

# 【新增引入】用于最优目标匹配，解决"窜位"问题
from scipy.optimize import linear_sum_assignment

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


def draw_edge(ax, p1, p2, bend=0.0, style='solid', alpha=0.6, color="black", lw=1.1):
    x1, y1 = p1
    x2, y2 = p2
    ctrl_x1 = x1 + (x2 - x1) * 0.4
    ctrl_y1 = y1 + bend
    ctrl_x2 = x2 - (x2 - x1) * 0.4
    ctrl_y2 = y2 + bend
    path = Path([(x1, y1), (ctrl_x1, ctrl_y1), (ctrl_x2, ctrl_y2), (x2, y2)],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    arrow = FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=12,
                            color=color, lw=lw, alpha=alpha, linestyle=style, zorder=1)
    ax.add_patch(arrow)


def visualize_topology(ax, A, N_mzm, N_det, title):
    levels = compute_levels(A.copy())
    layers = defaultdict(list)
    for i, lv in enumerate(levels): layers[lv].append(i)

    pos = {};
    port_pos = {}
    sorted_levels = sorted(layers.keys())
    V_SPACING, H_SPACING = 3.5, 6.0
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
                    draw_edge(ax, p1, p2, bend=bend, style='--', alpha=0.4, color="gray", lw=1.0)
                else:
                    bend = 0.6 * (((edge_idx % 5) / 2.0) - 1.0)
                    draw_edge(ax, p1, p2, bend=bend, style='-', alpha=0.8, color="black", lw=1.2)
                edge_idx += 1

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis("off")
    ax.autoscale_view()
    ax.margins(0.1)


# ==========================================
# 2. 本地组件
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
            return np.zeros_like(propagator.t)
        elif isinstance(state, (list, tuple)):
            state = state[0] if len(state) > 0 and state[0] is not None else np.zeros_like(propagator.t)
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
        # 频率默认值设为 1.25GHz
        self.default_parameters = [3.5, 1.75, 1.75, 1.25e9, 0.0, 0.1]
        self.upper_bounds = [10.0, 10.0, 7.0, 50.0e9, 2 * np.pi, 1.0]
        self.lower_bounds = [1.0, -10.0, 0.0, 0.0, -2 * np.pi, 0.0]
        self.data_types = ['float'] * 6
        self.step_sizes = [0.5, 0.5, 0.5, 1.0e9, 0.1, 0.01]
        self.parameter_imprecisions = [0.01] * 6
        self.parameter_units = ['V', 'V', 'V', 'Hz', 'rad', 'dB']

        # 【核心修改】：将第四个元素 (f_rf) 的锁改为 False，允许算法自由优化频率
        self.parameter_locks = [True, False, False, False, False, True]

        self.parameter_symbols = [r"$V_{\pi}$", r"$V_{bias}$", r"$V_{RF}$", r"$f_{RF}$", r"$\theta_{RF}$", "IL"]
        super().__init__(**kwargs)
        self._range_input_edges = (1, 50)
        self.num_inputs = 2
        self.num_outputs = 2

    def update_attributes(self, num_inputs, num_outputs):
        self.num_inputs = max(2, num_inputs)
        self.num_outputs = max(2, num_outputs)
        if not hasattr(self, 'parameters') or len(self.parameters) != 6:
            self.parameters = self.default_parameters

    def propagate(self, states, propagator, num_inputs, num_outputs, save_transforms=False):
        v_pi, v_bias, v_rf, f_rf, phase_rf, loss = self.parameters
        t_len = len(np.squeeze(propagator.t))
        if states is None:
            states = []
        elif not isinstance(states, (list, tuple)):
            states = [states]

        top_states = [];
        bot_states = []
        for k, s in enumerate(states):
            signal = np.zeros(t_len, dtype=complex) if s is None else np.squeeze(s)
            phys_port = self.port_mapping[k] if hasattr(self, 'port_mapping') and k < len(self.port_mapping) else (
                    k % 2)
            if phys_port == 0:
                top_states.append(signal)
            else:
                bot_states.append(signal)

        in_top = sum(top_states) if top_states else np.zeros(t_len, dtype=complex)
        in_bot = sum(bot_states) if bot_states else np.zeros(t_len, dtype=complex)

        t = np.squeeze(propagator.t)
        rf_signal = v_rf * np.cos(2 * np.pi * f_rf * t + phase_rf)
        phi_t = (np.pi * (v_bias + rf_signal)) / (2.0 * v_pi)
        cos_phi = np.cos(phi_t)
        sin_phi = np.sin(phi_t)

        factor = (1.0 - loss)
        out1 = (in_top * cos_phi - 1j * in_bot * sin_phi) * factor
        out2 = (-1j * in_top * sin_phi + in_bot * cos_phi) * factor

        result = [out1, out2]
        while len(result) < getattr(self, 'num_outputs', 2):
            result.append(np.zeros_like(out1))
        return result


class Local_TerminalSource(TerminalSource):
    def update_attributes(self, num_inputs, num_outputs): self.num_outputs = max(1, num_outputs)

    def propagate(self, states, propagator, num_inputs=0, num_outputs=1, save_transforms=False):
        base_state = states[0] if isinstance(states, (list, tuple)) and len(states) > 0 else states
        return [base_state] * getattr(self, 'num_outputs', 1)


class Local_TerminalSink(TerminalSink):
    def update_attributes(self, num_inputs, num_outputs):
        self.num_inputs = max(1, num_inputs)
        self.num_outputs = max(1, num_outputs)

    def propagate(self, states, propagator, num_inputs=1, num_outputs=0, save_transforms=False):
        if isinstance(states, (list, tuple)):
            valid_states = [s for s in states if s is not None]
            self.state = sum(valid_states) if valid_states else np.zeros_like(propagator.t)
        else:
            self.state = states if states is not None else np.zeros_like(propagator.t)
        return [self.state] * getattr(self, 'num_outputs', 1)


class PulsedLaser_Sine(PulsedLaser):
    def __init__(self, envelope_period=1.0 / 1.25e9, **kwargs):
        super().__init__(**kwargs)
        self.envelope_period = envelope_period

    def get_pulse_train(self, t, pulse_width, rep_t, peak_power, pulse_shape='gaussian', phase_shift=0):
        phase = (t % self.envelope_period) / self.envelope_period
        envelope = 0.55 - 0.45 * np.cos(2 * np.pi * phase)
        dynamic_peak_power = peak_power * envelope
        shifted_t = t - phase_shift * rep_t
        wrapped_t = np.sin(np.pi * shifted_t / rep_t)
        unwrapped_t = np.arcsin(wrapped_t) * rep_t / np.pi
        pulse_width = pulse_width / (2 * np.sqrt(np.log(2)))
        pulse = self.gaussian(unwrapped_t, pulse_width) if pulse_shape == 'gaussian' else self.sech(unwrapped_t,
                                                                                                    pulse_width)
        return pulse * np.sqrt(np.abs(dynamic_peak_power))


# ==========================================
# 【核心修改区】: 动态连通性与弹性评价器
# ==========================================
class Local_FlexibleEvaluator(Evaluator):
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
            self.masks[nid] = np.squeeze((norm_tgt < 0.2).astype(float))    # TODO: 掩码阈值

    def evaluate_graph(self, graph, propagator=None):
        return self.evaluate(graph)

    def evaluate(self, graph):
        n_nodes = len(self.evaluation_nodes)
        cost_matrix = np.zeros((n_nodes, n_nodes))

        signals_norm = []
        for node in self.evaluation_nodes:
            signal = graph.measure_propagator(node)
            if isinstance(signal, (list, tuple)):
                signal = signal[0] if len(signal) > 0 and signal[0] is not None else None

            if signal is None: return 5000.0 * n_nodes  # 惩罚断路

            p_signal = np.abs(signal) ** 2
            max_sig = np.max(p_signal) + 1e-15
            signals_norm.append(np.squeeze(p_signal / max_sig))

        target_nids = list(self.targets.keys())

        for i, sig_norm in enumerate(signals_norm):
            for j, tgt_nid in enumerate(target_nids):
                norm_tgt = self.target_norms[tgt_nid]
                mask = self.masks[tgt_nid]

                shape_error = np.mean((sig_norm - norm_tgt) ** 2)
                noise_content = sig_norm * mask
                penalty = np.mean(noise_content ** 2) * 200.0

                cost_matrix[i, j] = shape_error + penalty

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_cost = cost_matrix[row_ind, col_ind].sum()

        return total_cost

######################## 更严格的评估函数 ##########################
    # def evaluate(self, graph):
    #     n_nodes = len(self.evaluation_nodes)
    #     cost_matrix = np.zeros((n_nodes, n_nodes))
    #
    #     signals_norm = []
    #     raw_peaks = []  # 【新增】记录每个通道的绝对峰值功率
    #
    #     for node in self.evaluation_nodes:
    #         signal = graph.measure_propagator(node)
    #         if isinstance(signal, (list, tuple)):
    #             signal = signal[0] if len(signal) > 0 and signal[0] is not None else None
    #
    #         if signal is None: return 5000.0 * n_nodes
    #
    #         p_signal = np.abs(signal) ** 2
    #         max_sig = np.max(p_signal) + 1e-15
    #         raw_peaks.append(max_sig)  # 【新增】记录峰值
    #         signals_norm.append(np.squeeze(p_signal / max_sig))
    #
    #     target_nids = list(self.targets.keys())
    #
    #     for i, sig_norm in enumerate(signals_norm):
    #         for j, tgt_nid in enumerate(target_nids):
    #             norm_tgt = self.target_norms[tgt_nid]
    #             mask = self.masks[tgt_nid]
    #
    #             shape_error = np.mean((sig_norm - norm_tgt) ** 2)
    #             noise_content = sig_norm * mask
    #             # 【修改】将权重从 20 提升到 100，极致压制底噪
    #             penalty = np.mean(noise_content ** 2) * 100.0
    #
    #             cost_matrix[i, j] = shape_error + penalty
    #
    #     row_ind, col_ind = linear_sum_assignment(cost_matrix)
    #     total_cost = cost_matrix[row_ind, col_ind].sum()
    #
    #     # 【新增】计算分配成功的 8 个通道的功率均衡度
    #     # 如果方差极大，说明各通道高低不平，施加高额惩罚
    #     power_variance = np.var(raw_peaks) / (np.mean(raw_peaks) ** 2 + 1e-15)
    #     uniformity_penalty = power_variance * 10.0  # 权重可调
    #
    #     return total_cost + uniformity_penalty


# ==========================================
# 3. 动态拓扑图构建引擎
# ==========================================
def build_graph_from_matrix(A, input_laser_template, N_mzm, N_det):
    nodes = {};
    edges = {}
    n = A.shape[0]

    nodes[0] = Local_TerminalSource()
    nodes[0].node_acronym = 'SRC'

    for i in range(1, 1 + N_mzm):
        mzm = Local_MZM_Modulated()
        mzm.node_acronym = f'MZM_{i}'
        mzm.port_mapping = []
        # 【核心修改】：彻底移除了分层频率预设的硬编码逻辑
        mzm.parameters = copy.deepcopy(mzm.default_parameters)  # 直接加载默认的 1.25GHz
        nodes[i] = mzm

    for i in range(1 + N_mzm, n):
        ch_idx = i - N_mzm
        nodes[i] = Local_TerminalSink(node_name=f'sink_{ch_idx}')
        nodes[i].node_acronym = f'CH_{ch_idx}'

    port_idx_map = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}
    current_in_port = {i: 0 for i in range(n)};
    current_out_port = {i: 0 for i in range(n)}

    for u in range(n):
        for v in range(n):
            if A[u, v] > 0:
                out_p, original_in_p = port_idx_map[A[u, v]]
                safe_in_p = current_in_port[v]
                current_in_port[v] += 1

                if v >= 1 and v <= N_mzm: nodes[v].port_mapping.append(original_in_p)

                if u == 0:
                    edge_obj = copy.deepcopy(input_laser_template)
                    edge_obj.src_port_idx = current_out_port[u]
                    current_out_port[u] += 1
                    edge_obj.dst_port_idx = safe_in_p
                    edges[(u, v)] = edge_obj
                else:
                    dl = Local_SafeDelayLine()
                    dl.parameters = [0.0];
                    dl.node_lock = True
                    dl.src_port_idx = out_p;
                    dl.dst_port_idx = safe_in_p
                    edges[(u, v)] = dl

    nodes[0].num_outputs = max(1, current_out_port[0])
    for i in range(1, 1 + N_mzm): nodes[i].num_inputs = max(2, current_in_port[i])
    for i in range(1 + N_mzm, n): nodes[i].num_inputs = max(1, current_in_port[i])

    return Graph.init_graph(nodes, edges)


def optimize_single_topology(idx_and_matrix, input_laser_tpl, n_mzm, n_det, prop, eval_obj):
    idx, A = idx_and_matrix
    print(f"[Core {mp.current_process().name}] 开始优化 Topology {idx + 1}")
    graph = build_graph_from_matrix(A, input_laser_tpl, n_mzm, n_det)
    graph.initialize_func_grad_hess(prop, eval_obj, exclude_locked=True)
    try:
        start_time = time.time()
        opt_graph, final_params, final_score, _ = parameters_optimize(
            graph, method='CMA', maxfevals=30000, popsize=100, verbose=False
        )
        elapsed = time.time() - start_time
        print(f"   [成功] Topology {idx + 1} 耗时: {elapsed:.1f}s | Cost: {final_score:.6f}")
        return {'id': idx + 1, 'matrix': A, 'score': final_score, 'params': final_params}
    except Exception as e:
        print(f"   [失败] Topology {idx + 1} 报错: {e}")
        return None


# ==========================================
# 主循环与归档
# ==========================================
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])
    io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io.init_save_dir(sub_path=None, unique_id=True)

    N_MZM, N_DET = 7, 8
    DATA_BASE_DIR = "/root/autodl-tmp/aesop/asope_data"

    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(DATA_BASE_DIR, f"Opt_Results_UnlockedFreq_{N_MZM}MZM_{N_DET}DET_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n[工作流] 本次实验（频率自由优化版）的所有结果将保存在独立文件夹: {run_dir}")

    input_freq, output_freq = 10.0e9, 1.25e9
    pulse_width, peak_power = (3e-12, 1.0)
    propagator = Propagator(window_t=4e-9, n_samples=2 ** 13, central_wl=1.55e-6)

    input_laser = PulsedLaser_Sine(
        envelope_period=1.0 / output_freq,
        parameters_from_name={
            'pulse_width': pulse_width, 'peak_power': peak_power,
            't_rep': 1.0 / input_freq, 'pulse_shape': 'gaussian',
            'central_wl': 1.55e-6, 'train': True
        }
    )
    input_laser.node_lock = True

    targets = {};
    target_ids_map = {};
    eval_ids = []
    target_shifts = [0.0, 4.0 / 8.0, 2.0 / 8.0, 6.0 / 8.0, 1.0 / 8.0, 5.0 / 8.0, 3.0 / 8.0, 7.0 / 8.0]

    for i in range(N_DET):
        nid = 1 + N_MZM + i
        eval_ids.append(nid)
        targets[nid] = input_laser.get_pulse_train(
            propagator.t, pulse_width, 1.0 / output_freq, peak_power / 8, phase_shift=target_shifts[i]
        )
        target_ids_map[nid] = targets[nid]

    evaluator = Local_FlexibleEvaluator(propagator, targets=target_ids_map, evaluation_nodes=eval_ids)

    npy_file = os.path.join(DATA_BASE_DIR, f"topologies_{N_MZM}MZM_{N_DET}DET_unique_all.npy")
    if not os.path.exists(npy_file): raise FileNotFoundError(f"找不到拓扑文件 {npy_file}！")

    all_topologies = np.load(npy_file)
    n_topos = len(all_topologies)

    # 提醒：根据需要将 MAX_TO_OPTIMIZE 改回 n_topos 以运行全量拓扑
    MAX_TO_OPTIMIZE = n_topos

    tasks = [(idx, A) for idx, A in enumerate(all_topologies[:MAX_TO_OPTIMIZE])]

    cpu_cores = max(1, os.cpu_count() - 2)
    print(f"\n🚀 启动多进程加速，分配了 {cpu_cores} 个 CPU 核心同时工作！")

    leaderboard = []
    worker_func = partial(optimize_single_topology, input_laser_tpl=input_laser,
                          n_mzm=N_MZM, n_det=N_DET, prop=propagator, eval_obj=evaluator)

    with mp.Pool(processes=cpu_cores) as pool:
        results = pool.map(worker_func, tasks)

    for res in results:
        if res is not None: leaderboard.append(res)

    if not leaderboard: sys.exit("所有优化均失败。")
    leaderboard.sort(key=lambda x: x['score'])

    txt_path = os.path.join(run_dir, "leaderboard_scores.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Optimization Run (Unlocked Frequency): {run_timestamp}\n")
        f.write("=========================================\n")
        for i, res in enumerate(leaderboard):
            f.write(f"Rank {i + 1:03d} | Topology ID: {res['id']:03d} | Cost: {res['score']:.6f}\n")

    # ==========================================
    # 可视化渲染逻辑重构：仅展示波形图 (移除了频谱分析)
    # ==========================================
    top_k = min(100, len(leaderboard))

    # 波形图保存路径
    save_dir_top = os.path.join(run_dir, "Detailed_Waveforms_Top100")
    os.makedirs(save_dir_top, exist_ok=True)

    print(f"\n正在生成 Top {top_k} 的详细波形图表...")

    for rank in range(top_k):
        res = leaderboard[rank]
        opt_graph = build_graph_from_matrix(res['matrix'], input_laser, N_MZM, N_DET)
        opt_graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)
        _, models, param_idx, _, _ = opt_graph.extract_parameters_to_list()
        opt_graph.distribute_parameters_from_list(res['params'], models, param_idx)
        opt_graph.propagate(propagator)

        # 时域波形图计算与渲染
        cost_matrix = np.zeros((N_DET, N_DET))
        signals_norm = []
        out_powers = []
        raw_signals = []

        for nid in eval_ids:
            tmp_out = opt_graph.measure_propagator(nid)
            p_tmp = power_(tmp_out) if tmp_out is not None else np.zeros_like(propagator.t)
            out_powers.append(np.max(p_tmp))
            raw_signals.append(p_tmp)
            max_sig = np.max(p_tmp) + 1e-15
            signals_norm.append(np.squeeze(p_tmp / max_sig))

        for i, sig_norm in enumerate(signals_norm):
            for j, tgt_nid in enumerate(eval_ids):
                norm_tgt = evaluator.target_norms[tgt_nid]
                mask = evaluator.masks[tgt_nid]
                cost_matrix[i, j] = np.mean((sig_norm - norm_tgt) ** 2) + np.mean((sig_norm * mask) ** 2) * 20.0

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        global_max_out = max(out_powers) if max(out_powers) > 0 else 1.0
        global_max_tgt = max([np.max(power_(targets[nid])) for nid in eval_ids])

        fig_detail = plt.figure(figsize=(18, 2 * N_DET))
        gs = fig_detail.add_gridspec(N_DET, 2, width_ratios=[1, 1.2])

        ax_topo = fig_detail.add_subplot(gs[:, 0])
        title_str = f"Rank {rank + 1} Topology (ID: {res['id']})\nFinal Cost: {res['score']:.4f}"
        if rank == 0: title_str = "🥇 Global Best " + title_str
        visualize_topology(ax_topo, res['matrix'], N_MZM, N_DET, title=title_str)

        param_str_lines = ["【Optimized Parameters】"]
        for nid, node_data in opt_graph.nodes.items():
            node_obj = list(node_data.values())[0] if isinstance(node_data, dict) else node_data
            if node_obj and hasattr(node_obj, 'node_acronym') and 'MZM' in node_obj.node_acronym:
                p = node_obj.parameters
                v_bias, v_rf, f_rf, phase = float(p[1]), float(p[2]), float(p[3]) / 1e9, float(p[4])
                param_str_lines.append(
                    f"{node_obj.node_acronym}: Vb={v_bias:.2f}V, Vr={v_rf:.2f}V, f={f_rf:.2f}GHz, Ph={phase:.2f}rad")

        for eid, edge_data in opt_graph.edges.items():
            edge_obj = list(edge_data.values())[0] if isinstance(edge_data, dict) else edge_data
            if edge_obj and hasattr(edge_obj, 'node_acronym') and edge_obj.node_acronym == 'DL':
                try:
                    delay_ps = float(np.array(edge_obj.parameters).flatten()[0]) * 1e12
                    if delay_ps > 0.001: param_str_lines.append(f"Delay {eid}: {delay_ps:.2f} ps")
                except:
                    pass

        param_text = "\n".join(param_str_lines)
        ax_topo.text(0.02, 0.02, param_text, transform=ax_topo.transAxes, fontsize=9,
                     verticalalignment='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

        center_t = propagator.t[propagator.n_samples // 2]
        t_span = 800e-12
        axes_wave = []

        for i, nid in enumerate(eval_ids):
            ax = fig_detail.add_subplot(gs[i, 1], sharex=axes_wave[0] if i > 0 else None)
            axes_wave.append(ax)

            p_out = raw_signals[i]
            matched_tgt_idx = col_ind[i]
            matched_tgt_nid = eval_ids[matched_tgt_idx]
            p_tgt = power_(targets[matched_tgt_nid])

            p_out_norm = p_out / (global_max_out + 1e-12)
            p_tgt_norm = p_tgt / (global_max_tgt + 1e-12)

            ax.plot(propagator.t * 1e12, p_out_norm, 'r-', linewidth=2.0, label='Demuxed Signal')
            ax.plot(propagator.t * 1e12, p_tgt_norm, 'b--', alpha=0.4, label='Matched Target')

            ax.set_ylabel('Norm. Power', fontsize=8)
            ax.set_title(f'Node CH_{i + 1} ➔ Locked to Phase: {target_shifts[matched_tgt_idx]:.3f}', fontsize=10,
                         fontweight='bold', color='darkblue')
            ax.legend(loc='upper right', fontsize=8)
            ax.set_xlim((center_t - t_span) * 1e12, (center_t + t_span) * 1e12)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.1)

            if i < N_DET - 1: ax.tick_params(labelbottom=False)

        axes_wave[-1].set_xlabel('Time (ps)')
        plt.tight_layout()

        file_path = os.path.join(save_dir_top, f"Rank_{rank + 1:03d}_TopoID_{res['id']:03d}.png")
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close(fig_detail)

    print(f"\n【图表保存】大功告成！详细波形图存入 '{save_dir_top}'！")

    # 导出 Pickel 数据
    top_100_export_data = []
    for rank in range(top_k):
        res = leaderboard[rank]
        top_100_export_data.append(
            {'rank': rank + 1, 'id': res['id'], 'score': res['score'], 'matrix': res['matrix'],
             'raw_params': res['params']})

    with open(os.path.join(run_dir, f"Top{top_k}_Optimized_Params.pkl"), 'wb') as f:
        pickle.dump(top_100_export_data, f)