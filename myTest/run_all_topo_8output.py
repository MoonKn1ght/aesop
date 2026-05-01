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
# 2. 本地组件 (加入光场叠加的终极物理版)
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
        # 【保持 8 通道要求的 1.25 GHz 默认设置】
        self.default_parameters = [3.5, 1.75, 1.0, 1.25e9, 0.0, 0.1]
        self.upper_bounds = [10.0, 10.0, 20.0, 50.0e9, 2 * np.pi, 1.0]
        self.lower_bounds = [1.0, -10.0, 0.0, 0.0, -2 * np.pi, 0.0]
        self.data_types = ['float'] * 6
        self.step_sizes = [0.5, 0.5, 0.5, 1.0e9, 0.1, 0.01]
        self.parameter_imprecisions = [0.01] * 6
        self.parameter_units = ['V'] * 3 + ['Hz', 'rad', 'dB']
        # 【严格锁定 v_pi, f_rf, loss】
        self.parameter_locks = [True, False, False, True, False, True]
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

        top_states = []
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
        target_len = getattr(self, 'num_outputs', 2)
        while len(result) < target_len:
            result.append(np.zeros_like(out1))
        return result


class Local_TerminalSource(TerminalSource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._range_output_edges = (1, 50)

    def update_attributes(self, num_inputs, num_outputs):
        self.num_outputs = max(1, num_outputs)

    def propagate(self, states, propagator, num_inputs=0, num_outputs=1, save_transforms=False):
        base_state = states[0] if isinstance(states, (list, tuple)) and len(states) > 0 else states
        target_len = getattr(self, 'num_outputs', 1)
        return [base_state] * target_len


class Local_TerminalSink(TerminalSink):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._range_input_edges = (1, 50)

    def update_attributes(self, num_inputs, num_outputs):
        self.num_inputs = max(1, num_inputs)
        self.num_outputs = max(1, num_outputs)

    def propagate(self, states, propagator, num_inputs=1, num_outputs=0, save_transforms=False):
        if isinstance(states, (list, tuple)):
            valid_states = [s for s in states if s is not None]
            self.state = sum(valid_states) if valid_states else np.zeros_like(propagator.t)
        else:
            self.state = states if states is not None else np.zeros_like(propagator.t)

        target_len = getattr(self, 'num_outputs', 1)
        return [self.state] * target_len


class PulsedLaser_Sawtooth(PulsedLaser):
    def __init__(self, envelope_period=1.0 / 1.25e9, **kwargs):
        """
        锯齿波/线性爬坡包络激光器。
        envelope_period: 包络周期。为了让 8 个通道严格递增，周期必须等于解复用频率 (1.25GHz)
        """
        super().__init__(**kwargs)
        self.envelope_period = envelope_period

    def get_pulse_train(self, t, pulse_width, rep_t, peak_power, pulse_shape='gaussian', phase_shift=0):
        # 【核心逻辑：锯齿波爬坡】
        # 计算当前时间在周期内的相位 (0.0 到 1.0 之间)
        phase = (t % self.envelope_period) / self.envelope_period

        # 让包络幅值严格从 0.1 线性增长到 1.0
        envelope = 0.1 + 0.9 * phase

        # 将动态包络赋予峰值功率
        dynamic_peak_power = peak_power * envelope

        # 下面完全保留你的原版底层物理代码
        shifted_t = t - phase_shift * rep_t
        wrapped_t = np.sin(np.pi * shifted_t / rep_t)
        unwrapped_t = np.arcsin(wrapped_t) * rep_t / np.pi
        pulse_width = pulse_width / (2 * np.sqrt(np.log(2)))

        if pulse_shape == 'gaussian':
            pulse = self.gaussian(unwrapped_t, pulse_width)
        elif pulse_shape == 'sech':
            pulse = self.sech(unwrapped_t, pulse_width)
        elif pulse_shape == 'delta':
            import scipy.signal as sig
            pulse = sig.unit_impulse(shape=t.shape)
        else:
            raise RuntimeError(f"Pulsed Laser: {pulse_shape} is not a defined pulse shape")

        # dynamic_peak_power 现在是一个时间数组，实现脉冲高度的逐级递增
        state = pulse * np.sqrt(np.abs(dynamic_peak_power))
        return state


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

            # 安全提取信号
            if isinstance(signal, (list, tuple)):
                signal = signal[0] if len(signal) > 0 and signal[0] is not None else None

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
        nodes[i] = mzm

    for i in range(1 + N_mzm, n):
        ch_idx = i - N_mzm
        nodes[i] = Local_TerminalSink(node_name=f'sink_{ch_idx}')
        nodes[i].node_acronym = f'CH_{ch_idx}'

    port_idx_map = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}
    current_in_port = {i: 0 for i in range(n)}
    current_out_port = {i: 0 for i in range(n)}

    for u in range(n):
        for v in range(n):
            if A[u, v] > 0:
                out_p, original_in_p = port_idx_map[A[u, v]]
                safe_in_p = current_in_port[v]
                current_in_port[v] += 1

                if v >= 1 and v <= N_mzm:
                    nodes[v].port_mapping.append(original_in_p)

                if u == 0:
                    edge_obj = copy.deepcopy(input_laser_template)
                    edge_obj.src_port_idx = current_out_port[u]
                    current_out_port[u] += 1
                    edge_obj.dst_port_idx = safe_in_p
                    edges[(u, v)] = edge_obj
                else:
                    dl = Local_SafeDelayLine()
                    dl.parameters = [0.0]
                    dl.node_lock = True
                    dl.src_port_idx = out_p
                    dl.dst_port_idx = safe_in_p
                    edges[(u, v)] = dl

    nodes[0].num_outputs = max(1, current_out_port[0])
    for i in range(1, 1 + N_mzm):
        nodes[i].num_inputs = max(2, current_in_port[i])
    for i in range(1 + N_mzm, n):
        nodes[i].num_inputs = max(1, current_in_port[i])

    return Graph.init_graph(nodes, edges)


# ==========================================
# 4. 多进程计算打工函数
# ==========================================
def optimize_single_topology(idx_and_matrix, input_laser_tpl, n_mzm, n_det, prop, eval_obj):
    idx, A = idx_and_matrix
    print(f"[Core {mp.current_process().name}] 开始优化 Topology {idx + 1}")

    graph = build_graph_from_matrix(A, input_laser_tpl, n_mzm, n_det)
    graph.initialize_func_grad_hess(prop, eval_obj, exclude_locked=True)

    try:
        start_time = time.time()
        # 对于 8 通道的高维空间，建议加大 maxfevals 和 popsize
        opt_graph, final_params, final_score, _ = parameters_optimize(
            graph,
            method='CMA',
            maxfevals=10000,  # 把评估上限从 1000 放宽到 10000
            popsize=60,  # 增大单代样本量，帮助算法更好摸清高维空间的梯度
            verbose=False
        )

        elapsed = time.time() - start_time
        print(f"   [成功] Topology {idx + 1} 耗时: {elapsed:.1f}s | Cost: {final_score:.6f}")
        return {'id': idx + 1, 'matrix': A, 'score': final_score, 'params': final_params}

    except Exception as e:
        print(f"   [失败] Topology {idx + 1} 报错: {e}")
        return None


# ==========================================
# 5. 批量优化主循环与归档
# ==========================================
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])
    io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io.init_save_dir(sub_path=None, unique_id=True)

    N_MZM, N_DET = 5, 8

    # 统一外部数据目录run_all_topo_8output.py
    DATA_BASE_DIR = "/root/autodl-tmp/aesop/asope_data"

    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(DATA_BASE_DIR, f"Opt_Results_{N_MZM}MZM_{N_DET}DET_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n[工作流] 本次实验的所有结果将保存在独立文件夹: {run_dir}")

    input_freq, output_freq = 10.0e9, 1.25e9
    pulse_width, peak_power = (3e-12, 1.0)
    propagator = Propagator(window_t=4e-9, n_samples=2 ** 13, central_wl=1.55e-6)

    # 使用自定义的锯齿波包络激光器，周期锁定为 1.25GHz 的倒数
    input_laser = PulsedLaser_Sawtooth(
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
    # 8通道相移规则
    target_shifts = [i / 8.0 for i in range(N_DET)]

    for i in range(N_DET):
        nid = 1 + N_MZM + i
        eval_ids.append(nid)
        targets[nid] = input_laser.get_pulse_train(
            propagator.t, pulse_width, 1.0 / output_freq, peak_power / 8, phase_shift=target_shifts[i]
        )
        target_ids_map[nid] = targets[nid]

    evaluator = Local_StrictEvaluator(propagator, targets=target_ids_map, evaluation_nodes=eval_ids)

    npy_file = os.path.join(DATA_BASE_DIR, f"topologies_{N_MZM}MZM_{N_DET}DET_unique_all.npy")
    if not os.path.exists(npy_file):
        raise FileNotFoundError(f"找不到拓扑文件 {npy_file}！请检查路径是否正确。")

    all_topologies = np.load(npy_file)
    n_topos = len(all_topologies)
    print(f"\n[Info] 加载了 {n_topos} 个矩阵。即将使用多进程 CMA 进行物理参数搜索...")

    MAX_TO_OPTIMIZE = n_topos
    tasks = [(idx, A) for idx, A in enumerate(all_topologies[:MAX_TO_OPTIMIZE])]

    cpu_cores = max(1, os.cpu_count() - 2)
    print(f"\n🚀 启动多进程加速，分配了 {cpu_cores} 个 CPU 核心同时工作！")

    leaderboard = []

    worker_func = partial(optimize_single_topology,
                          input_laser_tpl=input_laser,
                          n_mzm=N_MZM, n_det=N_DET,
                          prop=propagator, eval_obj=evaluator)

    with mp.Pool(processes=cpu_cores) as pool:
        results = pool.map(worker_func, tasks)

    for res in results:
        if res is not None:
            leaderboard.append(res)

    if not leaderboard: sys.exit("所有优化均失败。")

    leaderboard.sort(key=lambda x: x['score'])

    txt_path = os.path.join(run_dir, "leaderboard_scores.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Optimization Run: {run_timestamp}\n")
        f.write("=========================================\n")
        for i, res in enumerate(leaderboard):
            f.write(f"Rank {i + 1:03d} | Topology ID: {res['id']:03d} | Cost: {res['score']:.6f}\n")
    print(f"\n[数据保存] 完整的排行榜得分已保存至: {txt_path}")

    # ==========================================
    # 6. 可视化阶段 I：展示最优个体的“分页全家福”
    # ==========================================
    top_k = min(100, len(leaderboard))
    print(f"\n=========================================")
    print(f"🏆 准备生成 Top {top_k} 的可视化数据...")

    n_per_page = 10
    num_pages = math.ceil(top_k / n_per_page)
    first_page_fig = None

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
            first_page_fig = fig_top
        else:
            plt.close(fig_top)

    # ==========================================
    # 7. 可视化阶段 II：动态适配 8 通道详细呈现图
    # ==========================================
    save_dir_top = os.path.join(run_dir, "Detailed_Waveforms_Top100")
    os.makedirs(save_dir_top, exist_ok=True)

    print(f"\n正在生成 Top {top_k} 的详细波形与拓扑对应图表...")

    for rank in range(top_k):
        res = leaderboard[rank]

        # 重建模型并塞入参数
        opt_graph = build_graph_from_matrix(res['matrix'], input_laser, N_MZM, N_DET)
        opt_graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)
        _, models, param_idx, _, _ = opt_graph.extract_parameters_to_list()
        opt_graph.distribute_parameters_from_list(res['params'], models, param_idx)
        opt_graph.propagate(propagator)

        fig_detail = plt.figure(figsize=(18, 2 * N_DET))
        gs = fig_detail.add_gridspec(N_DET, 2, width_ratios=[1, 1.2])

        ax_topo = fig_detail.add_subplot(gs[:, 0])
        title_str = f"Rank {rank + 1} Topology (ID: {res['id']})\nFinal Cost: {res['score']:.4f}"
        if rank == 0: title_str = "🥇 Global Best " + title_str
        visualize_topology(ax_topo, res['matrix'], N_MZM, N_DET, title=title_str)

        # ==========================================
        # 【修改】：防弹版参数信息提取框
        # ==========================================
        param_str_lines = ["【Optimized Parameters】"]

        # 1. 安全提取 MZM 节点参数 (兼容底层封装字典)
        for nid, node_data in opt_graph.nodes.items():
            node_obj = None
            if isinstance(node_data, dict):
                for key, val in node_data.items():
                    if hasattr(val, 'parameters'):
                        node_obj = val
                        break
            else:
                node_obj = node_data

            if node_obj and hasattr(node_obj, 'node_acronym') and 'MZM' in node_obj.node_acronym:
                p = node_obj.parameters
                param_str_lines.append(
                    f"{node_obj.node_acronym}: V_bias={p[1]:.2f}V, V_rf={p[2]:.2f}V, Phase={p[4]:.2f}rad"
                )

        # 2. 安全提取边上的延迟线(Delay Line)参数
        for eid, edge_data in opt_graph.edges.items():
            edge_obj = None
            if isinstance(edge_data, dict):
                for key, val in edge_data.items():
                    if hasattr(val, 'parameters'):
                        edge_obj = val
                        break
            else:
                edge_obj = edge_data

                # 2. 安全提取边上的延迟线(Delay Line)参数
                for eid, edge_data in opt_graph.edges.items():
                    edge_obj = None
                    if isinstance(edge_data, dict):
                        for key, val in edge_data.items():
                            if hasattr(val, 'parameters'):
                                edge_obj = val
                                break
                    else:
                        edge_obj = edge_data

                    if edge_obj and hasattr(edge_obj, 'parameters') and len(edge_obj.parameters) > 0:
                        # 【核心修复】：不管它套了几层列表或数组，强行拍平取第一个数字
                        raw_delay = np.array(edge_obj.parameters).flatten()[0]
                        delay_ps = float(raw_delay) * 1e12
                        if delay_ps > 0.001:
                            param_str_lines.append(f"Delay {eid}: {delay_ps:.2f} ps")

        param_text = "\n".join(param_str_lines)
        ax_topo.text(0.02, 0.02, param_text, transform=ax_topo.transAxes,
                     fontsize=9, verticalalignment='bottom', horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
        # ==========================================

        center_t = propagator.t[propagator.n_samples // 2]
        t_span = 800e-12
        labels = [f"Ch{i + 1} ({target_shifts[i]:.3f})" for i in range(N_DET)]
        axes_wave = []

        # 计算 8 个 Target 的全局最大功率
        global_max_tgt = max([np.max(power_(targets[nid])) for nid in eval_ids])

        # 提前遍历一次，计算 8 个实际输出(Demuxed)的全局最大功率
        out_powers = []
        for nid in eval_ids:
            tmp_out = opt_graph.measure_propagator(nid)
            out_powers.append(np.max(power_(tmp_out)) if tmp_out is not None else 0.0)
        global_max_out = max(out_powers) if max(out_powers) > 0 else 1.0

        for i, nid in enumerate(eval_ids):
            ax = fig_detail.add_subplot(gs[i, 1], sharex=axes_wave[0] if i > 0 else None)
            axes_wave.append(ax)

            out = opt_graph.measure_propagator(nid)
            if out is None: out = np.zeros_like(propagator.t)

            # 获取绝对功率
            p_out = power_(out)
            p_tgt = power_(targets[nid])

            # 各自除以各自的“全局最大值”
            p_out /= (global_max_out + 1e-12)
            p_tgt /= (global_max_tgt + 1e-12)

            ax.plot(propagator.t * 1e12, p_out, 'r-', linewidth=2.0, label='Demuxed')
            ax.plot(propagator.t * 1e12, p_tgt, 'b--', alpha=0.4, label='Target')

            ax.set_ylabel('Norm. Power', fontsize=8)
            ax.set_title(f'{labels[i]} - Node CH_{i + 1}', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.set_xlim((center_t - t_span) * 1e12, (center_t + t_span) * 1e12)
            ax.grid(True, alpha=0.3)

            # 锁死 Y 轴，让 8 个通道的阶梯对比一目了然
            ax.set_ylim(-0.05, 1.1)

            if i < N_DET - 1: ax.tick_params(labelbottom=False)

        axes_wave[-1].set_xlabel('Time (ps)')
        plt.tight_layout()

        file_path = os.path.join(save_dir_top, f"Rank_{rank + 1:03d}_TopoID_{res['id']:03d}.png")
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close(fig_detail)

        if (rank + 1) % 10 == 0:
            print(f"  -> 已完成 {rank + 1} / {top_k} 张图片的渲染...")

    print(f"\n【图表保存】大功告成！Top {top_k} 的详细分析图均已存入 '{save_dir_top}' 文件夹中！")

    if first_page_fig: plt.show()

    # ==========================================
    # 8. 提取并保存 Top 100 个体的拓扑与参数
    # ==========================================
    top_100_export_data = []

    for rank in range(top_k):
        res = leaderboard[rank]
        opt_graph = build_graph_from_matrix(res['matrix'], input_laser, N_MZM, N_DET)
        opt_graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)
        _, models, param_idx, _, _ = opt_graph.extract_parameters_to_list()
        opt_graph.distribute_parameters_from_list(res['params'], models, param_idx)

        node_params = {}
        for nid, node_data in opt_graph.nodes.items():
            node_obj = None
            if isinstance(node_data, dict):
                for key, val in node_data.items():
                    if hasattr(val, 'parameters'):
                        node_obj = val
                        break
            else:
                node_obj = node_data
            if node_obj and hasattr(node_obj, 'parameters'):
                node_params[nid] = copy.deepcopy(node_obj.parameters)

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

        top_100_export_data.append({
            'rank': rank + 1,
            'id': res['id'],
            'score': res['score'],
            'matrix': res['matrix'],
            'raw_params': res['params'],
            'node_params': node_params,
            'edge_params': edge_params
        })

    export_file_path = os.path.join(run_dir, f"Top{top_k}_Optimized_Params.pkl")
    with open(export_file_path, 'wb') as f:
        pickle.dump(top_100_export_data, f)
    print(f"\n[参数归档] Top {top_k} 的拓扑矩阵与参数已安全导出至: {export_file_path}")