import sys
import pathlib
import os
import platform
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import autograd.numpy as np
import copy
import pickle

print(f"--- 正在使用的 Python 路径: {sys.executable} ---")

# ==========================================
# 0. 环境路径设置
# ==========================================
parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

from lib.graph import Graph
from simulator.fiber.node_types import MultiPath, SinglePath
from simulator.fiber.assets.propagator import Propagator
from simulator.fiber.assets.functions import power_
from simulator.fiber.node_types_subclasses.inputs import PulsedLaser
from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink
from simulator.fiber.evaluator import Evaluator

plt.close('all')

# ==========================================
# 1. 核心参数与文件路径设置
# ==========================================
N_MZM, N_DET = 7, 8
input_freq, output_freq = 10.0e9, 1.25e9
pulse_width, peak_power = (3e-12, 1.0)

# 【注意】：请把这里换成你真正跑出 Rank 1 的那个 pkl 文件的绝对路径！
PKL_FILE_PATH = "/root/autodl-tmp/aesop/asope_data/Opt_Results_7MZM_8DET_20260409_185719/Top100_Optimized_Params.pkl"

# ==========================================
# 2. 物理组件与绘图辅助 (纯净版，不含优化)
# ==========================================
from collections import defaultdict, deque
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path

PORT_MAP = {1: ("out_top", "in_top"), 2: ("out_top", "in_bottom"),
            3: ("out_bottom", "in_top"), 4: ("out_bottom", "in_bottom")}


def compute_levels(A, source=0):
    n = A.shape[0];
    indeg = A.astype(bool).sum(axis=0)
    q = deque([source]);
    level = [-1] * n;
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
    x, y = center;
    w, h = 1.2, 0.6
    rect = Rectangle((x - w / 2, y - h / 2), w, h, fc="#d2691e", ec="brown", lw=1.5, zorder=20)
    ax.add_patch(rect)
    ports = {"in_top": (x - w / 2, y + h / 4), "in_bottom": (x - w / 2, y - h / 4),
             "out_top": (x + w / 2, y + h / 4), "out_bottom": (x + w / 2, y - h / 4)}
    for p in ports.values(): ax.plot(*p, "ko", ms=3, zorder=21)
    ax.text(x, y, label, ha="center", va="center", fontsize=8, zorder=22, color='white', fontweight='bold')
    return ports


def draw_edge(ax, p1, p2, bend=0.0, style='solid', alpha=0.6, color="black", lw=1.1):
    x1, y1 = p1;
    x2, y2 = p2
    ctrl_x1 = x1 + (x2 - x1) * 0.4;
    ctrl_y1 = y1 + bend
    ctrl_x2 = x2 - (x2 - x1) * 0.4;
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
    port_pos = {};
    sorted_levels = sorted(layers.keys())
    V_SPACING, H_SPACING = 3.5, 6.0;
    all_ys = []
    for lv in sorted_levels:
        nodes = layers[lv];
        n_nodes = len(nodes)
        total_span = (n_nodes - 1) * V_SPACING
        ys = np.linspace(total_span / 2, -total_span / 2, n_nodes) if n_nodes > 1 else [0]
        for i, node in enumerate(nodes):
            pos[node] = (lv * H_SPACING, ys[i]);
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
    ax.axis("off");
    ax.autoscale_view();
    ax.margins(0.1)


class Local_SafeDelayLine(SinglePath):
    node_acronym = 'DL'
    number_of_parameters = 1
    node_lock = True

    def __init__(self, **kwargs):
        self.default_parameters = [0.0];
        self.upper_bounds = [10e-9];
        self.lower_bounds = [0.0]
        self.data_types = ['float'];
        self.step_sizes = [None];
        self.parameter_imprecisions = [0.0]
        self.parameter_units = ['s'];
        self.parameter_locks = [False];
        self.parameter_names = ['delay']
        self.parameter_symbols = [r"$\tau$"];
        self._n = 1.444
        super().__init__(**kwargs)

    def propagate(self, state, propagator, save_transforms=False):
        if state is None:
            state = np.zeros_like(propagator.t)
        elif isinstance(state, (list, tuple)):
            state = state[0] if len(state) > 0 and state[0] is not None else np.zeros_like(propagator.t)
        state = np.squeeze(state);
        delay = self.parameters[0];
        dt = propagator.dt;
        n = len(state)
        spectrum = np.fft.fft(state);
        freqs = np.fft.fftfreq(n, d=dt)
        phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay)
        return np.squeeze(np.fft.ifft(spectrum * phase_shift))


class Local_MZM_Modulated(MultiPath):
    node_acronym = 'MZM_MOD'
    node_lock = False

    def __init__(self, **kwargs):
        self.number_of_parameters = 6
        self.parameter_names = ['v_pi', 'v_bias', 'v_rf', 'f_rf', 'phase_rf', 'loss']
        self.default_parameters = [3.5, 1.75, 1.0, 1.25e9, 0.0, 0.1]
        self.upper_bounds = [10.0, 10.0, 20.0, 50.0e9, 2 * np.pi, 1.0];
        self.lower_bounds = [1.0, -10.0, 0.0, 0.0, -2 * np.pi, 0.0]
        self.data_types = ['float'] * 6;
        self.step_sizes = [0.5, 0.5, 0.5, 1.0e9, 0.1, 0.01]
        self.parameter_imprecisions = [0.01] * 6;
        self.parameter_units = ['V'] * 3 + ['Hz', 'rad', 'dB']
        self.parameter_locks = [True, False, False, True, False, True]
        self.parameter_symbols = [r"$V_{\pi}$", r"$V_{bias}$", r"$V_{RF}$", r"$f_{RF}$", r"$\theta_{RF}$", "IL"]
        super().__init__(**kwargs)
        self._range_input_edges = (1, 50);
        self.num_inputs = 2;
        self.num_outputs = 2

    def update_attributes(self, num_inputs, num_outputs):
        self.num_inputs = max(2, num_inputs);
        self.num_outputs = max(2, num_outputs)
        if not hasattr(self, 'parameters') or len(self.parameters) != 6: self.parameters = self.default_parameters

    def propagate(self, states, propagator, num_inputs, num_outputs, save_transforms=False):
        v_pi, v_bias, v_rf, f_rf, phase_rf, loss = self.parameters;
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
        t = np.squeeze(propagator.t);
        rf_signal = v_rf * np.cos(2 * np.pi * f_rf * t + phase_rf)
        phi_t = (np.pi * (v_bias + rf_signal)) / (2.0 * v_pi)
        cos_phi = np.cos(phi_t);
        sin_phi = np.sin(phi_t);
        factor = (1.0 - loss)
        out1 = (in_top * cos_phi - 1j * in_bot * sin_phi) * factor
        out2 = (-1j * in_top * sin_phi + in_bot * cos_phi) * factor
        result = [out1, out2];
        target_len = getattr(self, 'num_outputs', 2)
        while len(result) < target_len: result.append(np.zeros_like(out1))
        return result


class Local_TerminalSource(TerminalSource):
    def __init__(self, **kwargs): super().__init__(**kwargs); self._range_output_edges = (1, 50)

    def update_attributes(self, num_inputs, num_outputs): self.num_outputs = max(1, num_outputs)

    def propagate(self, states, propagator, num_inputs=0, num_outputs=1, save_transforms=False):
        base_state = states[0] if isinstance(states, (list, tuple)) and len(states) > 0 else states
        target_len = getattr(self, 'num_outputs', 1);
        return [base_state] * target_len


class Local_TerminalSink(TerminalSink):
    def __init__(self, **kwargs):
        super().__init__(**kwargs); self._range_input_edges = (1, 50)

    def update_attributes(self, num_inputs, num_outputs):
        self.num_inputs = max(1, num_inputs); self.num_outputs = max(1, num_outputs)

    def propagate(self, states, propagator, num_inputs=1, num_outputs=0, save_transforms=False):
        if isinstance(states, (list, tuple)):
            valid_states = [s for s in states if s is not None]
            self.state = sum(valid_states) if valid_states else np.zeros_like(propagator.t)
        else:
            self.state = states if states is not None else np.zeros_like(propagator.t)
        target_len = getattr(self, 'num_outputs', 1);
        return [self.state] * target_len


class PulsedLaser_Sawtooth(PulsedLaser):
    def __init__(self, envelope_period=1.0 / 1.25e9, **kwargs):
        super().__init__(**kwargs);
        self.envelope_period = envelope_period

    def get_pulse_train(self, t, pulse_width, rep_t, peak_power, pulse_shape='gaussian', phase_shift=0):
        phase = (t % self.envelope_period) / self.envelope_period
        envelope = 0.1 + 0.9 * phase
        dynamic_peak_power = peak_power * envelope
        shifted_t = t - phase_shift * rep_t
        wrapped_t = np.sin(np.pi * shifted_t / rep_t)
        unwrapped_t = np.arcsin(wrapped_t) * rep_t / np.pi
        pulse_width = pulse_width / (2 * np.sqrt(np.log(2)))
        if pulse_shape == 'gaussian':
            pulse = self.gaussian(unwrapped_t, pulse_width)
        elif pulse_shape == 'sech':
            pulse = self.sech(unwrapped_t, pulse_width)
        elif pulse_shape == 'delta':
            import scipy.signal as sig; pulse = sig.unit_impulse(shape=t.shape)
        else:
            raise RuntimeError(f"Pulsed Laser: {pulse_shape} is not a defined pulse shape")
        state = pulse * np.sqrt(np.abs(dynamic_peak_power))
        return state


def build_graph_from_matrix(A, input_laser_template, N_mzm, N_det):
    nodes = {};
    edges = {};
    n = A.shape[0]
    nodes[0] = Local_TerminalSource();
    nodes[0].node_acronym = 'SRC'
    for i in range(1, 1 + N_mzm):
        mzm = Local_MZM_Modulated();
        mzm.node_acronym = f'MZM_{i}';
        mzm.port_mapping = [];
        nodes[i] = mzm
    for i in range(1 + N_mzm, n):
        ch_idx = i - N_mzm;
        nodes[i] = Local_TerminalSink(node_name=f'sink_{ch_idx}');
        nodes[i].node_acronym = f'CH_{ch_idx}'
    port_idx_map = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}
    current_in_port = {i: 0 for i in range(n)};
    current_out_port = {i: 0 for i in range(n)}
    for u in range(n):
        for v in range(n):
            if A[u, v] > 0:
                out_p, original_in_p = port_idx_map[A[u, v]];
                safe_in_p = current_in_port[v];
                current_in_port[v] += 1
                if v >= 1 and v <= N_mzm: nodes[v].port_mapping.append(original_in_p)
                if u == 0:
                    edge_obj = copy.deepcopy(input_laser_template);
                    edge_obj.src_port_idx = current_out_port[u]
                    current_out_port[u] += 1;
                    edge_obj.dst_port_idx = safe_in_p;
                    edges[(u, v)] = edge_obj
                else:
                    dl = Local_SafeDelayLine();
                    dl.parameters = [0.0];
                    dl.node_lock = True
                    dl.src_port_idx = out_p;
                    dl.dst_port_idx = safe_in_p;
                    edges[(u, v)] = dl
    nodes[0].num_outputs = max(1, current_out_port[0])
    for i in range(1, 1 + N_mzm): nodes[i].num_inputs = max(2, current_in_port[i])
    for i in range(1 + N_mzm, n): nodes[i].num_inputs = max(1, current_in_port[i])
    return Graph.init_graph(nodes, edges)


def safe_extract_float(param_list, index=0, default=0.0):
    try:
        return float(np.array(param_list[index]).flatten()[0])
    except (IndexError, TypeError, ValueError):
        return default


# ==========================================
# 3. 核心提取与画图函数
# ==========================================
def verify_and_plot_rank(rank_idx_to_check=0):
    if not os.path.exists(PKL_FILE_PATH):
        raise FileNotFoundError(f"找不到参数文件: {PKL_FILE_PATH}")

    with open(PKL_FILE_PATH, 'rb') as f:
        top_100_data = pickle.load(f)

    target_data = top_100_data[rank_idx_to_check]
    print(f"\n=========================================")
    print(f"🚀 正在复现 Rank {target_data['rank']} (ID: {target_data['id']}) 的最优拓扑参数...")
    print(f"   记录的最低 Cost: {target_data['score']:.6f}")
    print("=========================================\n")

    # 1. 准备物理引擎
    propagator = Propagator(window_t=4e-9, n_samples=2 ** 13, central_wl=1.55e-6)
    input_laser = PulsedLaser_Sawtooth(
        envelope_period=1.0 / output_freq,
        parameters_from_name={
            'pulse_width': pulse_width, 'peak_power': peak_power,
            't_rep': 1.0 / input_freq, 'pulse_shape': 'gaussian',
            'central_wl': 1.55e-6, 'train': True
        }
    )
    input_laser.node_lock = True

    targets = {}
    eval_ids = []
    target_shifts = [i / N_DET for i in range(N_DET)]

    for i in range(N_DET):
        nid = 1 + N_MZM + i
        eval_ids.append(nid)
        targets[nid] = input_laser.get_pulse_train(
            propagator.t, pulse_width, 1.0 / output_freq, peak_power / N_DET, phase_shift=target_shifts[i]
        )

    # 2. 从 matrix 重建光路图并塞入参数
    print("-> 正在重构光路拓扑并注入字典参数...")
    opt_graph = build_graph_from_matrix(target_data['matrix'], input_laser, N_MZM, N_DET)

    for nid, params in target_data.get('node_params', {}).items():
        node = opt_graph.nodes.get(nid)
        target_node = node[list(node.keys())[0]] if isinstance(node, dict) else node
        if hasattr(target_node, 'parameters'): target_node.parameters = params

    for eid, params in target_data.get('edge_params', {}).items():
        edge = opt_graph.edges.get(eid)
        target_edge = edge[list(edge.keys())[0]] if isinstance(edge, dict) else edge
        if hasattr(target_edge, 'parameters'): target_edge.parameters = params

    print("-> 正在进行物理正向传播仿真...")
    opt_graph.propagate(propagator)

    # 3. 绘图
    print("-> 仿真完成，正在渲染验证图...")
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(N_DET + 1, 2, width_ratios=[3, 1.2], hspace=0.4, wspace=0.1)

    t_ps = propagator.t * 1e12
    center_t = t_ps[len(t_ps) // 2]
    t_span = 800

    # (A) 第一行：输入信号
    ax_in = fig.add_subplot(gs[0, 0])
    in_signal = input_laser.get_pulse_train(propagator.t, pulse_width, 1.0 / input_freq, peak_power)
    ax_in.plot(t_ps, power_(in_signal), 'k-', lw=1.2, label='10GHz Input')
    ax_in.set_title(f"Rank {target_data['rank']} (Cost: {target_data['score']:.4f}) Verification", fontweight='bold')
    ax_in.set_ylabel("Input (W)")
    ax_in.set_xlim(center_t - t_span, center_t + t_span)
    ax_in.grid(True, alpha=0.3)
    ax_in.legend(loc='upper right')

    # 计算目标最大功率（用于全局归一化）
    # 计算目标最大功率（用于全局归一化）
    global_max_tgt = max([np.max(power_(targets[nid])) for nid in eval_ids])

    # 【新增】：计算实际输出的最大功率（用于实际波形的全局归一化）
    out_powers = []
    for nid in eval_ids:
        tmp_out = opt_graph.measure_propagator(nid)
        out_powers.append(np.max(power_(tmp_out)) if tmp_out is not None else 0.0)
    global_max_out = max(out_powers) if max(out_powers) > 0 else 1.0

    # (B) 后续行：各通道对比
    for i, nid in enumerate(eval_ids):
        ax = fig.add_subplot(gs[i + 1, 0], sharex=ax_in)
        out_signal = opt_graph.measure_propagator(nid)
        p_out = power_(out_signal) if out_signal is not None else np.zeros_like(t_ps)
        p_tgt = power_(targets[nid])

        # 【核心修复】：使用全局最大值作为分母！让阶梯波形重现天日！
        p_out_norm = p_out / (global_max_out + 1e-12)
        p_tgt_norm = p_tgt / (global_max_tgt + 1e-12)

        ax.plot(t_ps, p_out_norm, 'r-', lw=1.8, label=f'Demuxed Ch{i + 1}')
        ax.plot(t_ps, p_tgt_norm, 'b--', alpha=0.4, label='Target')
        ax.set_ylabel("Norm Pwr")
        ax.set_title(f"Shift: {target_shifts[i]:.3f} <---> Actual Ch{i + 1}", fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right')

        if i < N_DET - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_xlabel("Time (ps)", fontsize=11)

    # (C) 右侧：优美的参数面板
    ax_params = fig.add_subplot(gs[:, 1])
    ax_params.axis('off')

    param_text = [f"Rank {target_data['rank']} (ID: {target_data['id']})", "=" * 30]

    # 提取并格式化 MZM 参数
    param_text.append("[MZM Parameters]")
    for nid in range(1, 1 + N_MZM):
        node = opt_graph.nodes.get(nid)
        target_node = node[list(node.keys())[0]] if isinstance(node, dict) else node
        if hasattr(target_node, 'parameters'):
            p = target_node.parameters
            param_text.append(f"> MZM_{nid}:")
            param_text.append(f"  V_bias : {safe_extract_float(p, 1):6.2f} V")
            param_text.append(f"  V_RF   : {safe_extract_float(p, 2):6.2f} V")
            param_text.append(f"  Phase  : {safe_extract_float(p, 4):6.2f} rad\n")

    # 提取延迟线参数
    param_text.append("[Delay Lines > 1ps]")
    for eid, edge_data in opt_graph.edges.items():
        target_edge = edge_data[list(edge_data.keys())[0]] if isinstance(edge_data, dict) else edge_data
        if hasattr(target_edge, 'parameters') and len(target_edge.parameters) > 0:
            d_ps = safe_extract_float(target_edge.parameters, 0) * 1e12
            if d_ps > 1.0:
                param_text.append(f"  Path {eid[0]}->{eid[1]} : {d_ps:.1f} ps")

    ax_params.text(0.05, 0.95, "\n".join(param_text), transform=ax_params.transAxes,
                   fontsize=10, fontfamily='monospace', verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    # 保存并显示
    save_name = f"Final_Verify_Rank{target_data['rank']}_ID{target_data['id']}.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"✅ 验证图谱已成功保存为: {save_name}\n")


if __name__ == '__main__':
    # 只需要修改这里，即可验证不同名次的拓扑
    verify_and_plot_rank(rank_idx_to_check=8)