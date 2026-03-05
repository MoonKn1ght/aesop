import sys
import pathlib
import os
import platform
import matplotlib.pyplot as plt
import autograd.numpy as np
import copy
import time
import gc

from sympy import false

# ==========================================
# 0. 环境路径设置
# ==========================================
parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

from lib.functions import InputOutput, parse_command_line_args
from lib.graph import Graph
from lib.base_classes import NodeType
from simulator.fiber.node_types import MultiPath, SinglePath

from simulator.fiber.assets.propagator import Propagator
from simulator.fiber.assets.functions import power_
from simulator.fiber.node_types_subclasses.inputs import PulsedLaser
from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink
from algorithms.parameter_optimization import parameters_optimize
from simulator.fiber.evaluator import Evaluator

plt.close('all')


# ==========================================
# 1. 本地组件 (完整保留你的原版代码)
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
        if isinstance(state, list): state = state[0]
        state = np.squeeze(state)
        if state.ndim > 1: raise ValueError(f"Input shape {state.shape} wrong.")
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

    def update_attributes(self, num_inputs, num_outputs):
        if not hasattr(self, 'parameters') or len(self.parameters) != 6: self.parameters = self.default_parameters

    def propagate(self, states, propagator, num_inputs, num_outputs, save_transforms=False):
        v_pi, v_bias, v_rf, f_rf, phase_rf, loss = self.parameters
        clean_states = []
        for s in states: clean_states.append(np.squeeze(s))
        if len(clean_states) == 1:
            s0 = clean_states[0]
            in_mat = np.stack([s0, np.zeros_like(s0)], axis=0)
        else:
            in_mat = np.stack(clean_states, axis=0)
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


### 评估器
# class Local_StrictEvaluator(Evaluator):
#     def __init__(self, propagator, targets, evaluation_nodes):
#         super().__init__()
#         self.propagator = propagator
#         self.targets = targets
#         self.evaluation_nodes = evaluation_nodes
#         self.masks = {}
#         for nid, target_field in self.targets.items():
#             p_tgt = power_(target_field)
#             max_p = np.max(p_tgt) + 1e-15
#             norm_tgt = p_tgt / max_p
#             self.masks[nid] = (norm_tgt < 0.05).astype(float)
#
#     def evaluate_graph(self, graph, propagator=None):
#         return self.evaluate(graph)
#
#     def evaluate(self, graph):
#         total_cost = 0.0
#         for node in self.evaluation_nodes:
#             signal = graph.measure_propagator(node)
#             target = self.targets[node]
#             if signal is None:
#                 total_cost += 5000.0
#                 continue
#
#             p_signal = np.abs(signal) ** 2
#             p_target = np.abs(target) ** 2
#
#             max_sig = np.max(p_signal) + 1e-15
#             max_tar = np.max(p_target) + 1e-15
#             p_sig_norm = p_signal / max_sig
#             p_tar_norm = p_target / max_tar
#
#             shape_error = np.mean((p_sig_norm - p_tar_norm) ** 2)
#
#             noise_content = p_sig_norm * self.masks[node]
#             penalty = np.mean(noise_content ** 2) * 100.0
#
#             total_cost += (shape_error + penalty)
#         return total_cost


# ### 升级版评估器：动态加权误差计算
# class Local_StrictEvaluator(Evaluator):
#     def __init__(self, propagator, targets, evaluation_nodes):
#         super().__init__()
#         self.propagator = propagator
#         self.targets = targets
#         self.evaluation_nodes = evaluation_nodes
#         self.masks = {}
#         self.weights = {}  # 新增：权重映射表
#
#         for nid, target_field in self.targets.items():
#             p_tgt = power_(target_field)
#             max_p = np.max(p_tgt) + 1e-15
#             norm_tgt = p_tgt / max_p
#
#             # 1. 杂波掩模：标记出功率小于 5% 的黑暗区域
#             self.masks[nid] = (norm_tgt < 0.05).astype(float)
#
#             # 2. 误差权重掩模：你的核心思想实现
#             # 基础权重设为 1.0，在目标脉冲所在的区域，将误差计算权重放大 50 倍
#             w = np.ones_like(norm_tgt)
#             w[norm_tgt >= 0.05] = 50.0
#             self.weights[nid] = w
#
#     def evaluate_graph(self, graph, propagator=None):
#         return self.evaluate(graph)
#
#     def evaluate(self, graph):
#         total_cost = 0.0
#         for node in self.evaluation_nodes:
#             signal = graph.measure_propagator(node)
#             target = self.targets[node]
#             if signal is None:
#                 total_cost += 5000.0
#                 continue
#
#             p_signal = np.abs(signal) ** 2
#             p_target = np.abs(target) ** 2
#
#             max_sig = np.max(p_signal) + 1e-15
#             max_tar = np.max(p_target) + 1e-15
#             p_sig_norm = p_signal / max_sig
#             p_tar_norm = p_target / max_tar
#
#             # 【修改核心】带动态权重的均方误差 (Weighted MSE)
#             # 现在，脉冲位置哪怕有一点没对齐，产生的误差都会被放大 50 倍
#             shape_error = np.mean(self.weights[node] * (p_sig_norm - p_tar_norm) ** 2)
#
#             # 适度降低背景杂波的惩罚力度（从 100.0 降到 20.0），避免算法过于在乎背景而忽略了脉冲对齐
#             noise_content = p_sig_norm * self.masks[node]
#             penalty = np.mean(noise_content ** 2) * 20.0
#
#             total_cost += (shape_error + penalty)
#
#         return total_cost

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
            # 【修复点 1】给目标波形脱壳
            norm_tgt = np.squeeze(p_tgt / max_p)

            self.target_norms[nid] = norm_tgt
            # 【修复点 2】给掩模脱壳
            self.masks[nid] = np.squeeze((norm_tgt < 0.05).astype(float))

    def evaluate_graph(self, graph, propagator=None):
        return self.evaluate(graph)

    def evaluate(self, graph):
        total_cost = 0.0
        for node in self.evaluation_nodes:
            signal = graph.measure_propagator(node)
            if signal is None:
                total_cost += 5000.0
                continue

            p_signal = np.abs(signal) ** 2
            max_sig = np.max(p_signal) + 1e-15
            # 【修复点 3】给测量到的仿真信号脱壳
            p_sig_norm = np.squeeze(p_signal / max_sig)

            norm_tgt = self.target_norms[node]

            # 现在的 p_sig_norm 和 norm_tgt 都是绝对纯净的一维数组了
            xcorr = np.correlate(p_sig_norm, norm_tgt, mode='full')
            best_shift_idx = np.argmax(xcorr) - (len(norm_tgt) - 1)

            shifted_tgt = np.roll(norm_tgt, best_shift_idx)
            shifted_mask = np.roll(self.masks[node], best_shift_idx)

            shape_error = np.mean((p_sig_norm - shifted_tgt) ** 2)
            noise_content = p_sig_norm * shifted_mask
            penalty = np.mean(noise_content ** 2) * 20.0

            total_cost += (shape_error + penalty)

        return total_cost


### 构建拓扑
def build_active_demux_graph(input_laser, target_delay_dt, output_freq):
    nodes = {}
    edges = {}
    nodes[0] = TerminalSource()
    nodes[0].node_acronym = 'SRC'
    nodes[1] = Local_MZM_Modulated()
    nodes[1].node_acronym = 'MZM_ROOT_5G'
    nodes[2] = Local_MZM_Modulated()
    nodes[2].node_acronym = 'MZM_TOP_2.5G'
    nodes[3] = Local_MZM_Modulated()
    nodes[3].node_acronym = 'MZM_BOT_2.5G'



    # ### 人为设置好的正确参数
    # # Root: 5G (分配奇偶，负责上下半区，不动)
    # nodes[1].parameters = [3.5, 1.5, 1.0, output_freq * 2.0, np.pi, 0.1]
    #
    # # Top: 2.5G (刚刚证明了 0.0 是让 Ch1, Ch2 完全正确的答案)
    # nodes[2].parameters = [3.5, 1.5, 1.0, output_freq, 0.0, 0.1]
    #
    # # Bot: 2.5G (恢复到上一轮让 Ch3, Ch4 表现完美的答案！)
    # nodes[3].parameters = [3.5, 1.5, 1.0, output_freq, np.pi / 2, 0.1]

    ### 用CMA
    # Root: 5G (分配奇偶，负责上下半区，不动)
    nodes[1].parameters = [3.5, 1.5, 1.0, output_freq * 2.0, 0.0, 0.1]

    # Top: 2.5G (刚刚证明了 0.0 是让 Ch1, Ch2 完全正确的答案)
    nodes[2].parameters = [3.5, 1.5, 1.0, output_freq, 0.0, 0.1]

    # Bot: 2.5G (恢复到上一轮让 Ch3, Ch4 表现完美的答案！)
    nodes[3].parameters = [3.5, 1.5, 1.0, output_freq, 0.0, 0.1]

    for i in range(4, 8):
        nodes[i] = TerminalSink(node_name=f'sink{i - 3}')
        nodes[i].node_acronym = f'CH_{i - 3}'

    input_laser.src_port_idx = 0
    input_laser.dst_port_idx = 0
    edges[(0, 1)] = input_laser

    # 【修改点 1】把基础延迟设为 0.0
    base_delay = 0.0

    def make_delay(delay_time, src_p, dst_p):
        dl = Local_SafeDelayLine()
        dl.parameters = [delay_time]
        dl.node_lock = True
        dl.src_port_idx = src_p
        dl.dst_port_idx = dst_p
        return dl

    edges[(1, 2)] = make_delay(base_delay, 0, 0)
    edges[(1, 3)] = make_delay(base_delay + 2 * target_delay_dt, 1, 0)
    edges[(2, 4)] = make_delay(base_delay, 0, 0)
    edges[(2, 5)] = make_delay(base_delay + 1 * target_delay_dt, 1, 0)
    edges[(3, 6)] = make_delay(base_delay, 0, 0)
    edges[(3, 7)] = make_delay(base_delay + 1 * target_delay_dt, 1, 0)

    graph = Graph.init_graph(nodes, edges)
    return graph


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])
    io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io.init_save_dir(sub_path=None, unique_id=True)

    input_freq = 10.0e9
    output_freq = 2.5e9

    # 【修改点 2】把级联间的时间差也设为 0.0
    target_dt = 0.0

    pulse_width, peak_power = (3e-12, 1.0)
    n_samples = 2 ** 13
    propagator = Propagator(window_t=4e-9, n_samples=n_samples, central_wl=1.55e-6)

    print(f"\n[Info] Hierarchical OTI Demux (IDEAL ZERO DELAY):")
    print(f"  Root MZM: 5.0 GHz (Split Even/Odd)")
    print(f"  Leaf MZM: 2.5 GHz (Final Demux)")
    print(f"  Target Sequence: 0, 0.5, 0.25, 0.75")

    input_laser = PulsedLaser(parameters_from_name={
        'pulse_width': pulse_width, 'peak_power': peak_power,
        't_rep': 1.0 / input_freq, 'pulse_shape': 'gaussian',
        'central_wl': 1.55e-6, 'train': True
    })
    input_laser.node_lock = True

    targets = {}
    # 【修改点 3】由于完全没有物理延迟，飞行时间补偿也是 0.0
    base_flight = 0.0
    phase_comp = base_flight * output_freq
    t_pulse = pulse_width
    t_peak = peak_power / 4

    targets[4] = input_laser.get_pulse_train(propagator.t, t_pulse, 1.0 / output_freq, t_peak,
                                             phase_shift=0.00 + phase_comp)
    targets[5] = input_laser.get_pulse_train(propagator.t, t_pulse, 1.0 / output_freq, t_peak,
                                             phase_shift=0.50 + phase_comp)
    targets[6] = input_laser.get_pulse_train(propagator.t, t_pulse, 1.0 / output_freq, t_peak,
                                             phase_shift=0.25 + phase_comp)
    targets[7] = input_laser.get_pulse_train(propagator.t, t_pulse, 1.0 / output_freq, t_peak,
                                             phase_shift=0.75 + phase_comp)

    target_ids_map = {4: targets[4], 5: targets[5], 6: targets[6], 7: targets[7]}
    eval_ids = [4, 5, 6, 7]

    # evaluator = Local_StrictEvaluator(
    #     propagator, targets=target_ids_map,
    #     evaluation_nodes=eval_ids
    # )
    evaluator = Local_XCorrEvaluator(
        propagator, targets=target_ids_map,
        evaluation_nodes=eval_ids
    )

    print("\n[Step 1] 构建分层架构 (5G + 2.5G) [理想无延迟]...")
    graph = build_active_demux_graph(input_laser, target_dt, output_freq=output_freq)

    print("\n[Step 2] 开始优化...")
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    start_time = time.time()
    opt_graph, final_params, final_score, _ = parameters_optimize(
        # graph, method='L-BFGS',
        # graph, method='L-BFGS+GA',
        graph, method='CMA',
        n_generations=20, population_size=20,
        verbose=True, num_cpus=1
    )
    elapsed = time.time() - start_time
    print(f"\n[Result] 优化完成! 耗时: {elapsed:.2f}s | Final Cost: {final_score:.6f}")

    print("\n[Step 3] 绘制波形图...")
    opt_graph.propagate(propagator)

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    center_idx = propagator.n_samples // 2
    center_t = propagator.t[center_idx]
    t_span = 800e-12

    labels = ["Ch1 (0ps / 0.0)", "Ch2 (200ps / 0.5)", "Ch3 (100ps / 0.25)", "Ch4 (300ps / 0.75)"]

    for i, nid in enumerate(eval_ids):
        out = opt_graph.measure_propagator(nid)
        if out is None: out = np.zeros_like(propagator.t)

        p_out = power_(out)
        p_tgt = power_(targets[nid])
        p_out /= (np.max(p_out) + 1e-12)
        p_tgt /= (np.max(p_tgt) + 1e-12)

        ax = axes[i]
        ax.plot(propagator.t * 1e12, p_out, 'r-', linewidth=2.0, label='Demuxed')
        ax.plot(propagator.t * 1e12, p_tgt, 'k--', alpha=0.4, label='Target')
        ax.set_ylabel(f'Norm Power')
        ax.set_title(f'{labels[i]} - Node {nid}')
        ax.legend(loc='upper right')
        ax.set_xlim((center_t - t_span) * 1e12, (center_t + t_span) * 1e12)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (ps)')
    fig.suptitle(f'Ideal Hierarchical Demux (Zero Delay)\nCost: {final_score:.4f}')
    plt.tight_layout()
    plt.show()