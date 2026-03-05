import sys
import pathlib
import os
import platform
import matplotlib.pyplot as plt
import autograd.numpy as np
import copy
import time
import gc

# ==========================================
# 0. 环境路径设置
# ==========================================
parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

# ==========================================
# 1. 模块导入 (已修正 Import 路径)
# ==========================================
from lib.functions import InputOutput, parse_command_line_args
from lib.graph import Graph

# 【修改点】按您的要求拆分导入
from lib.base_classes import NodeType
from simulator.fiber.node_types import MultiPath

from simulator.fiber.assets.propagator import Propagator
from simulator.fiber.assets.functions import power_
from simulator.fiber.node_types_subclasses.inputs import PulsedLaser
from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink
from simulator.fiber.evaluator_subclasses.evaluator_pulserep_new import PulseRepetition_multi
from algorithms.parameter_optimization import parameters_optimize

plt.close('all')


# ==========================================
# 2. 自定义修复组件 (DelayWaveguide & MZM)
# ==========================================

class DelayWaveguide(NodeType):
    """
    【自定义波导 (Edge类型)】
    支持群延时(ng)，输入输出为单信号，强制1D。
    """
    node_acronym = 'WG'
    node_lock = True

    def __init__(self, length=0.0, ng=1.46, **kwargs):
        self.number_of_parameters = 0
        self.default_parameters = []
        self.parameter_names = []
        self.upper_bounds = [];
        self.lower_bounds = [];
        self.data_types = []
        self.step_sizes = [];
        self.parameter_imprecisions = []
        self.parameter_units = [];
        self.parameter_locks = [];
        self.parameter_symbols = []

        super().__init__(**kwargs)
        self.length = length
        self.ng = ng

    def update_attributes(self, num_inputs, num_outputs):
        pass

    def propagate(self, input_state, propagator, save_transforms=False):
        field = input_state
        if np.ndim(field) > 1: field = np.reshape(field, (-1,))

        tau = self.length * self.ng / 299792458.0
        if tau <= 1e-15: return field

        dt = propagator.dt
        n = len(field)
        spectrum = np.fft.fft(field)
        freqs = np.fft.fftfreq(n, d=dt)
        phase_shift = np.exp(-1j * 2 * np.pi * freqs * tau)
        return np.fft.ifft(spectrum * phase_shift)


class MZM2x2Node_Fixed(MultiPath):
    """
    【自定义 MZM (Node类型)】
    修复矩阵乘法维度问题。
    """
    node_acronym = 'MZM'
    node_lock = False

    def __init__(self, **kwargs):
        self.number_of_parameters = 3
        self.parameter_names = ['v_pi', 'v_bias', 'insertion_loss']
        self.default_parameters = [3.5, 1.75, 0.1]
        self.upper_bounds = [10.0, 10.0, 1.0]
        self.lower_bounds = [1.0, -10.0, 0.0]
        self.data_types = ['float', 'float', 'float']
        self.step_sizes = [0.1, 0.1, 0.01]
        self.parameter_imprecisions = [0.01, 0.01, 0.001]
        self.parameter_units = ['V', 'V', 'dB']
        self.parameter_locks = [False, False, False]
        self.parameter_symbols = [r"$V_{\pi}$", r"$V_{bias}$", r"$IL$"]
        super().__init__(**kwargs)
        self._range_input_edges = [1, 2]

    def update_attributes(self, num_inputs, num_outputs):
        if not hasattr(self, 'parameters') or len(self.parameters) != 3:
            self.parameters = self.default_parameters

    def propagate(self, states, propagator, num_inputs, num_outputs, save_transforms=False):
        v_pi, v_bias, loss = self.parameters
        phi = (np.pi * v_bias) / (2.0 * v_pi)
        S = np.array([[np.cos(phi), -1j * np.sin(phi)], [-1j * np.sin(phi), np.cos(phi)]])

        clean_states = []
        for s in states:
            if np.ndim(s) > 1:
                clean_states.append(np.reshape(s, (-1,)))
            else:
                clean_states.append(s)

        if len(clean_states) == 1:
            s0 = clean_states[0]
            in_mat = np.stack([s0, np.zeros_like(s0)], axis=0)
        else:
            in_mat = np.stack(clean_states, axis=0)

        out_mat = np.matmul(S, in_mat) * (1.0 - loss)
        return [out_mat[0], out_mat[1]]


# ==========================================
# 3. 构建经典 OTI 二叉树架构
# ==========================================
def build_classic_oti_graph(input_laser, n_eff, delay_unit_len):
    """
    构建 1:4 OTI 架构。
    目标：Ch1(0), Ch2(100ps), Ch3(200ps), Ch4(300ps)
    Delay Unit = 100ps 对应的波导长度
    """
    nodes = {}
    edges = {}

    # 1. 节点定义
    nodes[0] = TerminalSource();
    nodes[0].node_acronym = 'SRC'

    # Root MZM
    nodes[1] = MZM2x2Node_Fixed();
    nodes[1].node_acronym = 'MZM_Root'
    # Level 2 MZMs
    nodes[2] = MZM2x2Node_Fixed();
    nodes[2].node_acronym = 'MZM_Top'
    nodes[3] = MZM2x2Node_Fixed();
    nodes[3].node_acronym = 'MZM_Bot'

    # 随机初始化参数
    for i in range(1, 4):
        nodes[i].parameters = [np.random.uniform(3.0, 4.0), np.random.uniform(1.0, 2.0), 0.1]

    # 4个输出 Sink
    for i in range(4, 8):
        nodes[i] = TerminalSink(node_name=f'sink{i - 3}')
        nodes[i].node_acronym = f'CH_{i - 3}'

    # 2. 连线定义
    input_laser.src_port_idx = 0;
    input_laser.dst_port_idx = 0
    edges[(0, 1)] = input_laser

    base_len = 0.05  # 5cm 基础长度

    def make_wg(length, src_p, dst_p):
        wg = DelayWaveguide(length=length, ng=n_eff)
        wg.src_port_idx = src_p
        wg.dst_port_idx = dst_p
        return wg

    # === Level 1: Root -> Top & Bot ===
    # 上路 (去 Ch 1&2): 基础延迟
    edges[(1, 2)] = make_wg(base_len, 0, 0)

    # 下路 (去 Ch 3&4): 基础延迟 + 200ps (2个单位)
    # 这样 Ch 3&4 的起点比 Ch 1&2 晚 200ps
    edges[(1, 3)] = make_wg(base_len + 2 * delay_unit_len, 1, 0)

    # === Level 2: Top -> Ch 1&2 ===
    # Ch 1: 0ps (相对 Top) -> 总相对延迟 0ps
    edges[(2, 4)] = make_wg(base_len, 0, 0)
    # Ch 2: 100ps (相对 Top) -> 总相对延迟 100ps
    edges[(2, 5)] = make_wg(base_len + 1 * delay_unit_len, 1, 0)

    # === Level 2: Bot -> Ch 3&4 ===
    # Ch 3: 0ps (相对 Bot) -> 总相对延迟 200 + 0 = 200ps
    edges[(3, 6)] = make_wg(base_len, 0, 0)
    # Ch 4: 100ps (相对 Bot) -> 总相对延迟 200 + 100 = 300ps
    edges[(3, 7)] = make_wg(base_len + 1 * delay_unit_len, 1, 0)

    graph = Graph.init_graph(nodes, edges)
    return graph


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])
    io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io.init_save_dir(sub_path=None, unique_id=True)

    # --- 物理参数 ---
    p, q = (4, 1)

    # 1. 输入: 2.5 GHz (400ps)
    rep_freq = 2.5e9
    rep_t = 1.0 / rep_freq

    # 2. 目标交织间隔: 100ps (10 GHz 等效)
    target_interleave_dt = 100e-12

    pulse_width, peak_power = (3e-12, 1.0)

    # 采样点 (8192 足够)
    n_samples_debug = 2 ** 13
    propagator = Propagator(window_t=4e-9, n_samples=n_samples_debug, central_wl=1.55e-6)

    c = 299792458.0
    n_eff = 1.46
    v_group = c / n_eff

    # 100ps 对应的物理长度
    length_unit = v_group * target_interleave_dt

    # 估算基准飞行时间 (两层波导)
    avg_flight_time = (0.05 * 2) * n_eff / c
    flight_phase_shift = avg_flight_time / rep_t

    print(f"\n[Info] OTI 经典架构参数验证:")
    print(f"  输入: 2.5 GHz (周期 400ps)")
    print(f"  目标: 4 路交织 -> 10 GHz (间隔 100ps)")
    print(f"  波导单位长度 (100ps): {length_unit * 1000:.4f} mm")
    print(f"  基准飞行时间补偿: {avg_flight_time * 1e12:.2f} ps")

    # --- 激光器 ---
    input_laser = PulsedLaser(parameters_from_name={
        'pulse_width': pulse_width, 'peak_power': peak_power,
        't_rep': rep_t, 'pulse_shape': 'gaussian',
        'central_wl': 1.55e-6, 'train': True
    })
    input_laser.node_lock = True

    # --- 目标波形 (带补偿) ---
    targets = {}
    # 相位偏移计算: 100ps 是 400ps 周期的 0.25
    targets[4] = input_laser.get_pulse_train(propagator.t, pulse_width, rep_t, peak_power / 4,
                                             phase_shift=0.00 + flight_phase_shift)  # 0ps
    targets[5] = input_laser.get_pulse_train(propagator.t, pulse_width, rep_t, peak_power / 4,
                                             phase_shift=0.25 + flight_phase_shift)  # 100ps
    targets[6] = input_laser.get_pulse_train(propagator.t, pulse_width, rep_t, peak_power / 4,
                                             phase_shift=0.50 + flight_phase_shift)  # 200ps
    targets[7] = input_laser.get_pulse_train(propagator.t, pulse_width, rep_t, peak_power / 4,
                                             phase_shift=0.75 + flight_phase_shift)  # 300ps

    target_ids_map = {4: targets[4], 5: targets[5], 6: targets[6], 7: targets[7]}
    eval_ids = [4, 5, 6, 7]

    evaluator = PulseRepetition_multi(
        propagator, targets=target_ids_map, pulse_width=pulse_width,
        rep_t=rep_t, peak_power=peak_power, evaluation_nodes=eval_ids
    )

    # ==========================================
    # 5. 构建与优化 (单次运行)
    # ==========================================
    print("\n[Step 1] 构建经典 OTI 架构...")
    graph = build_classic_oti_graph(input_laser, n_eff, length_unit)

    print("\n[Step 2] 开始参数优化 (L-BFGS+GA)...")
    graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

    # 因为架构是固定的，优化非常快，30代足够完美收敛
    start_time = time.time()
    opt_graph, final_params, final_score, _ = parameters_optimize(
        graph, method='L-BFGS+GA',
        n_generations=20, population_size=20,
        verbose=True, num_cpus=1
    )
    elapsed = time.time() - start_time
    print(f"\n[Result] 优化完成! 耗时: {elapsed:.2f}s | Final Cost: {final_score:.6f}")

    # ==========================================
    # 6. 绘图验证
    # ==========================================
    print("\n[Step 3] 绘制波形图...")
    opt_graph.propagate(propagator)

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)

    # 自动定位中心
    center_idx = propagator.n_samples // 2
    center_t = propagator.t[center_idx]
    # 显示范围：前后 600ps (覆盖 1200ps，看3个周期)
    t_span = 600e-12

    labels = ["Ch1 (0ps)", "Ch2 (100ps)", "Ch3 (200ps)", "Ch4 (300ps)"]

    for i, nid in enumerate(eval_ids):
        out = opt_graph.measure_propagator(nid)
        if out is None: out = np.zeros_like(propagator.t)

        ax = axes[i]
        ax.plot(propagator.t * 1e12, power_(out), 'r-', linewidth=2.0, label='Optimized')
        ax.plot(propagator.t * 1e12, power_(targets[nid]), 'k--', alpha=0.6, label='Target')

        ax.set_ylabel(f'Power (W)')
        ax.set_title(f'{labels[i]} - Node {nid}')
        ax.legend(loc='upper right')

        ax.set_xlim((center_t - t_span) * 1e12, (center_t + t_span) * 1e12)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (ps)')
    fig.suptitle(f'Classic OTI Optimization Result\nScore: {final_score:.6f}')
    plt.tight_layout()

    save_name = 'classic_oti_result.png'
    io.save_fig(fig, save_name)
    print(f"结果已保存至: {save_name}")
    plt.show()