import sys
import pathlib
import os
import platform
import matplotlib.pyplot as plt
import autograd.numpy as np
import copy
import time
import gc  # 引入垃圾回收

# ==========================================
# 0. 环境路径设置
# ==========================================
parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

# ==========================================
# 1. 模块导入
# ==========================================
from lib.functions import InputOutput, parse_command_line_args
from lib.graph import Graph
from lib.base_classes import NodeType
from simulator.fiber.node_types import MultiPath
from simulator.fiber.assets.propagator import Propagator
from simulator.fiber.assets.functions import power_
from simulator.fiber.node_types_subclasses.inputs import PulsedLaser
from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink
from simulator.fiber.evaluator_subclasses.evaluator_pulserep_new import PulseRepetition_multi
from algorithms.parameter_optimization import parameters_optimize
from lib.topology_generator import generate_topology, is_dag, is_physically_valid, check_connectivity

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
        if np.ndim(field) > 1: field = np.reshape(field, (-1,))  # 维度清洗

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
# 3. 增强版图转换函数 (带端口溢出保护)
# ==========================================
def convert_matrix_to_graph_enhanced(matrix, N_MZM, N_DET, input_laser, n_eff, length_unit):
    nodes = {}
    edges = {}

    nodes[0] = TerminalSource();
    nodes[0].node_acronym = 'SRC'
    for i in range(1, N_MZM + 1):
        nodes[i] = MZM2x2Node_Fixed()
        nodes[i].node_acronym = f'MZM_{i}'
        nodes[i].parameters = [np.random.uniform(3.0, 4.0), np.random.uniform(1.0, 2.0), 0.1]
    for i in range(1, N_DET + 1):
        sink_id = N_MZM + i
        nodes[sink_id] = TerminalSink(node_name=f'sink{i}')
        nodes[sink_id].node_acronym = f'SINK_{i}'

    port_tracker = {i: {'in': 0, 'out': 0} for i in range(matrix.shape[0])}
    base_len = 0.05

    for src in range(matrix.shape[0]):
        for dst in range(matrix.shape[0]):
            if matrix[src, dst] == 1:
                # 端口检查
                current_out_port = port_tracker[src]['out']
                if src == 0 and current_out_port >= 1: continue  # Source 限1
                if 1 <= src <= N_MZM and current_out_port >= 2: continue  # MZM 限2

                src_port = port_tracker[src]['out']
                dst_port = port_tracker[dst]['in']
                port_tracker[src]['out'] += 1
                port_tracker[dst]['in'] += 1

                if src == 0:
                    comp = copy.deepcopy(input_laser)
                    comp.src_port_idx = src_port;
                    comp.dst_port_idx = dst_port
                    edges[(src, dst)] = comp
                else:
                    delay_steps = np.random.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.2, 0.2])
                    total_len = base_len + delay_steps * length_unit
                    wg = DelayWaveguide(length=total_len, ng=n_eff)
                    wg.src_port_idx = src_port;
                    wg.dst_port_idx = dst_port
                    edges[(src, dst)] = wg

    graph = Graph.init_graph(nodes, edges)
    return graph


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])
    io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io.init_save_dir(sub_path=None, unique_id=True)

    N_MZM, N_DET = 3, 4
    p, q = (4, 1)
    pulse_width, rep_t, peak_power = (3e-12, 1 / 10.0e9, 1.0)

    # 【优化 1】降低采样点数，大幅提升速度，防止卡死
    # 2**12 = 4096，2**13 = 8192。先用 4096 跑通流程。
    n_samples_debug = 2 ** 12
    propagator = Propagator(window_t=4e-9, n_samples=n_samples_debug, central_wl=1.55e-6)

    c = 299792458.0
    n_eff = 1.46
    v_group = c / n_eff
    target_dt = rep_t / p
    length_unit = v_group * target_dt

    avg_flight_time = (0.05 * 2) * n_eff / c
    flight_phase_shift = avg_flight_time / rep_t
    print(f"System Flight Time Est: {avg_flight_time * 1e12:.1f} ps (Phase Shift: {flight_phase_shift:.2f})")

    input_laser = PulsedLaser(parameters_from_name={
        'pulse_width': pulse_width, 'peak_power': peak_power,
        't_rep': rep_t, 'pulse_shape': 'gaussian',
        'central_wl': 1.55e-6, 'train': True
    })
    input_laser.node_lock = True

    targets = {}
    for i in range(1, 5):
        phase = (i - 1) * 0.25 + flight_phase_shift
        targets[f'sink{i}'] = input_laser.get_pulse_train(
            propagator.t, pulse_width, rep_t, peak_power / p, phase_shift=phase
        )

    target_ids_map = {}
    eval_ids = []
    id_name_map = {}
    for i in range(1, 5):
        sid = N_MZM + i
        target_ids_map[sid] = targets[f'sink{i}']
        eval_ids.append(sid)
        id_name_map[sid] = f'sink{i}'

    evaluator = PulseRepetition_multi(
        propagator, targets=target_ids_map, pulse_width=pulse_width,
        rep_t=rep_t, peak_power=peak_power, evaluation_nodes=eval_ids
    )

    # ==========================================
    # 5. 批量生成与优化循环 (防卡死版)
    # ==========================================
    N_SAMPLES = 10      # TODO
    results_record = []

    print(f"\n>>> 开始批量实验: 目标 {N_SAMPLES} 个样本")
    print(f"    当前设置: n_samples={n_samples_debug}, num_cpus=1")

    for sample_idx in range(1, N_SAMPLES + 1):
        print(f"\n[{sample_idx}/{N_SAMPLES}] 正在尝试生成合法的拓扑...")

        # A. 生成拓扑 (带超时打印)
        valid_A = None
        max_attempts = 2000
        for attempt in range(max_attempts):
            temp_A = generate_topology(N_MZM, N_DET)
            # 简单检查
            if is_dag(temp_A) and is_physically_valid(temp_A, N_MZM, N_DET) and check_connectivity(temp_A, N_MZM,
                                                                                                   N_DET):
                valid_A = temp_A
                print(f"  >> 成功! 在第 {attempt + 1} 次尝试生成了合法拓扑。")
                break

            # 每 500 次打印一下，证明没死机
            if (attempt + 1) % 500 == 0:
                print(f"  ...已尝试 {attempt + 1} 次...")

        if valid_A is None:
            print("  [Skip] 无法生成合法拓扑，跳过本次。")
            continue

        # B. 转换图
        try:
            graph = convert_matrix_to_graph_enhanced(valid_A, N_MZM, N_DET, input_laser, n_eff, length_unit)
        except Exception as e:
            print(f"  [Error] Graph conversion failed: {e}")
            continue

        # C. 优化
        try:
            print("  >> 开始优化参数 (L-BFGS+GA)...")
            start_time = time.time()
            graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

            # 【优化 2】减少代数，快速验证
            # TODO: set parameters optimization para
            opt_graph, final_params, final_score, _ = parameters_optimize(
                graph, method='L-BFGS+GA',
                n_generations=30, population_size=20,  # 少量个体
                verbose=False, num_cpus=1
            )
            elapsed = time.time() - start_time
            print(f"  >> 优化完成! 耗时: {elapsed:.2f}s | Score: {final_score:.6f}")

            # D. 保存与绘图
            opt_graph.propagate(propagator)

            fig, axes = plt.subplots(N_DET, 1, figsize=(8, 8), sharex=True)
            if N_DET == 1: axes = [axes]

            center_t = propagator.t[propagator.n_samples // 2]

            for i, nid in enumerate(eval_ids):
                out = opt_graph.measure_propagator(nid)
                if out is None: out = np.zeros_like(propagator.t)

                axes[i].plot(propagator.t * 1e12, power_(out), 'r-', label='Opt')
                axes[i].plot(propagator.t * 1e12, power_(target_ids_map[nid]), 'k--', alpha=0.5, label='Target')
                axes[i].set_ylabel(f'Sink {i + 1}')
                axes[i].set_xlim((center_t - 200e-12) * 1e12, (center_t + 200e-12) * 1e12)

            axes[0].legend(loc='upper right', fontsize='x-small')
            fig.suptitle(f'Sample #{sample_idx} (Score: {final_score:.4f})')

            fig_name = f'sample_{sample_idx}_waveform.png'
            io.save_fig(fig, fig_name)
            plt.close(fig)

            results_record.append({
                'id': sample_idx, 'score': final_score,
                'graph': copy.deepcopy(opt_graph), 'params': final_params, 'matrix': valid_A
            })

            # 【优化 3】强制垃圾回收，防止内存泄漏
            del opt_graph
            del graph
            gc.collect()

        except Exception as e:
            print(f"  [Error] Optimization failed: {e}")
            import traceback

            traceback.print_exc()
            continue

    # ==========================================
    # 6. Top-K 展示
    # ==========================================
    print("\n" + "=" * 50)
    print("实验结束，Top 3 结果展示")

    results_record.sort(key=lambda x: x['score'])

    top_k = min(3, len(results_record))
    for i in range(top_k):
        res = results_record[i]
        print(f"Rank {i + 1}: Sample #{res['id']} | Score: {res['score']:.6f}")
        print(f"  Matrix:\n{res['matrix']}")

    print(f"\n所有样本波形图已保存至: {io.save_path}")