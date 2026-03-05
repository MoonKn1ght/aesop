import sys
import pathlib
import os
import platform
import matplotlib.pyplot as plt
import autograd.numpy as np
import copy
import time
import gc
import networkx as nx  # 新增：用于绘制拓扑图

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
from simulator.fiber.node_types import MultiPath,SinglePath
from simulator.fiber.assets.propagator import Propagator
from simulator.fiber.assets.functions import power_
from simulator.fiber.node_types_subclasses.inputs import PulsedLaser
from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink
from algorithms.parameter_optimization import parameters_optimize
from lib.topology_generator import generate_topology, is_dag, is_physically_valid, check_connectivity
from simulator.fiber.evaluator import Evaluator

plt.close('all')


# ==========================================
# 2. 自定义修复组件 (DelayWaveguide & MZM)
# ==========================================
class DelayWaveguide(SinglePath):
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


class MZM2x2Node(MultiPath):
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
        self.parameter_locks = [True, False, False, True, False, True]  # 解锁偏置、幅度、相位供CMA优化
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


### 评估器 (保留你的旧版 PADC Evaluator)
class PADC_Evaluator(Evaluator):
    def __init__(self, propagator, targets, evaluation_nodes):
        super().__init__()
        self.propagator = propagator
        self.targets = targets
        self.evaluation_nodes = evaluation_nodes
        self.masks = {}
        for nid, target_field in self.targets.items():
            p_tgt = power_(target_field)
            max_p = np.max(p_tgt) + 1e-15
            norm_tgt = p_tgt / max_p
            self.masks[nid] = (norm_tgt < 0.05).astype(float)

    def evaluate_graph(self, graph, propagator=None):
        return self.evaluate(graph)

    def evaluate(self, graph):
        total_cost = 0.0
        for node in self.evaluation_nodes:
            signal = graph.measure_propagator(node)
            target = self.targets[node]
            if signal is None:
                total_cost += 5000.0
                continue
            p_signal = np.abs(signal) ** 2
            p_target = np.abs(target) ** 2
            max_sig = np.max(p_signal) + 1e-15
            max_tar = np.max(p_target) + 1e-15
            p_sig_norm = p_signal / max_sig
            p_tar_norm = p_target / max_tar
            shape_error = np.mean((p_sig_norm - p_tar_norm) ** 2)
            noise_content = p_sig_norm * self.masks[node]
            penalty = np.mean(noise_content ** 2) * 100.0
            total_cost += (shape_error + penalty)
        return total_cost


# ==========================================
# 3. 增强版图转换与绘图函数
# ==========================================
def convert_matrix_to_graph_enhanced(matrix, N_MZM, N_DET, input_laser, n_eff, length_unit, output_freq):
    nodes = {}
    edges = {}

    nodes[0] = TerminalSource()
    nodes[0].node_acronym = 'SRC'
    for i in range(1, N_MZM + 1):
        nodes[i] = MZM2x2Node()
        nodes[i].node_acronym = f'MZM_{i}'
        # 【修复点】：补充完整的6个参数，射频频率设为输出频率(2.5G)，初始相位设为0
        nodes[i].parameters = [3.5, 1.75, 1.0, output_freq, 0.0, 0.1]

    for i in range(1, N_DET + 1):
        sink_id = N_MZM + i
        nodes[sink_id] = TerminalSink(node_name=f'sink{i}')
        nodes[sink_id].node_acronym = f'SINK_{i}'

    port_tracker = {i: {'in': 0, 'out': 0} for i in range(matrix.shape[0])}
    base_len = 0.05

    for src in range(matrix.shape[0]):
        for dst in range(matrix.shape[0]):
            if matrix[src, dst] == 1:
                current_out_port = port_tracker[src]['out']
                if src == 0 and current_out_port >= 1: continue
                if 1 <= src <= N_MZM and current_out_port >= 2: continue

                src_port = port_tracker[src]['out']
                dst_port = port_tracker[dst]['in']
                port_tracker[src]['out'] += 1
                port_tracker[dst]['in'] += 1

                if src == 0:
                    comp = copy.deepcopy(input_laser)
                    comp.src_port_idx = src_port
                    comp.dst_port_idx = dst_port
                    edges[(src, dst)] = comp
                else:
                    delay_steps = np.random.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.2, 0.2])
                    total_len = base_len + delay_steps * length_unit
                    wg = DelayWaveguide()
                    wg.parameters = [total_len * n_eff / 299792458.0]  # 直接转换为时间延迟
                    wg.src_port_idx = src_port
                    wg.dst_port_idx = dst_port
                    edges[(src, dst)] = wg

    graph = Graph.init_graph(nodes, edges)
    return graph


def plot_and_save_topology(matrix, N_MZM, N_DET, sample_idx, save_dir):
    """绘制并保存单个拓扑结构图"""
    G = nx.DiGraph(matrix)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(6, 4))
    colors = []
    labels = {}
    for node in G.nodes():
        if node == 0:
            colors.append('lightgreen');
            labels[node] = 'SRC'
        elif 1 <= node <= N_MZM:
            colors.append('lightblue');
            labels[node] = f'MZM{node}'
        else:
            colors.append('salmon');
            labels[node] = f'CH{node - N_MZM}'
    nx.draw(G, pos, with_labels=True, labels=labels, node_color=colors, node_size=800, font_size=8, arrows=True)
    plt.title(f"Topology Layout - Sample #{sample_idx}")
    plt.savefig(os.path.join(save_dir, f"topology_sample_{sample_idx}.png"), dpi=150)
    plt.close()


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
    output_freq = 1.0 / (rep_t * p)  # 2.5 GHz

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

    # 【替换为旧版的 PADC 评估器】
    evaluator = PADC_Evaluator(
        propagator, targets=target_ids_map, evaluation_nodes=eval_ids
    )

    # ==========================================
    # 5. 批量生成与优化循环
    # ==========================================
    N_SAMPLES = 5  # 先生成 5 个样本测试
    results_record = []

    print(f"\n>>> 开始拓扑进化实验: 目标 {N_SAMPLES} 个样本")
    print(f"    当前设置: n_samples={n_samples_debug}, CMA Optimizer")

    for sample_idx in range(1, N_SAMPLES + 1):
        print(f"\n[{sample_idx}/{N_SAMPLES}] 正在尝试生成合法的拓扑...")

        valid_A = None
        for attempt in range(2000):
            temp_A = generate_topology(N_MZM, N_DET)
            if is_dag(temp_A) and is_physically_valid(temp_A, N_MZM, N_DET) and check_connectivity(temp_A, N_MZM,
                                                                                                   N_DET):
                valid_A = temp_A
                print(f"  >> 成功获得合法拓扑 (尝试 {attempt + 1} 次)")
                break

        if valid_A is None:
            print("  [Skip] 无法生成，跳过。")
            continue

        # 【功能 1】保存单独的拓扑图
        plot_and_save_topology(valid_A, N_MZM, N_DET, sample_idx, io.save_path)

        try:
            graph = convert_matrix_to_graph_enhanced(valid_A, N_MZM, N_DET, input_laser, n_eff, length_unit,
                                                     output_freq)
        except Exception as e:
            print(f"  [Error] Graph conversion failed: {e}")
            continue

        try:
            print("  >> 开始参数优化...")
            start_time = time.time()
            graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)


            # opt_graph, final_params, final_score, _ = parameters_optimize(
            #     graph, method = 'L-BFGS+GA',
            #     n_generations=30, population_size=30,
            #     verbose=False, num_cpus=1
            # )
            # 【使用 CMA 进行优化】
            opt_graph, final_params, final_score, _ = parameters_optimize(
                graph,
                method='CMA',
                verbose=False,
                num_cpus=1
            )
            elapsed = time.time() - start_time
            print(f"  >> 优化完成! 耗时: {elapsed:.2f}s | Score: {final_score:.6f}")

            # 【功能 2】保存单独的波形结果图
            opt_graph.propagate(propagator)
            fig, axes = plt.subplots(N_DET, 1, figsize=(8, 10), sharex=True)
            if N_DET == 1: axes = [axes]
            center_t = propagator.t[propagator.n_samples // 2]

            for i, nid in enumerate(eval_ids):
                out = opt_graph.measure_propagator(nid)
                if out is None: out = np.zeros_like(propagator.t)
                axes[i].plot(propagator.t * 1e12, power_(out), 'r-', label='Demuxed')
                axes[i].plot(propagator.t * 1e12, power_(target_ids_map[nid]), 'k--', alpha=0.5, label='Target')
                axes[i].set_ylabel(f'CH {i + 1}')
                axes[i].set_xlim((center_t - 400e-12) * 1e12, (center_t + 400e-12) * 1e12)

            axes[0].legend(loc='upper right', fontsize='x-small')
            fig.suptitle(f'Topology #{sample_idx} Waveform (Cost: {final_score:.4f})')
            io.save_fig(fig, f'sample_{sample_idx}_waveform.png')
            plt.close(fig)

            results_record.append({
                'id': sample_idx, 'score': final_score,
                'graph': copy.deepcopy(opt_graph), 'matrix': valid_A
            })

            del opt_graph;
            del graph;
            gc.collect()

        except Exception as e:
            print(f"  [Error] Optimization failed: {e}")
            continue

    # ==========================================
    # 6. 【功能 3】Top 3 汇总大图绘制
    # ==========================================
    if len(results_record) > 0:
        print("\n" + "=" * 50)
        print("绘制 Top 3 优秀变异体对比图...")
        results_record.sort(key=lambda x: x['score'])
        top_k = min(3, len(results_record))

        # 创建网格：行数为 top_k，列数为通道数(4)
        fig, axes = plt.subplots(top_k, N_DET, figsize=(16, 3 * top_k), sharex=True)
        if top_k == 1: axes = [axes]

        for idx in range(top_k):
            res = results_record[idx]
            opt_g = res['graph']
            opt_g.propagate(propagator)

            for ch_i, nid in enumerate(eval_ids):
                ax = axes[idx][ch_i]
                out = opt_g.measure_propagator(nid)
                if out is None: out = np.zeros_like(propagator.t)

                p_out = power_(out)
                p_tgt = power_(target_ids_map[nid])

                ax.plot(propagator.t * 1e12, p_out / (np.max(p_out) + 1e-15), 'r-', label='Demuxed')
                ax.plot(propagator.t * 1e12, p_tgt / (np.max(p_tgt) + 1e-15), 'k--', alpha=0.3, label='Target')

                if ch_i == 0:
                    ax.set_ylabel(f"Rank {idx + 1}\n(Cost {res['score']:.2f})", fontweight='bold')
                if idx == 0:
                    ax.set_title(f"Channel {ch_i + 1}")
                ax.set_xlim((center_t - 300e-12) * 1e12, (center_t + 300e-12) * 1e12)

        plt.tight_layout()
        io.save_fig(fig, 'Top3_Best_Topologies_Comparison.png')
        plt.close(fig)

        print(f"所有结果已成功保存至: {io.save_path}")