import sys
import pathlib
import os
import platform
import matplotlib.pyplot as plt
import psutil
import copy
import numpy as np

# 设置系统路径
parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

# 导入必要的模块
from lib.functions import InputOutput, parse_command_line_args
from lib.graph import Graph
from simulator.fiber.assets.propagator import Propagator
from simulator.fiber.assets.functions import power_
from simulator.fiber.node_types_subclasses.inputs import PulsedLaser
# from simulator.fiber.evaluator_subclasses.evaluator_pulserep import PulseRepetition
from simulator.fiber.evaluator_subclasses.evaluator_pulserep_new import PulseRepetition_multi
from algorithms.parameter_optimization import parameters_optimize
from lib.topology_generator import generate_topology, is_dag, is_physically_valid, check_connectivity
from lib.MatrixToGraph import convert_matrix_to_graph

plt.close('all')

if __name__ == '__main__':
    # ==========================================
    # 1. 基础配置
    # ==========================================
    options_cl = parse_command_line_args(sys.argv[1:])
    io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io.init_save_dir(sub_path=None, unique_id=True)

    # 拓扑参数
    N_MZM = 3
    N_DET = 4

    # 物理参数
    p, q = (4, 1) # 2倍频
    pulse_width, rep_t, peak_power = (3e-12, 1/10.0e9, 1.0)
    propagator = Propagator(window_t=4e-9, n_samples=2**15, central_wl=1.55e-6)

    # 定义激光源 (Input)
    input_laser = PulsedLaser(parameters_from_name={
        'pulse_width': pulse_width, 'peak_power': peak_power,
        't_rep': rep_t, 'pulse_shape': 'gaussian',
        'central_wl': 1.55e-6, 'train': True
    })
    input_laser.node_lock = True
    input_laser.protected = True

    targets = {}
    n = int(p / q)

    # TODO: targets incorrect
    targets['sink1'] = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width,
                                                   rep_t=rep_t * (p / q),
                                                   peak_power=peak_power * 1.0, phase_shift=0)
    targets['sink2'] = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width,
                                                   rep_t=rep_t * (p / q),
                                                   peak_power=peak_power * 1.0, phase_shift=1 / 2)
    targets['sink3'] = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width,
                                                   rep_t=rep_t * (p / q),
                                                   peak_power=peak_power * 1.0, phase_shift=1 / 4)
    targets['sink4'] = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width,
                                                   rep_t=rep_t * (p / q),
                                                   peak_power=peak_power * 1.0, phase_shift=3 / 4)

    evaluation_nodes = list(targets.keys())
    # evaluator = PulseRepetition(propagator, target, pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)
    evaluator = PulseRepetition_multi(propagator,
                                      targets=targets,
                                      pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power,
                                      evaluation_nodes=evaluation_nodes)
    # ==========================================
    # 2. 循环采样与优化 (The Loop)
    # ==========================================
    # 【修改点】设定要跑多少个样本
    N_SAMPLES = 10

    # 【修改点】记录历史最佳
    global_best_score = float('inf')
    global_best_graph = None
    global_best_params = None

    print(f"\n>>> 启动批量优化，计划尝试 {N_SAMPLES} 个样本...")

    for sample_idx in range(N_SAMPLES):
        print(f"\n-----------------------------------------------")
        print(f"Sample [{sample_idx + 1} / {N_SAMPLES}] 正在生成与优化...")

        # --- A. 寻找一个合法的矩阵 ---
        valid_A = None
        attempts = 0
        while attempts < 5000:
            attempts += 1
            temp_A = generate_topology(N_MZM, N_DET)
            if is_dag(temp_A.copy()) and is_physically_valid(temp_A, N_MZM, N_DET) and check_connectivity(temp_A, N_MZM, N_DET):
                valid_A = temp_A
                break

        if valid_A is None:
            print(f"  [Skip] 无法生成合法拓扑，跳过。")
            continue

        # --- B. 转换为 Graph ---
        try:
            graph = convert_matrix_to_graph(valid_A, N_MZM, N_DET)
        except Exception as e:
            print(f"  [Error] Graph转换失败: {e}")
            continue

        # --- C. 修复 Evaluator ID ---
        # 确保评估器盯着正确的 Sink 节点 (N_MZM + 1)
        target_sink_id = N_MZM + 1
        evaluator.evaluation_node = target_sink_id

        # --- D. 运行参数优化 ---
        try:
            # 初始化梯度
            graph.initialize_func_grad_hess(propagator, evaluator, exclude_locked=True)

            # 运行优化 (L-BFGS+GA)
            # 这里 population_size 和 generations 不需要设太大，因为我们要跑很多轮
            optimized_graph, final_params, final_score, log = parameters_optimize(
                graph,
                method='L-BFGS+GA',
                verbose=False,        # 关闭详细日志，保持清爽
                n_generations=10,     # 每个样本跑10代
                population_size=30    # 每个样本种群大小30
            )

            print(f"  >> 本次得分 (Cost): {final_score:.6f}")

            # --- E. 擂台赛：记录最优 ---
            if final_score < global_best_score:
                print(f"  ★ 新纪录！Cost 从 {global_best_score:.6f} 降至 {final_score:.6f}")
                global_best_score = final_score
                # 必须用 copy.deepcopy，否则 graph 对象可能会被后续循环覆盖或修改
                global_best_graph = copy.deepcopy(optimized_graph)
                global_best_params = final_params

        except Exception as e:
            print(f"  [Error] 优化计算出错: {e}")
            continue

    # ==========================================
    # 3. 最终结果展示 (只展示最好的)
    # ==========================================
    print("\n" + "="*50)

    if global_best_graph is None:
        print("所有样本均优化失败或未找到合法拓扑。")
    else:
        print(f"批量优化完成！")
        print(f"历史最佳 Cost: {global_best_score:.6f}")
        print(f"最优参数组合: {global_best_params}")

        # 绘图
        output_signal = global_best_graph.get_output_signal(propagator)

        fig, ax = plt.subplots(figsize=(10, 6))
        # 红色实线：最优输出
        ax.plot(propagator.t * 1e12, power_(output_signal), color='red', label='Best Output', linewidth=2)
        # 黑色虚线：目标
        ax.plot(propagator.t * 1e12, power_(target_signal), color='k', linestyle='--', label='Target', alpha=0.6)

        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Power (W)')
        ax.set_title(f'Best Result from {N_SAMPLES} Samples (Cost: {global_best_score:.4e})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_name = 'best_result_hof.png'
        io.save_fig(fig, save_name)
        print(f"最优结果图片已保存至: {save_name}")
        plt.show()