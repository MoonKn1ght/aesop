"""
4 output system
"""

"""

# place main ASOPE directory on the path which will be accessed by all ray workers
import sys
import pathlib
import os
import platform
import copy
import numpy as np

from testing.test_hessian_decisions import evaluator

parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'
# os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.gpip install et("PYTHONPATH", "")
os.environ["PYTHONPATH"] = parent_dir + os.sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)
"""

import sys
import pathlib
import os
import platform

# 首先设置路径 - 必须在所有导入之前
parent_dir = str(pathlib.Path(__file__).absolute().parent.parent)
sep = ';' if platform.system() == 'Windows' else ':'

# 优化1: 使用正确的环境变量分隔符
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")

# 优化2: 使用insert(0)而不是append，确保优先级最高
sys.path.insert(0, parent_dir)

# 优化3: 添加路径验证输出
print(f"项目根目录: {parent_dir}")
print(f"系统路径: {sys.path[:2]}...")  # 只打印前两项避免过长输出

# 现在导入标准库
import copy
import numpy as np

# 优化4: 添加导入异常处理
try:
    from testing.test_hessian_decisions import evaluator
    print("成功导入 testing 模块")
except ImportError as e:
    print(f"导入 testing 模块失败: {e}")
    print("当前系统路径:")
    for p in sys.path:
        print(f" - {p}")
    sys.exit(1)  # 导入失败时退出程序

# 其他导入保持不变...


# various imports
import matplotlib.pyplot as plt
import psutil

from lib.functions import InputOutput

from lib.graph import Graph
from simulator.fiber.assets.propagator import Propagator

from simulator.fiber.evolver import HessianProbabilityEvolver, ProbabilityLookupEvolver, OperatorBasedProbEvolver
from simulator.fiber.node_types_subclasses import *

from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink

from simulator.fiber.evaluator_subclasses.evaluator_pulserep import PulseRepetition, PulseRepetition_dual,PulseRepetition_multi

from simulator.fiber.node_types_subclasses.inputs import PulsedLaser
from simulator.fiber.node_types_subclasses.outputs import MeasurementDevice
from simulator.fiber.node_types_subclasses.single_path import PhaseModulator,IntensityModulator,OpticalAmplifier
from simulator.fiber.node_types_subclasses.multi_path import VariablePowerSplitter


from algorithms.topology_optimization import topology_optimization, save_hof, plot_hof

from lib.functions import parse_command_line_args

plt.close('all')
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])

    # ==================
    # TODO: addition
    # import os
    # from pathlib import Path
    #
    # # 如果目录是根目录下的 /asope_data，重定向到用户主目录
    # if options_cl.dir in ["/asope_data", None]:
    #     user_home = Path.home()
    #     options_cl.dir = str(user_home / "asope_data")
    #
    #     # 确保目录存在
    #     user_data_dir = Path(options_cl.dir)
    #     user_data_dir.mkdir(parents=True, exist_ok=True)
    #     print(f"使用用户目录: {user_data_dir}")

    # ==================

    print(options_cl.verbose)
    io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io.init_save_dir(sub_path=None, unique_id=True)
    io.save_machine_metadata(io.save_path)

    ga_opts = {'n_generations': 10,
               'n_population': 10,
               'n_hof': 5,
               'verbose': options_cl.verbose,
               'num_cpus': psutil.cpu_count()-1}

    propagator = Propagator(window_t=2e-9, n_samples=2**15, central_wl=1.55e-6)
    pulse_width, rep_t, peak_power = (3e-12, 1/10.0e9, 1.0)

    p, q = (4, 1)
    input_laser = PulsedLaser(parameters_from_name={'pulse_width': pulse_width, 'peak_power': peak_power,
                                                    't_rep': rep_t, 'pulse_shape': 'gaussian',
                                                    'central_wl': 1.55e-6, 'train': True})
    input_laser.node_lock = True
    input_laser.protected = True

    input = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)
    targets = {}
    n=int(p/q)

    #TODO: targets incorrect
    targets['sink1'] = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width ,
                                                   rep_t=rep_t * (p / q),
                                                   peak_power=peak_power * 1.0, phase_shift=0)
    targets['sink2'] = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width ,
                                                   rep_t=rep_t * (p / q),
                                                   peak_power=peak_power * 1.0, phase_shift=1/2)
    targets['sink3'] = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width ,
                                                   rep_t=rep_t * (p / q),
                                                   peak_power=peak_power * 1.0, phase_shift=1/4)
    targets['sink4'] = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width ,
                                                   rep_t=rep_t * (p / q),
                                                   peak_power=peak_power * 1.0, phase_shift=3/4)

    evaluation_nodes = list(targets.keys())
    # evaluator = PulseRepetition(propagator, target, pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)
    evaluator = PulseRepetition_multi(propagator,
                                     targets=targets,
                                     pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power,
                                       evaluation_nodes=evaluation_nodes)

    evolver = HessianProbabilityEvolver(verbose=False)
    # evolver = OperatorBasedProbEvolver(verbose=False)

    md = MeasurementDevice()
    md.protected = True

    # nodes = {'source': TerminalSource(),
    #          0: DualOutputMZM(parameters=[np.pi,2.5e9,0]),
    #          1: DualOutputMZM(parameters=[np.pi,1.25e9,0]),
    #          2: DualOutputMZM(parameters=[np.pi,1.25e9,np.pi/2]),
    #          'sink1': TerminalSink(node_name='sink1'),
    #          'sink2': TerminalSink(node_name='sink2'),
    #          'sink3': TerminalSink(node_name='sink3'),
    #          'sink4': TerminalSink(node_name='sink4')
    #          }
    # edges = {('source', 0): input_laser,
    #          (0,1):OpticalAmplifier(),
    #          (0,2):OpticalAmplifier(),
    #          (1, 'sink1'): md,
    #          (1, 'sink2'): md,
    #          (2, 'sink3'): md,
    #          (2, 'sink4'): md
    #          }

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             1: VariablePowerSplitter(),
             2: VariablePowerSplitter(),
             3: VariablePowerSplitter(),
             4: VariablePowerSplitter(),
             5: VariablePowerSplitter(),
             'sink1': TerminalSink(node_name='sink1'),
             'sink2': TerminalSink(node_name='sink2'),
             'sink3': TerminalSink(node_name='sink3'),
             'sink4': TerminalSink(node_name='sink4')
             }
    edges = {('source', 0): input_laser,
             (0,1,0): PhaseModulator(),
             (0,1,1): PhaseModulator(),
             (1,2): OpticalAmplifier(),
             (1,4): OpticalAmplifier(),
             (2, 3, 0): PhaseModulator(),
             (2, 3, 1): PhaseModulator(),
             (4, 5, 0): PhaseModulator(),
             (4, 5, 1): PhaseModulator(),

             (3, 'sink1'): md,
             (3, 'sink2'): md,
             (5, 'sink3'): md,
             (5, 'sink4'): md
             }
    graph = Graph.init_graph(nodes=nodes, edges=edges)
    update_rule = 'tournament'

    hof, log = topology_optimization(copy.deepcopy(graph), propagator, evaluator, evolver, io,
                                     ga_opts=ga_opts, local_mode=False, update_rule=update_rule,
                                     parameter_opt_method='L-BFGS+GA',
                                     include_dashboard=False, crossover_maker=None,
                                     ged_threshold_value=0.8)

    save_hof(hof, io)
    plot_hof(hof, propagator, evaluator, io)

    fig, ax = plt.subplots(1, 1, figsize=[5, 3])
    ax.fill_between(log['generation'], log['best'], log['mean'], color='grey', alpha=0.2)
    ax.plot(log['generation'], log['best'], label='Best')
    ax.plot(log['generation'], log['mean'], label='Population mean')
    ax.plot(log['generation'], log['minimum'], color='darkgrey', label='Population minimum')
    ax.plot(log['generation'], log['maximum'], color='black', label='Population maximum')
    ax.set(xlabel='Generation', ylabel='Cost')
    ax.legend()
    plt.show()
    io.save_fig(fig, 'topology_log.png')