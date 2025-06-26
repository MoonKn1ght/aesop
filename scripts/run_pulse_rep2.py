"""
?????IM
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
os.environ["PYTHONPATH"] = parent_dir + sep + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)

# various imports
import matplotlib.pyplot as plt
import psutil

from lib.functions import InputOutput

from lib.graph import Graph
from simulator.fiber.assets.propagator import Propagator

from simulator.fiber.evolver import HessianProbabilityEvolver, ProbabilityLookupEvolver, OperatorBasedProbEvolver
from simulator.fiber.node_types_subclasses import *

from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink

from simulator.fiber.evaluator_subclasses.evaluator_pulserep import PulseRepetition, PulseRepetition_dual

from simulator.fiber.node_types_subclasses.inputs import PulsedLaser
from simulator.fiber.node_types_subclasses.outputs import MeasurementDevice
from simulator.fiber.node_types_subclasses.single_path import PhaseModulator,IntensityModulator
from simulator.fiber.node_types_subclasses.multi_path import VariablePowerSplitter,DualOutputMZM

from algorithms.topology_optimization import topology_optimization, save_hof, plot_hof

from lib.functions import parse_command_line_args

plt.close('all')
if __name__ == '__main__':
    options_cl = parse_command_line_args(sys.argv[1:])
    print(options_cl.verbose)
    io = InputOutput(directory=options_cl.dir, verbose=options_cl.verbose)
    io.init_save_dir(sub_path=None, unique_id=True)
    io.save_machine_metadata(io.save_path)

    ga_opts = {'n_generations': 3,
               'n_population': 1,
               'n_hof': 1,
               'verbose': options_cl.verbose,
               'num_cpus': psutil.cpu_count()-1}

    propagator = Propagator(window_t=4e-9, n_samples=2**15, central_wl=1.55e-6)
    pulse_width, rep_t, peak_power = (3e-12, 1/10.0e9, 1.0)

    p, q = (2, 1)
    input_laser = PulsedLaser(parameters_from_name={'pulse_width': pulse_width, 'peak_power': peak_power,
                                                    't_rep': rep_t, 'pulse_shape': 'gaussian',
                                                    'central_wl': 1.55e-6, 'train': True})
    input_laser.node_lock = True
    input_laser.protected = True

    input = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)  # ???????
    # target = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width * (p / q), rep_t=rep_t * (p / q), peak_power=peak_power * (p / q))   #???? 2?
    # target1 = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width * (p / q), rep_t=rep_t * (p / q),
    #                                      peak_power=peak_power * (p / q), phase_shift=0)
    # target2 = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width * (p / q), rep_t=rep_t * (p / q),
    #                                      peak_power=peak_power * (p / q),phase_shift=0.5)   # shift 0.5T
    target1 = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width * (p / q), rep_t=rep_t * (p / q),
                                          peak_power=peak_power * 1.0, phase_shift=0)
    target2 = input_laser.get_pulse_train(propagator.t, pulse_width=pulse_width * (p / q), rep_t=rep_t * (p / q),
                                          peak_power=peak_power * 1.0, phase_shift=0.5)  # shift 0.5T

    # evaluator = PulseRepetition(propagator, target, pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)
    evaluator = PulseRepetition_dual(propagator,
                                     targets={'sink1': np.array(target1),
                                            'sink2': np.array(target2)  # ?????numpy??
                                            },
                                     pulse_width=pulse_width, rep_t=rep_t, peak_power=peak_power)

    evolver = HessianProbabilityEvolver(verbose=False)
    # evolver = OperatorBasedProbEvolver(verbose=False)

    md = MeasurementDevice()
    md.protected = True

    nodes = {'source': TerminalSource(),
             0: VariablePowerSplitter(),
             1: VariablePowerSplitter(),
             'sink1': TerminalSink(node_name='sink1'),
             'sink2': TerminalSink(node_name='sink2')}
    edges = {('source', 0): input_laser,
             (0,1): IntensityModulator(),
             (1, 'sink1'): md,
             (0, 'sink2'): md
             }
    graph = Graph.init_graph(nodes=nodes, edges=edges)
    update_rule = 'tournament'

    hof, log = topology_optimization(copy.deepcopy(graph), propagator, evaluator, evolver, io,
                                     ga_opts=ga_opts, local_mode=False, update_rule=update_rule,
                                     parameter_opt_method='ADAM+GA',
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

    io.save_fig(fig, 'topology_log.png')