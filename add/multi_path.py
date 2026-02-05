"""

"""

from pint import UnitRegistry
unit = UnitRegistry()
import autograd.numpy as np
import matplotlib.pyplot as plt

from ..node_types import MultiPath

from ..assets.decorators import register_node_types_all
from ..assets.functions import power_, psd_, fft_, ifft_, ifft_shift_, dB_to_amplitude_ratio

# @register_node_types_all
# class VariablePowerSplitter(MultiPath):
#     node_acronym = 'BS'
#     number_of_parameters = 0
#     node_lock = False
#
#     def __init__(self, **kwargs):
#
#         self.upper_bounds = []
#         self.lower_bounds = []
#         self.data_types = []
#         self.step_sizes = []
#         self.parameter_imprecisions = []
#         self.parameter_units = []
#         self.parameter_locks = []
#         self.parameter_names = []
#         self.default_parameters = []
#         self.parameter_symbols = []
#         super().__init__(**kwargs)
#
#     def update_attributes(self, num_inputs, num_outputs):
#         num_parameters = num_outputs - 1
#         self.number_of_parameters = num_parameters
#
#         self.upper_bounds = [1.0] * num_parameters
#         self.lower_bounds = [0.0] * num_parameters
#         self.data_types = ['float'] * num_parameters
#         self.step_sizes = [None] * num_parameters
#         self.parameter_imprecisions = [0.05] * num_parameters
#         self.parameter_units = [None] * num_parameters
#         self.parameter_locks = [False] * num_parameters
#         self.parameter_names = [f'ratio-{i}' for i in range(num_parameters)]
#         self.default_parameters = [1 - 1 / i for i in range(num_parameters+1, 1, -1)]
#         self.parameter_symbols = [f'x_{i}' for i in range(num_parameters)]
#
#         self.parameters = self.default_parameters
#         return
#
#     def propagate(self, states, propagator, num_inputs, num_outputs, save_transforms=False):
#         DEBUG = False
#         if DEBUG: print(f'PowerSplitter', num_inputs, num_outputs, len(self.parameters), self.parameters)
#
#         a = self.parameters
#         w = np.array([(1 - an) * np.product(a[:n]) for n, an in enumerate(a)] + [np.product(a)])
#         if DEBUG: print(a, w, sum(w))
#
#         i, j = np.arange(0, num_inputs, 1), np.arange(0, num_outputs, 1)
#         I, J = np.meshgrid(i, j)
#         I = I / num_inputs
#         J = J / num_outputs
#         X = np.matmul(np.expand_dims(np.sqrt(np.array(w)), axis=1), np.expand_dims(np.ones_like(i), axis=0))
#         if DEBUG: print(X)
#
#         S = X / np.sqrt(num_outputs) * np.exp(-1j * 2 * np.pi * (I + J))
#         if DEBUG: print(S)
#         states_tmp = np.stack(states, 1)
#
#         states_scattered = np.matmul(S, states_tmp)
#         states_scattered_lst = [states_scattered[:, i, :] for i in range(states_scattered.shape[1])]
#         return states_scattered_lst

####BS建模修正
@register_node_types_all
class VariablePowerSplitter(MultiPath):
    node_acronym = 'BS'
    number_of_parameters = 0
    node_lock = False

    def __init__(self, **kwargs):

        self.upper_bounds = []
        self.lower_bounds = []
        self.data_types = []
        self.step_sizes = []
        self.parameter_imprecisions = []
        self.parameter_units = []
        self.parameter_locks = []
        self.parameter_names = []
        self.default_parameters = []
        self.parameter_symbols = []
        super().__init__(**kwargs)

    def update_attributes(self, num_inputs, num_outputs):
        num_parameters = num_outputs - 1
        self.number_of_parameters = num_parameters

        self.upper_bounds = [1.0] * num_parameters
        self.lower_bounds = [0.0] * num_parameters
        self.data_types = ['float'] * num_parameters
        self.step_sizes = [None] * num_parameters
        self.parameter_imprecisions = [0.05] * num_parameters
        self.parameter_units = [None] * num_parameters
        self.parameter_locks = [False] * num_parameters
        self.parameter_names = [f'ratio-{i}' for i in range(num_parameters)]
        self.default_parameters = [1 - 1 / i for i in range(num_parameters+1, 1, -1)]
        self.parameter_symbols = [f'x_{i}' for i in range(num_parameters)]

        self.parameters = self.default_parameters
        return

    def propagate(self, states, propagator, num_inputs, num_outputs, save_transforms=False):
        DEBUG = False
        if DEBUG: print(f'PowerSplitter', num_inputs, num_outputs, len(self.parameters), self.parameters)

        a = self.parameters
        w = np.array([(1 - an) * np.product(a[:n]) for n, an in enumerate(a)] + [np.product(a)])
        if DEBUG: print(a, w, sum(w))

        i, j = np.arange(0, num_inputs, 1), np.arange(0, num_outputs, 1)
        I, J = np.meshgrid(i, j)
        I = I / num_inputs
        J = J / num_outputs
        X = np.matmul(np.expand_dims(np.sqrt(np.array(w)), axis=1), np.expand_dims(np.ones_like(i), axis=0))
        if DEBUG: print(X)

        S = X / np.sqrt(num_outputs) * np.exp(-1j * np.pi/2 * (I != J))
        if DEBUG: print(S)
        states_tmp = np.stack(states, 1)

        states_scattered = np.matmul(S, states_tmp)
        states_scattered_lst = [states_scattered[:, i, :] for i in range(states_scattered.shape[1])]
        return states_scattered_lst


# @register_node_types_all
class FrequencySplitter(MultiPath):
    node_acronym = 'FS'
    number_of_parameters = 0
    node_lock = False

    def __init__(self, **kwargs):
        self.upper_bounds = []
        self.lower_bounds = []
        self.data_types = []
        self.step_sizes = []
        self.parameter_imprecisions = []
        self.parameter_units = []
        self.parameter_locks = []
        self.parameter_names = []
        self.default_parameters = []
        self.parameter_symbols = []
        super().__init__(**kwargs)

    def update_attributes(self, num_inputs, num_outputs):
        num_parameters = num_outputs - 1
        self.number_of_parameters = num_parameters

        self.upper_bounds = [1.0] * num_parameters
        self.lower_bounds = [0.0] * num_parameters
        self.data_types = ['float'] * num_parameters
        self.step_sizes = [None] * num_parameters
        self.parameter_imprecisions = [0.05] * num_parameters
        self.parameter_units = [None] * num_parameters
        self.parameter_locks = [False] * num_parameters
        self.parameter_names = [f'ratio-{i}' for i in range(num_parameters)]
        self.default_parameters = [1 - 1 / i for i in range(num_parameters + 1, 1, -1)]
        self.parameter_symbols = [f'x_{i}' for i in range(num_parameters)]

        self.parameters = self.default_parameters
        return

    def propagate(self, states, propagator, num_inputs, num_outputs, save_transforms=False):
        DEBUG = False
        if DEBUG: print(f'FrequencySplitter', num_inputs, num_outputs, len(self.parameters), self.parameters)

        state = np.sum(np.stack(states, 1), axis=1)

        a = self.parameters
        g = np.array([0] + [(1 - an) * np.product(a[:n]) for n, an in enumerate(a)] + [np.product(a)])
        w = np.array([sum(g[:n]) for n in range(1, len(g))] + [1])
        left_cutoffs, right_cutoffs = w[:-1], w[1:]

        k = 500
        new_states = []
        tmp_x = np.linspace(0, 1, propagator.f.shape[0]).reshape(propagator.f.shape)
        for j, (left_cutoff, right_cutoff) in enumerate(zip(left_cutoffs, right_cutoffs)):
            logistic = ((1.0 / (1.0 + np.exp(-k * (tmp_x - left_cutoff))))
                        * (1.0 / (1.0 + np.exp(k * (tmp_x - right_cutoff)))))
            new_states.append(ifft_(ifft_shift_(logistic) * fft_(state, propagator.dt), propagator.dt))
        return new_states


# @register_node_types_all
# class DualOutputMZM(MultiPath):
#     """ Dual-output Mach-Zehnder Modulator with complementary arms """
#     node_acronym = 'MZM'
#     number_of_parameters = 3  # depth, frequency, bias
#     node_lock = False
#
#     def __init__(self, **kwargs):
#         self.default_parameters = [1.0, 5.0e9, 0.0]
#
#         self.max_frequency = 50.0e9
#         self.min_frequency = 1.0e9
#         self.step_frequency = 1.0e9
#
#         self.upper_bounds = [2 * np.pi, 50.0e9, 2 * np.pi]
#         self.lower_bounds = [0.0, 1.0e9, 0.0]
#         self.data_types = ['float', 'float', 'float']
#         self.step_sizes = [None, 1e9, None]
#         self.parameter_imprecisions = [0.1, 10e6, 0.05]
#         self.parameter_units = [unit.rad, unit.Hz, unit.rad]
#         self.parameter_locks = [False, False, False]
#         self.parameter_names = ['depth', 'frequency', 'bias']
#         self.parameter_symbols = [r"$\phi_m$", r"$f_{RF}$", r"$\phi_{DC}$"]
#
#         # self._loss_dB = -3.0  # 3dB splitting loss per arm
#         self._loss_dB = 0.0
#         super().__init__(**kwargs)
#
#         return
#
#     def update_attributes(self, num_inputs, num_outputs):
#         # 输入端口强制校验
#         if num_inputs != 2:
#             # raise ValueError(f"{self.__class__.__name__} requires 1 input (got {num_inputs})")
#             print(f"Warning:{self.__class__.__name__} requires 1 input (got {num_inputs})")
#         # 输出端口强制设置为2
#         if num_outputs != 2:
#             print(f"Warning: Forcing {self.__class__.__name__} output to 2 (requested {num_outputs})")
#
#         # 更新关键属性
#         self.num_inputs = 2
#         self.num_outputs = 2
#         self.number_of_parameters = 3
#
#         if hasattr(super(), 'update_attributes'):
#             super().update_attributes(num_inputs, num_outputs)
#         else:
#             self.parameters = self.default_parameters.copy()
#     def propagate(self, states, propagator, num_inputs, num_outputs, save_transforms=False):
#         """
#         E_out1 = (E_in/2) * [1 + e^{j(φ_DC + φ_m cos(2πf t))}] * α
#         E_out2 = (E_in/2) * [1 - e^{j(φ_DC + φ_m cos(2πf t))}] * α
#         α = 10^(-loss_dB/20)  # 50% power loss
#         """
#         depth = self.parameters[0]
#         frequency = self.parameters[1]
#         bias = self.parameters[2]
#
#         # 调制相位项
#         # phi = bias + depth * np.cos(2 * np.pi * frequency * propagator.t)
#         # exp_jphi = np.exp(1j * phi)
#         theta = 2 * np.pi * frequency * propagator.t
#         phi1 = bias + depth * np.cos(theta)
#         phi2 = bias + depth * np.cos(theta+np.pi)
#
#         # 互补输出计算
#         alpha = dB_to_amplitude_ratio(self._loss_dB)
#         out1 = (states[0] / 2) * (1 + np.exp(1j*phi1)) * alpha
#         out2 = (states[0] / 2) * (1 + np.exp(1j*phi2)) * alpha
#
#         if save_transforms: # save the power of each arm in time domain
#             self.transform = (
#                 ('t', power_(out1), 'arm1'),
#                 ('t', power_(out2), 'arm2')
#             )
#
#         return [out1, out2]

