"""

"""

from pint import UnitRegistry
unit = UnitRegistry()
import autograd.numpy as np
import matplotlib.pyplot as plt

from ..node_types import MultiPath

from ..assets.decorators import register_node_types_all
from ..assets.functions import power_, psd_, fft_, ifft_, ifft_shift_

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

        S = X / np.sqrt(num_outputs) * np.exp(-1j * 2 * np.pi * (I + J))
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

@register_node_types_all
class MZM2x2Node(MultiPath):
    """
    【动态调制 MZM】支持 RF 驱动的 2x2 MZM。
    功能：可以实现光开关、解复用 (Demux)、脉冲挑选。
    物理模型：传输矩阵随时间 t 变化。
    """
    node_acronym = 'MZM_MOD'
    node_lock = False

    def __init__(self, **kwargs):
        # 参数定义：
        # 0: V_pi (半波电压, 物理常量)
        # 1: V_bias (直流偏置, 决定静态工作点)
        # 2: V_rf (射频幅度, 决定开关的深度)
        # 3: f_rf (射频频率, 决定开关速度)
        # 4: phase_rf (射频相位, 决定开关的时间对齐)
        # 5: loss (插损)
        self.number_of_parameters = 6
        self.parameter_names = ['v_pi', 'v_bias', 'v_rf', 'f_rf', 'phase_rf', 'loss']

        # 默认值：2.5GHz 开关，相位可调
        self.default_parameters = [3.5, 1.75, 3.5, 2.5e9, 0.0, 0.1]

        self.upper_bounds = [10.0, 10.0, 10.0, 50.0e9, 2 * np.pi, 1.0]
        self.lower_bounds = [1.0, -10.0, 0.0, 0.0, -2 * np.pi, 0.0]

        # 锁定 V_pi, f_rf, loss (只优化 bias, v_rf, phase_rf)
        self.parameter_locks = [True, False, False, True, False, True]

        self.parameter_symbols = [r"$V_{\pi}$", r"$V_{bias}$", r"$V_{RF}$", r"$f_{RF}$", r"$\theta_{RF}$", "IL"]

        super().__init__(**kwargs)
        self._range_input_edges = [1, 2]

    def update_attributes(self, num_inputs, num_outputs):
        if not hasattr(self, 'parameters') or len(self.parameters) != 6:
            self.parameters = self.default_parameters

    def propagate(self, states, propagator, num_inputs, num_outputs, save_transforms=False):
        # 1. 解包参数
        v_pi, v_bias, v_rf, f_rf, phase_rf, loss = self.parameters

        # 2. 维度清洗 (确保输入是 1D 数组)
        clean_states = []
        for s in states:
            if np.ndim(s) > 1:
                clean_states.append(np.reshape(s, (-1,)))
            else:
                clean_states.append(s)

        # 准备输入矩阵 (2, N)
        if len(clean_states) == 1:
            s0 = clean_states[0]
            in_mat = np.stack([s0, np.zeros_like(s0)], axis=0)
        else:
            in_mat = np.stack(clean_states, axis=0)

        # 3. 计算随时间变化的相位 phi(t)
        # Formula: phi(t) = (pi / 2*V_pi) * (V_bias + V_rf * cos(2*pi*f*t + theta))
        t = propagator.t
        rf_signal = v_rf * np.cos(2 * np.pi * f_rf * t + phase_rf)
        total_voltage = v_bias + rf_signal

        phi_t = (np.pi * total_voltage) / (2.0 * v_pi)

        # 4. 构建动态传输矩阵 S(t)
        # S(t) = [[cos(phi_t), -j*sin(phi_t)], [-j*sin(phi_t), cos(phi_t)]]
        # 注意：这里 cos_phi 和 sin_phi 都是长度为 N 的数组
        cos_phi = np.cos(phi_t)
        sin_phi = np.sin(phi_t)

        # 5. 执行矩阵乘法 (Element-wise broadcasting)
        # Out1 = In1 * cos(phi) - j * In2 * sin(phi)
        # Out2 = -j * In1 * sin(phi) + In2 * cos(phi)

        E_in1 = in_mat[0, :]
        E_in2 = in_mat[1, :]

        factor = (1.0 - loss)
        out1 = (E_in1 * cos_phi - 1j * E_in2 * sin_phi) * factor
        out2 = (-1j * E_in1 * sin_phi + E_in2 * cos_phi) * factor

        return [out1, out2]