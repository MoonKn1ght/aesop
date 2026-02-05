"""

"""

from pint import UnitRegistry
unit = UnitRegistry()
import autograd.numpy as np
import matplotlib.pyplot as plt

from ..node_types import MultiPath

from ..assets.decorators import register_node_types_all
from ..assets.functions import power_, psd_, fft_, ifft_, ifft_shift_, dB_to_amplitude_ratio
#
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

# 2*2 MZM
# @register_node_types_all
# class MZM2x2(MultiPath):
#     """
#     Full 2x2 Mach-Zehnder Modulator.
#     Structure: Input 3dB Coupler -> Phase Modulation Arms -> Output 3dB Coupler.
#     Supports 2 inputs and 2 outputs.
#     """
#     node_acronym = 'MZM2x2'
#     number_of_parameters = 3  # depth, frequency, bias
#     node_lock = False
#
#     def __init__(self, **kwargs):
#         self.default_parameters = [1.0, 5.0e9, 0.0]
#
#         self.max_frequency = 50.0e9
#         self.min_frequency = 1.0e9
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
#         self._loss_dB = 0.0 #插损
#
#         super().__init__(**kwargs)
#
#     def update_attributes(self, num_inputs, num_outputs):
#         """
#         强制校验端口数量为 2x2 配置
#         """
#         target_inputs = 2
#         target_outputs = 2
#
#         if num_inputs != target_inputs:
#             print(f"Warning: {self.node_acronym} expects {target_inputs} inputs, got {num_inputs}. "
#                   f"Missing inputs will be treated as zero (dark).")
#
#         if num_outputs != target_outputs:
#             print(f"Note: {self.node_acronym} physically has {target_outputs} outputs.")
#
#         self.number_of_parameters = 3
#
#         # 如果父类有 update_attributes 则调用，否则手动重置参数
#         if hasattr(super(), 'update_attributes'):
#             super().update_attributes(num_inputs, num_outputs)
#         else:
#             self.parameters = self.default_parameters.copy()
#
#     def propagate(self, states, propagator, num_inputs, num_outputs, save_transforms=False):
#         """
#         Physics-based propagation for 2x2 MZM using Transfer Matrix Matrix.
#
#         Fields evolution:
#         1. Inputs: E1, E2
#         2. After Input Coupler (50:50):
#            E_arm_upper = (E1 + j*E2) / sqrt(2)
#            E_arm_lower = (j*E1 + E2) / sqrt(2)
#         3. After Phase Modulation (Push-Pull):
#            E'_arm_upper = E_arm_upper * exp(j * phi1)
#            E'_arm_lower = E_arm_lower * exp(j * phi2)
#         4. After Output Coupler (50:50):
#            E_out1 = (E'_arm_upper + j*E'_arm_lower) / sqrt(2)
#            E_out2 = (j*E'_arm_upper + E'_arm_lower) / sqrt(2)
#         """
#
#         # 1. 获取输入场 (Handle Inputs)
#         # 确保有两个输入，如果只有一个输入连接，第二个视为0
#         E_in1 = states[0]
#         if len(states) > 1:
#             E_in2 = states[1]
#         else:
#             E_in2 = np.zeros_like(E_in1)  # 悬空端口输入为0
#
#         # 2. 获取参数 (Parameters)
#         depth = self.parameters[0]
#         frequency = self.parameters[1]
#         bias = self.parameters[2]
#
#         # 3. 计算调制相位 (Phase Calculation)
#         # Push-Pull 配置: 上臂与下臂相位调制互为反相
#         # theta = 2 * pi * f * t
#         theta = 2 * np.pi * frequency * propagator.t
#
#         # 你的设计中: phi1 = bias + depth * cos(theta)
#         #            phi2 = bias + depth * cos(theta + pi) = bias - depth * cos(theta)
#         # 注意：通常 Bias 加在其中一臂或差分加载，这里假设 Bias 是共模或包含在单臂中，
#         # 按照你的草稿逻辑，两臂都有 Bias。
#
#         phi_RF = depth * np.cos(theta)
#
#         # Arm 1 phase
#         phi_arm1 = bias + phi_RF
#         # Arm 2 phase (Push-Pull)
#         phi_arm2 = bias - phi_RF
#
#         # 4. 物理传播计算 (Propagation)
#         sqrt2 = np.sqrt(2)
#
#         # -- Step A: 输入耦合器 (Input 3dB Coupler) --
#         # 标准定向耦合器矩阵: [1, j; j, 1] / sqrt(2)
#         E_arm1 = (E_in1 + 1j * E_in2) / sqrt2
#         E_arm2 = (1j * E_in1 + E_in2) / sqrt2
#
#         # -- Step B: 相位调制 (Modulation Arms) --
#         E_arm1_mod = E_arm1 * np.exp(1j * phi_arm1)
#         E_arm2_mod = E_arm2 * np.exp(1j * phi_arm2)
#
#         # -- Step C: 输出耦合器 (Output 3dB Coupler) --
#         # 同样的矩阵结构
#         E_out1_raw = (E_arm1_mod + 1j * E_arm2_mod) / sqrt2
#         E_out2_raw = (1j * E_arm1_mod + E_arm2_mod) / sqrt2
#
#         # 5. 应用损耗 (Loss)
#         alpha = dB_to_amplitude_ratio(self._loss_dB)
#         E_out1 = E_out1_raw * alpha
#         E_out2 = E_out2_raw * alpha
#
#         # 6. 保存中间变量用于调试/绘图 (Optional Debugging)
#         if save_transforms:
#             # 保存两臂内的光强分布，有助于观察干涉前的状态
#             self.transform = (
#                 ('t', power_(E_arm1_mod), 'Power_Arm_Upper'),
#                 ('t', power_(E_arm2_mod), 'Power_Arm_Lower')
#             )
#
#         # 返回结果列表
#         # 注意：Aesop系统通常希望返回的输出数量与 num_outputs 匹配
#         # 如果下游只连接了一个输出，这里依然返回两个，系统会自动取用连接的那一个
#         return [E_out1, E_out2]
#

@register_node_types_all
class MZM2x2Node(MultiPath):
    """
    2x2 马赫-曾德尔调制器 (MZM) 节点模型
    参数:
    - v_pi: 半波电压 (V)
    - v_bias: 偏置电压 (V)
    - insertion_loss: 插入损耗 (0.0 - 1.0)
    """
    node_acronym = 'MZM'
    node_lock = False
    number_of_parameters = 3

    def __init__(self, **kwargs):
        # 初始化参数列表
        self.parameter_names = ['v_pi', 'v_bias', 'insertion_loss']
        self.default_parameters = [3.5, 1.75, 0.1]  # 默认值
        self.lower_bounds = [2.0, 0.0, 0.0]  # 下界
        self.upper_bounds = [5.0, 7.0, 1.0]  # 上界
        self.data_types = ['float', 'float', 'float']
        self.parameter_imprecisions = [0.05, 0.05, 0.01]
        self.parameter_units = ['V', 'V', 'dB']
        self.parameter_locks = [False, False, False]
        self.parameter_symbols = ['V_{\pi}', 'V_{bias}', 'IL']

        super().__init__(**kwargs)
        # 定义端口范围限制
        self._range_input_edges = [1, 2]
        self._range_output_edges = [1, 2]

    def update_attributes(self, num_inputs, num_outputs):
        """
        根据连接情况更新属性。对于 2x2 MZM，参数数量是固定的。
        """
        self.number_of_parameters = len(self.default_parameters)
        self.parameters = self.default_parameters
        return

    def propagate(self, states, propagator, num_inputs, num_outputs, save_transforms=False):
        """
        时域/频域信号传播逻辑
        states: 输入光场列表 [E_in1, E_in2]
        """
        # 提取当前优化后的参数
        v_pi, v_bias, loss = self.parameters

        # 计算相位偏移 theta
        # 物理公式: phi = pi * (V_bias / V_pi) / 2
        phi = (np.pi * v_bias) / (2.0 * v_pi)

        # 构建 2x2 传输矩阵 S
        # S = [[cos(phi), -j*sin(phi)], [-j*sin(phi), cos(phi)]]
        S = np.array([
            [np.cos(phi), -1j * np.sin(phi)],
            [-1j * np.sin(phi), np.cos(phi)]
        ])

        # 处理输入状态 (确保输入是 2xN 矩阵)
        # 如果只有一个输入，则另一个输入补零
        if len(states) == 1:
            states_tmp = np.stack([states[0], np.zeros_like(states[0])], axis=1)
        else:
            states_tmp = np.stack(states, axis=1)  # shape: (n_samples, 2, ...)

        # 应用传输矩阵和损耗
        # 使用 np.matmul 进行矩阵相乘以支持自动微分
        states_scattered = np.matmul(S, states_tmp) * (1.0 - loss)

        # 将结果转回状态列表返回
        # states_scattered 的 shape 通常是 (2, n_samples) 后的切片
        output_states = [states_scattered[:, i, :] for i in range(states_scattered.shape[1])]

        return output_states