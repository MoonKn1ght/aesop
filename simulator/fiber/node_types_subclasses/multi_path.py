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
        self.node_lock = False

        # 1. 先定义参数数量
        self.number_of_parameters = 3

        # 2. 定义所有必要的属性列表 (GA 算法严重依赖这些！)
        self.parameter_names = ['v_pi', 'v_bias', 'insertion_loss']
        self.default_parameters = [3.5, 1.75, 0.1]
        self.upper_bounds = [10.0, 10.0, 1.0]
        self.lower_bounds = [1.0, -10.0, 0.0]
        self.data_types = ['float', 'float', 'float']

        # 【关键点】这就是报错缺少的属性
        # None 表示使用默认步长，或者你可以填具体的数值比如 [0.1, 0.1, 0.01]
        self.step_sizes = [None, None, None]

        self.parameter_imprecisions = [0.01, 0.01, 0.001]
        self.parameter_units = ['V', 'V', 'dB']
        self.parameter_locks = [False, False, False]
        self.parameter_symbols = [r"$V_{\pi}$", r"$V_{bias}$", r"$IL$"]

        # 3. 最后调用父类初始化
        super().__init__(**kwargs)

        # 4. 定义端口范围
        self._range_input_edges = [1, 2]
        self._range_output_edges = [1, 2]

    def update_attributes(self, num_inputs, num_outputs):
        """
        双重保险：确保在图构建完成后，这些属性依然存在。
        """
        self.number_of_parameters = 3

        # 强制检查并重新赋值 step_sizes，防止丢失
        if not hasattr(self, 'step_sizes') or len(self.step_sizes) != 3:
            self.step_sizes = [None, None, None]

        # 检查 parameters 是否存在
        if not hasattr(self, 'parameters') or len(self.parameters) != 3:
            self.parameters = self.default_parameters

        # 其他属性如果怕丢，也可以在这里补，但通常 step_sizes 是重灾区
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