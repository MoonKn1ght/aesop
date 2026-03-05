""" """

import autograd.numpy as np
import matplotlib.pyplot as plt

from ..evaluator import Evaluator
from ..assets.functions import fft_, ifft_, power_, psd_, rfspectrum_
from lib.functions import scale_units


class PulseRepetition(Evaluator):
    """  """

    def __init__(self, propagator,
                 target, pulse_width, rep_t, peak_power,
                 evaluation_node='sink', **kwargs):
        super().__init__(**kwargs)
        self.evaluation_node = evaluation_node
        self.target = target
        # self.target_power = power_(target)
        self.target_power = np.abs(target)
        self.pulse_width = pulse_width  # pulse width in s
        self.rep_t = rep_t  # target pattern repetition time in s
        self.peak_power = peak_power # peak power
        self.rep_f = 1.0/self.rep_t  # target reprate in Hz

        self.target_f = np.fft.fft(self.target, axis=0)

        self.target_rf = np.fft.fft(self.target_power, axis=0) # rf spectrum, used to shift global phase of generated
        self.scale_array = (np.fft.fftshift(
            np.linspace(0, len(self.target_f) - 1, len(self.target_f))) / propagator.n_samples).reshape(
            (propagator.n_samples, 1))

        self.target_harmonic_ind = (self.rep_f / propagator.df).astype('int') + 1
        if (self.target_harmonic_ind >= self.target_f.shape[0]):
            self.target_harmonic_ind = self.target_f.shape[0]


    def evaluate_graph(self, graph, propagator):
        graph.propagate(propagator)
        state = graph.measure_propagator(self.evaluation_node)

        score = self.waveform_temporal_similarity(state, propagator)
        return score

    @staticmethod
    def similarity_cosine(x_, y_):
        return np.sum(x_ * y_) / (np.sum(np.sqrt(np.power(x_, 2))) * np.sum(np.sqrt(np.power(y_, 2))))

    @staticmethod
    def similarity_l1_norm(x_, y_):
        return np.sum(np.abs(x_ - y_))

    @staticmethod
    def similarity_l2_norm(x_, y_):
        # return np.sum(np.power(x_ - y_, 2))
        return np.sum(np.power(x_ - y_, 2))

    def waveform_temporal_similarity(self, state, propagator):
        shifted = self.shift_function(state, propagator)
        similarity_func = self.similarity_l2_norm
        similarity = similarity_func(shifted, self.target_power)
        return similarity

    def shift_function(self, state, propagator):
        # state_power = power_(state)
        state_power = state
        state_rf = np.fft.fft(state_power, axis=0)

        if (state_rf[self.target_harmonic_ind] == 0):
            return state_power  # no phase shift in this case, and it'll break my lovely gradient otherwise (bit of a hack but...)

        # target_harmonic_ind = (self.rep_f / propagator.df).astype('int') + 1
        phase = np.angle(state_rf[self.target_harmonic_ind] / self.target_rf[self.target_harmonic_ind])
        shift = phase / (self.rep_f * propagator.dt)
        state_rf *= np.exp(-1j * shift * self.scale_array)
        shifted = np.abs(np.fft.ifft(state_rf, axis=0))


        # plt.figure()
        # plt.plot(propagator.t, state_power, label='original', ls='-')
        # plt.plot(propagator.t, shifted, label='shifted', ls='--')
        # plt.plot(propagator.t, power_(self.target), label='target original', ls=':')
        # # plt.plot(propagator.t, shifted_target, label='target shifted', ls='-.')
        # plt.legend()
        return shifted


    def compare(self, graph, propagator):
        evaluation_node = [node for node in graph.nodes if not graph.out_edges(node)][0]  # finds node with no outgoing edges

        fig, ax = plt.subplots(1, 1)
        state = graph.measure_propagator(evaluation_node)
        ax.plot(propagator.t, power_(state), label='Measured State')
        ax.plot(propagator.t, self.target, label='Target State')
        ax.set(xlabel='Time', ylabel='Power a.u.')
        ax.legend()
        scale_units(ax, unit='s', axes=['x'])
        plt.show()

        fig, ax = plt.subplots(1, 1)
        state = graph.measure_propagator(evaluation_node)
        ax.plot(rfspectrum_(state, propagator.dt), label='Measured State')
        ax.set(xlabel='', ylabel='Power a.u.')
        ax.legend()
        scale_units(ax, unit='Hz', axes=['x'])
        plt.show()
        return


class PulseRepetition_dual(Evaluator):
    def __init__(self, propagator,
                 targets, pulse_width, rep_t, peak_power,
                 evaluation_nodes=('sink1', 'sink2'), **kwargs):
        super().__init__(**kwargs)
        self.evaluation_nodes = evaluation_nodes
        self.targets = targets  #
        self.target_power = {}
        self.target_rf = {}
        self.target_f = {}
        self.target_harmonic_ind = {}

        self.pulse_width = pulse_width  # pulse width in s
        self.rep_t = rep_t  # target pattern repetition time in s
        self.peak_power = peak_power  # peak power
        self.rep_f = 1.0 / self.rep_t  # target reprate in Hz

        self.scale_array = np.fft.fftshift(
            np.linspace(0, propagator.n_samples - 1, propagator.n_samples)) / propagator.n_samples

        for node in self.evaluation_nodes:
            self.target_power[node] = np.abs(targets[node])
            self.target_rf[node] = np.fft.fft(self.target_power[node], axis=0)  # rf spectrum, used to shift global phase of generated
            self.target_f[node] = np.fft.fft(self.targets[node], axis=0)
            self.target_harmonic_ind[node] = (self.rep_f / propagator.df).astype('int') + 1
            if (self.target_harmonic_ind[node] >= self.target_f[node].shape[0]):
                self.target_harmonic_ind[node] = self.target_f[node].shape[0]


    def evaluate_graph(self, graph, propagator):

        # NOrmalized
        self.scores = {}
        state1=graph.measure_propagator('sink1')
        shifted1 = self.shift_function(state1,propagator,'sink1')
        ref1 = np.max(np.abs(self.targets['sink1']))
        ref2 = np.max(np.abs(shifted1))
        state2=graph.measure_propagator('sink2')
        score1 = self.waveform_temporal_similarity(state1, self.targets['sink1'], propagator, 'sink1', ref1)
        score2 = self.waveform_temporal_similarity(state2, self.targets['sink2'], propagator, 'sink2', ref2)
        total_score = score1 + score2
        # total_score = score2
        # total_score = score1
        #print('score1:', score1,'; score2:',score2)

        # No normalization
        # total_score = 0
        # for node in self.evaluation_nodes:
        #     state = graph.measure_propagator(node)
        #     score = self.waveform_temporal_similarity(
        #         state, self.targets[node], propagator,node
        #     )
        #     self.scores[node] = score
        #     total_score += score

        # k = 10
        # total_score += k*(len(graph.nodes)+len(graph.edges))
        return total_score

    # def evaluate_graph(self, graph, propagator):
    #     self.scores = {}
    #     # 获取第一个通路移相后的输出和最大值
    #     state1 = graph.measure_propagator(self.evaluation_nodes[0])
    #     shifted1 = self.shift_function(state1, propagator, self.evaluation_nodes[0])
    #     ref = np.max(np.abs(shifted1))  # 第一路最大幅度作为全局归一化参考
    #     total_score = 0
    #     for node in self.evaluation_nodes:
    #         state = graph.measure_propagator(node)
    #         score = self.waveform_temporal_similarity(state, self.targets[node], propagator, node, ref)
    #         self.scores[node] = score
    #         total_score += score
    #     k = 0.005
    #     total_score += k*(len(graph.nodes)+len(graph.edges))
    #     return total_score

###通道均一性考虑

    def evaluate_graph(self, graph, propagator, alpha=0.1):
        first_node = self.evaluation_nodes[0]
        # 获取所有通道移相后波形、最大值
        shifted = {}
        maxvals = {}
        # 计算第一通道
        state1 = graph.measure_propagator(first_node)
        shifted[first_node] = self.shift_function(state1, propagator, first_node)
        ref = np.max(np.abs(shifted[first_node]))  # 第一通道最大值
        total_score = 0
        # 记录所有通路loss和最大值
        for node in self.evaluation_nodes:
            state = graph.measure_propagator(node)
            shifted[node] = self.shift_function(state, propagator, node)
            maxvals[node] = np.max(np.abs(shifted[node]))
            score = self.waveform_temporal_similarity(state, self.targets[node], propagator, node, ref)
            total_score += score
        # 均一性loss
        uniformity_loss = 0
        for node in self.evaluation_nodes:
            if node == first_node:
                continue  # 跳过自己
            uniformity_loss += abs(maxvals[node] / ref - 1)
        # 合计总loss
        total_score += alpha * uniformity_loss

        k = 0.01
        total_score += k * (len(graph.nodes) + len(graph.edges))

        return total_score

    # Original
    # def waveform_temporal_similarity(self, state, target, propagator,node):
    #     shifted = self.shift_function(state,propagator,node)
    #     return np.sum((shifted - np.abs(target)) ** 2)

    # Normalized
    def waveform_temporal_similarity(self, state, target, propagator,node,ref):
        def normalize_signal(signal):
            return signal / ref if ref > 0 else signal

        def normalize_target(target):
            max_val = np.max(np.abs(target))
            return target / max_val if max_val > 0 else target

        shifted = self.shift_function(state,propagator,node)
        shifted_normalized = normalize_signal(shifted)
        target_normalized = normalize_target(target)
        #TODO: 要除以采样点数
        return np.sum((shifted_normalized - np.abs(target_normalized)) ** 2)/propagator.n_samples

    def shift_function(self, state, propagator,node):
        # state_power = power_(state)
        state_power = state
        state_rf = np.fft.fft(state_power, axis=0)

        if (state_rf[self.target_harmonic_ind[node]] == 0):
            return state_power  # no phase shift in this case, and it'll break my lovely gradient otherwise (bit of a hack but...)

        # target_harmonic_ind = (self.rep_f / propagator.df).astype('int') + 1
        phase = np.angle(state_rf[self.target_harmonic_ind[node]] / self.target_rf[node][self.target_harmonic_ind[node]])
        shift = phase / (self.rep_f * propagator.dt)
        scale_array_col = self.scale_array.reshape(-1, 1)
        state_rf *= np.exp(-1j * shift * scale_array_col)
        shifted = np.abs(np.fft.ifft(state_rf, axis=0))

        # plt.figure()
        # plt.plot(propagator.t, state_power, label='original', ls='-')
        # plt.plot(propagator.t, shifted, label='shifted', ls='--')
        # plt.plot(propagator.t, power_(self.targets[node]), label='target original', ls=':')
        # # plt.plot(propagator.t, shifted_target, label='target shifted', ls='-.')
        # plt.legend()
        # plt.show()
        return shifted


# class PulseRepetition_multi(Evaluator):
#     """
#     多端口脉冲重复频率评估器 (Multi-port Pulse Repetition Evaluator)
#
#     功能：同时评估 N 个输出端口的波形是否符合 N 个不同的目标波形。
#     包含：
#     1. 波形相似度 (MSE)
#     2. 通道间幅度均一性 (Uniformity Loss)
#     3. 系统复杂度惩罚 (Complexity Penalty)
#     """
#
#     def __init__(self, propagator,
#                  targets, pulse_width, rep_t, peak_power,
#                  evaluation_nodes, **kwargs):
#         super().__init__(**kwargs)
#
#         # --- 1. 核心数据存储 ---
#         self.evaluation_nodes = evaluation_nodes  # 这是一个列表，存的是整数 ID [4, 5, 6, 7]
#         self.targets = targets  # 这是一个字典，{4: wave, 5: wave...}
#
#         # --- 2. 健壮性自检 (Self Check) ---
#         if not self.evaluation_nodes:
#             raise ValueError("[Evaluator Error] evaluation_nodes 列表为空！请检查主程序初始化逻辑。")
#
#         # 检查 targets 的键是否覆盖了 evaluation_nodes
#         for node in self.evaluation_nodes:
#             if node not in self.targets:
#                 raise KeyError(f"[Evaluator Error] 目标字典 targets 中缺少节点 ID {node} 的波形数据！")
#
#         # --- 3. 物理参数与频谱预计算 ---
#         self.target_power = {}
#         self.target_rf = {}
#         self.target_f = {}
#         self.target_harmonic_ind = {}
#
#         self.pulse_width = pulse_width
#         self.rep_t = rep_t
#         self.peak_power = peak_power
#         self.rep_f = 1.0 / self.rep_t
#
#         # 预先计算频移数组，避免在循环中重复计算
#         self.scale_array = np.fft.fftshift(
#             np.linspace(0, propagator.n_samples - 1, propagator.n_samples)) / propagator.n_samples
#
#         for node in self.evaluation_nodes:
#             # 统一转为绝对值 (功率/幅度) 进行预处理
#             self.target_power[node] = np.abs(targets[node])
#
#             # 计算目标的射频频谱 (RF Spectrum)，用于相位对齐
#             self.target_rf[node] = np.fft.fft(self.target_power[node], axis=0)
#             self.target_f[node] = np.fft.fft(self.targets[node], axis=0)
#
#             # 计算基频对应的索引位置
#             harmonic_idx = (self.rep_f / propagator.df).astype('int') + 1
#             # 防止索引越界
#             if harmonic_idx >= self.target_f[node].shape[0]:
#                 harmonic_idx = self.target_f[node].shape[0] - 1
#
#             self.target_harmonic_ind[node] = harmonic_idx
#
#     def evaluate_graph(self, graph, propagator, alpha=0.01):
#         """
#         计算整个图的得分 (Loss)。得分越低越好。
#
#         :param alpha: 均一性损失的权重系数
#         """
#         # 确保有节点可测
#         if not self.evaluation_nodes:
#             return 1000.0  # 返回一个巨大的惩罚值
#
#         first_node = self.evaluation_nodes[0]
#         shifted = {}
#         maxvals = {}
#
#         # ---------------------------
#         # A. 第一路作为参考 (Reference)
#         # ---------------------------
#         try:
#             state1 = graph.measure_propagator(first_node)
#             # 关键：如果 measure 返回 None (断路)，给一个全零信号防止报错
#             if state1 is None:
#                 state1 = np.zeros(propagator.n_samples, dtype=complex)
#
#             shifted[first_node] = self.shift_function(state1, propagator, first_node)
#             ref = np.max(np.abs(shifted[first_node]))
#
#             # 防止除以零
#             if ref < 1e-9:
#                 ref = 1e-9
#
#         except Exception as e:
#             # 如果连第一路都崩了，直接给大惩罚
#             return 1000.0
#
#         total_score = 0
#
#         # ---------------------------
#         # B. 遍历所有通道计算波形误差
#         # ---------------------------
#         for node in self.evaluation_nodes:
#             try:
#                 state = graph.measure_propagator(node)
#                 if state is None:
#                     state = np.zeros(propagator.n_samples, dtype=complex)
#
#                 # 对齐相位
#                 shifted[node] = self.shift_function(state, propagator, node)
#                 maxvals[node] = np.max(np.abs(shifted[node]))
#
#                 # 计算相似度 (MSE)
#                 score = self.waveform_temporal_similarity(state, self.targets[node], propagator, node, ref)
#                 total_score += score
#
#             except KeyError:
#                 # 如果图里突然找不到这个节点 (比如被优化算法删了)
#                 total_score += 10.0  # 惩罚
#
#         # ---------------------------
#         # C. 均一性损失 (Uniformity Loss)
#         # ---------------------------
#         uniformity_loss = 0
#         for node in self.evaluation_nodes:
#             if node == first_node:
#                 continue
#             # 计算各通道最大值相对于参考通道的偏差
#             uniformity_loss += abs(maxvals.get(node, 0) / ref - 1)
#
#         total_score += alpha * uniformity_loss
#
#         # ---------------------------
#         # D. 复杂度惩罚 (Cost Penalty)
#         # ---------------------------
#         k = 0.003
#         total_score += k * (len(graph.nodes) + len(graph.edges))
#
#         return total_score
#
#     def waveform_temporal_similarity(self, state, target, propagator, node, ref):
#         """
#         计算波形的时域相似度 (Mean Squared Error)
#         """
#
#         def normalize_signal(signal):
#             # 使用全局 ref 进行归一化，而不是自己归一化自己
#             # 这样如果某路信号很弱，MSE 会很大，从而迫使算法提高该路功率
#             return signal / ref
#
#         def normalize_target(target):
#             # 目标波形自归一化到 1
#             max_val = np.max(np.abs(target))
#             return target / max_val if max_val > 1e-9 else target
#
#         # 1. 移相 (对齐时间零点)
#         shifted = self.shift_function(state, propagator, node)
#
#         # 2. 归一化
#         shifted_normalized = normalize_signal(shifted)
#         target_normalized = normalize_target(target)
#
#         # 3. 计算 MSE
#         # np.abs 确保比较的是包络/功率，忽略载波相位差异
#         return np.sum((shifted_normalized - np.abs(target_normalized)) ** 2) / propagator.n_samples
#
#     def shift_function(self, state, propagator, node):
#         """
#         利用 FFT 对齐波形的相位 (Time Shifting)
#         """
#         # 确保输入是 numpy 数组
#         state = np.array(state)
#         # 很多时候我们只关心光强分布，不关心载波相位
#         # 但原来的代码里似乎是用复数算的，这里保持原逻辑
#         state_rf = np.fft.fft(state, axis=0)
#
#         h_idx = self.target_harmonic_ind[node]
#
#         # 防御性编程：索引越界保护
#         if h_idx >= len(state_rf):
#             return np.abs(state)
#
#         # 如果基频分量为0，没法对齐，直接返回原模值
#         if np.abs(state_rf[h_idx]) < 1e-12:
#             return np.abs(state)
#
#         # 计算相位差
#         # target_rf 是我们在 __init__ 里预计算好的
#         phase = np.angle(state_rf[h_idx] / self.target_rf[node][h_idx])
#
#         shift = phase / (self.rep_f * propagator.dt)
#
#         # 应用相位校正
#         scale_array_col = self.scale_array.reshape(-1, 1)
#         state_rf *= np.exp(-1j * shift * scale_array_col)
#
#         # IFFT 回去并取模 (变成实数波形)
#         shifted = np.abs(np.fft.ifft(state_rf, axis=0))
#
#         return shifted

class PulseRepetition_multi(Evaluator):
    """
    【修复版】多端口脉冲重复频率评估器

    特点：
    1. 移除不稳定的 FFT 移相，直接比较光强包络 (Power Envelope)。
    2. 引入“功率均一性”惩罚，强迫 MZM 平衡各路输出。
    3. 全流程 Autograd 兼容，确保梯度计算顺滑。
    """

    def __init__(self, propagator,
                 targets, pulse_width, rep_t, peak_power,
                 evaluation_nodes, **kwargs):
        super().__init__(**kwargs)

        # --- 1. 基础数据 ---
        self.evaluation_nodes = evaluation_nodes  # [4, 5, 6, 7]
        self.targets = targets  # {4: wave, ...}

        # --- 2. 健壮性检查 ---
        if not self.evaluation_nodes:
            raise ValueError("[Evaluator Error] evaluation_nodes 为空！")

        # --- 3. 预处理目标波形 (转为光强并归一化) ---
        # 我们只比较形状 (Shape) 和 相对幅度 (Relative Amplitude)
        self.target_powers_norm = {}
        self.target_max_vals = {}

        for node in self.evaluation_nodes:
            if node not in targets:
                raise KeyError(f"Targets 缺少节点 {node}")

            # 获取目标光强 (Power)
            # 注意：targets[node] 可能是复数电场，先转模平方
            t_field = targets[node]
            t_power = np.abs(t_field) ** 2

            # 记录最大值，用于后续缩放
            t_max = np.max(t_power)
            if t_max < 1e-12: t_max = 1.0  # 防止除零

            self.target_max_vals[node] = t_max
            # 存储归一化后的目标形状，方便反复调用
            self.target_powers_norm[node] = t_power / t_max

    def evaluate_graph(self, graph, propagator):
        """
        计算总 Loss。
        Loss = (波形形状误差) + alpha * (功率不均一误差) + beta * (复杂度惩罚)
        """
        total_shape_loss = 0.0
        output_max_vals = []

        # 权重系数 (可以根据需要微调)
        ALPHA_UNIFORMITY = 5.0  # 强迫功率平衡
        BETA_COMPLEXITY = 0.001  # 极其微小的复杂度惩罚

        # -------------------------------------------------
        # 遍历所有监测点，累加形状误差
        # -------------------------------------------------
        for node in self.evaluation_nodes:
            # 1. 测量输出
            # 如果 measure 失败，返回全0数组以保持梯度图完整
            try:
                state = graph.measure_propagator(node)
                if state is None:
                    # 创建一个全0数组，且必须是 autograd 友好的
                    state = np.zeros(propagator.n_samples, dtype=complex)
            except:
                state = np.zeros(propagator.n_samples, dtype=complex)

            # 2. 转换为光强 (Power)
            # measure_propagator 返回的是复数电场 E
            # Power = |E|^2
            power = np.abs(state) ** 2

            # 3. 获取最大值 (用于均一性计算)
            p_max = np.max(power)
            # 为了 autograd 安全，避免 p_max 变为 0 导致后续除法爆炸
            # 使用一个平滑的 max 或者加一个 epsilon
            p_max_safe = p_max + 1e-12

            output_max_vals.append(p_max_safe)

            # 4. 形状相似度 (MSE of Normalized Envelopes)
            # 将当前输出归一化，只比较“形状”是否像目标
            power_norm = power / p_max_safe
            target_norm = self.target_powers_norm[node]

            # 计算均方误差 (MSE)
            mse = np.mean((power_norm - target_norm) ** 2)
            total_shape_loss += mse

        # -------------------------------------------------
        # 计算功率均一性 (Uniformity)
        # -------------------------------------------------
        # 我们希望所有端口的输出功率最大值都接近目标的功率最大值
        # 或者简单点：所有端口的功率应该相等

        # 这里使用“变异系数”或“相对于目标的偏差”
        # 目标是所有 output_max_vals 应该接近 1.0 (因为 peak_power=1.0)
        # 假设所有 Sink 的目标功率都是一样的
        target_ref_power = list(self.target_max_vals.values())[0]

        uniformity_loss = 0.0
        for val in output_max_vals:
            # 惩罚偏离目标功率的行为
            # 使用平方误差让梯度更明显
            uniformity_loss += ((val - target_ref_power) / target_ref_power) ** 2

        # -------------------------------------------------
        # 汇总
        # -------------------------------------------------
        final_score = total_shape_loss + (ALPHA_UNIFORMITY * uniformity_loss)

        # 加上微小的复杂度惩罚 (防止网络无限增长，虽然 Benchmark 里结构固定)
        complexity = len(graph.nodes) + len(graph.edges)
        final_score += BETA_COMPLEXITY * complexity

        return final_score