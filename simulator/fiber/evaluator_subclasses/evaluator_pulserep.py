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
        total_score = 0
        self.scores = {}
        state1=graph.measure_propagator('sink1')
        shifted1 = self.shift_function(state1,propagator,'sink1')
        ref1 = np.max(np.abs(self.targets['sink1']))
        ref2 = np.max(np.abs(shifted1))
        state2=graph.measure_propagator('sink2')
        score1 = self.waveform_temporal_similarity(state1, self.targets['sink1'], propagator, 'sink1', ref1)
        score2 = self.waveform_temporal_similarity(state2, self.targets['sink2'], propagator, 'sink2', ref2)
        total_score = score1+ score2


        # for node in self.evaluation_nodes:
        #     state = graph.measure_propagator(node)
        #     score = self.waveform_temporal_similarity(
        #         state, self.targets[node], propagator,node,ref
        #     )
        #     self.scores[node] = score
        #     total_score += score  #

        k = 10
        total_score += k*(len(graph.nodes)+len(graph.edges))
        return total_score

    def waveform_temporal_similarity(self, state, target, propagator,node,ref):
        # TODO
        def normalize_signal(signal):
            return signal / ref if ref > 0 else signal

        def normalize_target(target):
            max_val = np.max(np.abs(target))
            return target / max_val if max_val > 0 else target

        shifted = self.shift_function(state,propagator,node)
        shifted_normalized = normalize_signal(shifted)
        target_normalized = normalize_target(target)
        return np.sum((shifted_normalized - np.abs(target_normalized)) ** 2)

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
        # plt.plot(propagator.t, power_(self.target), label='target original', ls=':')
        # # plt.plot(propagator.t, shifted_target, label='target shifted', ls='-.')
        # plt.legend()
        return shifted


class PulseRepetition_multi(Evaluator):
    def __init__(self, propagator,
                 targets, pulse_width, rep_t, peak_power,
                 evaluation_nodes, **kwargs):
        super().__init__(**kwargs)
        self.evaluation_nodes = evaluation_nodes
        self.targets = targets
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
        self.scores = {}
        state1=graph.measure_propagator('sink1')
        shifted1 = self.shift_function(state1,propagator,'sink1')
        ref1 = np.max(np.abs(self.targets['sink1']))
        ref2 = np.max(np.abs(shifted1))
        total_score = self.waveform_temporal_similarity(state1, self.targets['sink1'], propagator, 'sink1', ref1)
        for node in self.evaluation_nodes[1:]:
            state = graph.measure_propagator(node)
            score = self.waveform_temporal_similarity(state, self.targets[node], propagator, node, ref2)
            total_score += score
        k = 50
        total_score += k*(len(graph.nodes)+len(graph.edges))
        return total_score

    def waveform_temporal_similarity(self, state, target, propagator,node,ref):
        # TODO
        def normalize_signal(signal):
            return signal / ref if ref > 0 else signal

        def normalize_target(target):
            max_val = np.max(np.abs(target))
            return target / max_val if max_val > 0 else target

        shifted = self.shift_function(state,propagator,node)
        shifted_normalized = normalize_signal(shifted)
        target_normalized = normalize_target(target)
        return np.sum((shifted_normalized - np.abs(target_normalized)) ** 2)

    def shift_function(self, state, propagator,node):
        # state_power = power_(state)
        state_power = state
        state_rf = np.fft.fft(state_power, axis=0)

        if (state_rf[self.target_harmonic_ind[node]] == 0):
            return state_power  # no phase shift in this case, and it'll break my lovely gradient otherwise (bit of a hack but...)

        # target_harmonic_ind = (self.rep_f / propagator.df).astype('int') + 1
        phase = np.angle(state_rf[self.target_harmonic_ind[node]] / self.target_rf[node][self.target_harmonic_ind[node]])
        shift = phase / (self.rep_f * propagator.dt)
        scale_array_col = self.scale_array.reshape(-1, 1)  # (32768,) ? (32768,1)
        state_rf *= np.exp(-1j * shift * scale_array_col)
        shifted = np.abs(np.fft.ifft(state_rf, axis=0))

        # plt.figure()
        # plt.plot(propagator.t, state_power, label='original', ls='-')
        # plt.plot(propagator.t, shifted, label='shifted', ls='--')
        # plt.plot(propagator.t, power_(self.target), label='target original', ls=':')
        # # plt.plot(propagator.t, shifted_target, label='target shifted', ls='-.')
        # plt.legend()
        return shifted

