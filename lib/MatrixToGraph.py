import numpy as np
from lib.graph import Graph

# ==========================================
# 1. 导入组件 (与您的参考代码保持一致)
# ==========================================
# Terminals (端点)
from simulator.fiber.node_types_subclasses.terminals import TerminalSource, TerminalSink

# Inputs/Outputs (作为边存在的组件)
from simulator.fiber.node_types_subclasses.inputs import PulsedLaser
from simulator.fiber.node_types_subclasses.outputs import MeasurementDevice

# Components (作为节点存在的组件)
# 假设 MZM2x2Node 已经放在了 multi_path.py 中
from simulator.fiber.node_types_subclasses.multi_path import MZM2x2Node
from simulator.fiber.node_types_subclasses.single_path import Waveguide


def convert_matrix_to_aesop_graph(A, N_MZM, N_DET):
    """
    将邻接矩阵转换为 AESOP 的 Graph 对象。
    风格：Source/Sink 作为 Terminal 节点，Laser/MeasurementDevice 作为 Edge。
    """
    n_nodes = A.shape[0]
    nodes = {}
    edges = {}

    # ==========================================
    # 2. 准备基础组件实例
    # ==========================================
    # 参考代码中的激光器参数
    pulse_width, rep_t, peak_power = (5e-12, 1 / 10.0e9, 1.0)

    # 创建一个激光器对象配置 (作为边使用)
    # 注意：AESOP 中同一个对象实例如果在多处作为边使用，
    # 可能会共享参数。如果希望每条路独立优化(虽然Source通常锁定)，建议每次通过 copy 或新建。
    def create_laser_edge():
        laser = PulsedLaser(parameters_from_name={
            'pulse_width': pulse_width,
            'peak_power': peak_power,
            't_rep': rep_t,
            'pulse_shape': 'gaussian',
            'central_wl': 1.55e-6,
            'train': True
        })
        # 锁定激光器参数，不参与优化 (参考代码行为)
        laser.node_lock = True
        laser.protected = True
        return laser

    # 创建测量设备对象 (作为边使用)
    def create_measure_edge():
        md = MeasurementDevice()
        md.protected = True
        return md

    # ==========================================
    # 3. 实例化节点 (Nodes)
    # ==========================================
    for i in range(n_nodes):
        if i == 0:
            # 节点 0: TerminalSource (只是一个端点，不产生信号)
            nodes[i] = TerminalSource()
            nodes[i].node_acronym = 'SRC'

        elif 1 <= i <= N_MZM:
            # 节点 1 ~ N_MZM: MZM 组件
            nodes[i] = MZM2x2Node()
            nodes[i].node_acronym = f'MZM_{i}'

        elif i > N_MZM:
            # 节点 > N_MZM: TerminalSink (只是一个端点，接收信号)
            det_idx = i - N_MZM
            # 给 sink 起个名字，方便 evaluator 识别 (如参考代码中的 targets={'sink1': ...})
            nodes[i] = TerminalSink(node_name=f'sink{det_idx}')
            nodes[i].node_acronym = f'SINK_{det_idx}'

    # ==========================================
    # 4. 实例化边 (Edges)
    # ==========================================
    # 端口映射表
    port_mapping = {
        1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)
    }

    rows, cols = np.where(A > 0)
    for u, v in zip(rows, cols):
        code = A[u, v]
        if code not in port_mapping: continue

        out_port, in_port = port_mapping[code]

        # --- 核心逻辑：根据连接的位置选择边的类型 ---

        if u == 0:
            # Case A: 从 Source 发出的边 -> 使用 PulsedLaser
            # 逻辑：信号产生发生在从 Source 出来的这条线上
            edge_component = create_laser_edge()

        elif v > N_MZM:
            # Case B: 连向 Sink 的边 -> 使用 MeasurementDevice
            # 逻辑：测量发生在进入 Sink 之前的这条线上
            edge_component = create_measure_edge()

        else:
            # Case C: 中间连接 (MZM -> MZM) -> 使用 Waveguide
            edge_component = Waveguide()

        # 注入端口信息 (供 MZM propagate 使用)
        edge_component.src_port_idx = out_port
        edge_component.dst_port_idx = in_port

        # 将边加入字典
        # AESOP 某些版本支持 (u, v, port_index) 的 key 格式来支持多边，
        # 但标准 Graph.init_graph 接收 (u, v)。
        # 如果您的 MZM 之间有多条连接 (u->v 有两条线)，这里可能需要 AESOP 的 MultiGraph 支持。
        # 鉴于矩阵 A[u,v] 只有一个值，我们假设两点间只有一条线。
        edges[(u, v)] = edge_component

    # ==========================================
    # 5. 初始化图
    # ==========================================
    aesop_graph = Graph.init_graph(nodes, edges)

    return aesop_graph