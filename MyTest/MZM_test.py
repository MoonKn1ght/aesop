import numpy as np
import matplotlib.pyplot as plt

def directional_coupler(split_ratio=0.5):
    """
    生成 2x2 定向耦合器的传输矩阵
    split_ratio: 分束比 (默认 0.5 即 3dB 耦合器)
    """
    kappa = np.sqrt(split_ratio)
    tau = np.sqrt(1 - split_ratio)

    return np.array([[tau, -1j * kappa],
                     [-1j * kappa, tau]])

def phase_shifter_matrix(phi_top, phi_bottom):
    return np.array([[np.exp(-1j * phi_top), 0],
                     [0, np.exp(-1j * phi_bottom)]])

voltage = np.linspace(-5, 5, 200)
V_pi = 3.0
phi_bias = np.pi / 2

transmissions_port1 = []
transmissions_port2 = []

for v in voltage:
    # 推挽驱动
    delta_phi = np.pi * (v / V_pi)
    phi_1 = phi_bias / 2 + delta_phi / 2
    phi_2 = -phi_bias / 2 - delta_phi / 2

    # 矩阵
    C_in = directional_coupler(0.5)
    P_mod = phase_shifter_matrix(phi_1, phi_2)
    C_out = directional_coupler(0.5)
    M_sys = C_out @ P_mod @ C_in

    E_in = np.array([1, 0]) # 输入光场
    E_out = M_sys @ E_in # 输出

    transmissions_port1.append(np.abs(E_out[0]) ** 2)   # 功率
    transmissions_port2.append(np.abs(E_out[1]) ** 2)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(voltage, transmissions_port1, label='Output Port 1 (Bar)', linewidth=2)
plt.plot(voltage, transmissions_port2, label='Output Port 2 (Cross)', linestyle='--', linewidth=2)

plt.title(f'2x2 MZM Transfer Function (V_pi={V_pi}V)')
plt.xlabel('Applied Voltage (V)')
plt.ylabel('Normalized Optical Power')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()