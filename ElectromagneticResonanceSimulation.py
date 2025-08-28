# -*- coding: utf-8 -*-
"""
This code simulates the electromagnetic resonance signals of metamaterials under various conditions by solving the
differential equations of LC resonance.
"""

import random
import numpy as np
from scipy import integrate, fft
import matplotlib.pyplot as plt
import tqdm
# from multiprocessing import Pool


def differential_equation(voltage_current, t, ind_s, cap_s, res_s):
    """
    基尔霍夫 欧姆定律微分方程
    :param voltage_current: 输入电压电流
    :param t: 时间
    :param ind_s: 电感参数
    :param cap_s: 电容参数
    :param res_s: 电阻参数
    :return: 电压、电流时间导数
    """
    v_2, v_3, v_4, v_5, v_6, i_l1, i_l2, i_l3, i_l4, i_l5 = voltage_current
    # 构造矩阵A和向量b
    cap_matrix = np.array([
        [cap_s[0] + cap_s[1], -cap_s[1], 0, 0, 0],
        [-cap_s[1], cap_s[1] + cap_s[2], -cap_s[2], 0, 0],
        [0, -cap_s[2], cap_s[2] + cap_s[3], -cap_s[3], 0],
        [0, 0, -cap_s[3], cap_s[3] + cap_s[4], -cap_s[4]],
        [0, 0, 0, -cap_s[4], cap_s[4]]
    ])
    ind_matrix = np.array([i_l1 - i_l2, i_l2 - i_l3, i_l3 - i_l4, i_l4 - i_l5, i_l5])
    dv_dt = np.linalg.solve(cap_matrix, ind_matrix)
    # 电感电流导数
    u_matrix = np.array([-v_2 - res_s[0] * i_l1,
                         v_2 - v_3 - res_s[1] * i_l2,
                         v_3 - v_4 - res_s[2] * i_l3,
                         v_4 - v_5 - res_s[3] * i_l4,
                         v_5 - v_6 - res_s[4] * i_l5])
    dil_dt = u_matrix / ind_s
    return np.concatenate((dv_dt, dil_dt))


def solve_lc_ode_one(x):
    """
    单次求解LC振荡ODE
    :param x: i_type, L, R, C, x0, t_span, index_g30
    :return: 频谱[freq, amp] ndarray
    """
    i_type, L, R, C, x0, t_span, index_g30 = x
    sol = integrate.odeint(differential_equation, x0, t_span, args=(L, C, R))
    amp = np.abs(fft.fft(sol[:, 4], norm="forward")) * 2
    amp = amp[index_g30]
    return np.concatenate((np.array([i_type]), amp))


def generate_data_set():
    """
    生成LC振荡的数据集
    :return:
    """
    # 原件参数
    inductance_off = np.array([7e-6, 7e-6, 6e-6, 6e-6, 6e-6])  # OFF时的电感
    inductance_on = np.array([2.6e-6, 2.3e-6, 2.5e-6, 2.7e-6, 2.7e-6])  # ON时的电感
    inductance_near = np.array([3.7e-6, 3.7e-6, 3.7e-6, 3.7e-6, 3.7e-6])  # j+1个单元ON且j个单元OFF时，第j个单元的电感
    capacitance = np.array([220e-9, 330e-9, 470e-9, 680e-9, 1000e-9])  # 电容
    resistance = np.array([0.3, 0.3, 0.3, 0.3, 0.3])  # 电阻
    # 初始条件
    current_0 = np.array([6e-3] * 5)  # 初始电流
    voltage_0 = np.cumsum(resistance * current_0)  # 初始电压
    x0 = np.concatenate([voltage_0, current_0])

    # 时间范围
    t_span = np.arange(0, 7000E-7, 1.0E-7)[:-1]
    freq = fft.fftfreq(t_span.shape[0], d=t_span[1]) / 1000
    index_g30 = (freq > 10) * (freq < 300)
    freq = freq[index_g30]
    axes = plt.axes()
    # _pool = Pool(8)
    # for sigma_f in np.arange(0, 0.22, 0.02):  # 频率变异系数(标准差/均值)
    for sigma_f in np.arange(0, 0.12, 0.02):  # 频率变异系数(标准差/均值)
        for delta_f in np.arange(0, 0.25, 0.05):
            for i_capacitance in range(1, 5):
                capacitance[i_capacitance] = ((1 + delta_f)*capacitance[i_capacitance-1]**0.5)**2
            # print(capacitance)
            # return 0
            for i_type in range(32):
                inductance_base = np.copy(inductance_off)
                data_save = np.zeros((1000, 407))
                # 构建电感值
                str_bin = bin(i_type + 0xf000)[-5:]
                for i_bit in range(4):
                    if int(str_bin[i_bit + 1]) > 0.5:
                        inductance_base[i_bit] = inductance_near[i_bit]
                for i_bit in range(5):
                    if int(str_bin[i_bit]) > 0.5:
                        inductance_base[i_bit] = inductance_on[i_bit]
                # 参数
                paras = []
                freq_base = 1.0 / (2 * np.pi * np.sqrt(inductance_base * capacitance))
                randoms = np.random.normal(1, sigma_f, (1000, 5))
                for i_s in tqdm.tqdm(range(1000)):
                    freq_random = randoms[i_s, :] * freq_base
                    freq_random[freq_random < 10000] = 10000
                    inductance_random = 1.0 / ((freq_random * 2 * np.pi) ** 2 * capacitance)

                    sol = integrate.odeint(differential_equation, x0, t_span,
                                           args=(inductance_random, capacitance, resistance))
                    amp = np.abs(fft.fft(sol[:, 4], norm="forward")) * 2
                    amp = amp[index_g30]

                    axes.cla()
                    axes.plot(freq, amp)
                    axes.set_title("DeltaF{:.3f}_{:d}_{:s}".format(delta_f, i_type, str_bin))
                    plt.draw()
                    plt.pause(0.02)

                # 并行计算
                #     paras.append((i_type, inductance_random, resistance, capacitance, x0, t_span, index_g30))
                # res = _pool.map(solve_lc_ode_one, paras)
                # data_save[:, :] = np.array(res)
                # np.save("DynamicData/DataSet/delta_f/dataset/{:d}-delta_f{:.3f}-sigma_f{:.3f}".format(
                #     i_type, delta_f, sigma_f), data_save)


if __name__ == '__main__':
    generate_data_set()
