# -*- coding: utf-8 -*-
"""
This code employs a mass-spring model to incorporate the static force-displacement relationship and inter-unit
coupling of metamaterials. By solving the corresponding dynamic differential equations, it simulates soliton
propagation within metamaterial chains.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as sip
import scipy.integrate as sit
import matplotlib.cm as cm
import os


class Dynamic:
    def __init__(self):
        data_inter = np.load("StaticData/data_inter.npy")
        # plt.plot(data_inter[:, 0], data_inter[:, 1])
        # plt.plot(data_inter[:, 0], data_inter[:, 2])
        # plt.grid()
        # plt.show()
        self.fun_fx_activate = sip.interp1d(data_inter[:, 0], data_inter[:, 1])
        self.fun_fx_deactivate = sip.interp1d(data_inter[:, 0], data_inter[:, 2])
        self.num_nodes = 40  # 节点数
        self.mass = 0.0035  # 单元质量
        self.damping = 500  # 阻尼, 降低调整收缩速率
        self.is_save_data = True  # 节点数
        self.is_closed_fig = False  # 节点数
        self.cons_len = 0.0225  # 缩短后长度
        self.u_min_potential = 0.0151  # 第二势能坐标
        self.friction = 14.30  # 摩擦

    def fun_fx(self, length_spring, length_spring_front):
        value_active = 1.0
        displacement = length_spring_front - self.cons_len
        if displacement < 0.006:
            value_active = (displacement - 0.002) / 0.004
        if displacement < 0.002:
            value_active = 0.0
        force = (1.0 - value_active) * self.fun_fx_activate(length_spring - self.cons_len)
        force += value_active * self.fun_fx_deactivate(length_spring - self.cons_len)
        return force

    def fun_ode(self, location_velocity, t):
        if int(t * 100000) % 100 == 0:
            print("time = {:.5f}s".format(t))
        locations = np.copy(location_velocity[:self.num_nodes])
        velocity = np.copy(location_velocity[self.num_nodes:])
        accelerations = np.zeros((self.num_nodes, 2))  # 来自左边与右边球的力产生的加速度
        # a[0, 1] = fun_fx(voltage_current[1] - voltage_current[0], 0.01+0.012359)/mass + (v[1]-v[0])*damp
        # 第一个球

        # 左端激活
        accelerations[0, 0] = 0
        accelerations[0, 1] = self.fun_fx(locations[1] - locations[0], 0.01) / self.mass + (
                velocity[1] - velocity[0]) * self.damping
        # # 左端不激活
        # accelerations[0, 1] = self.fun_fx(locations[1] - locations[0], self.cons_len + 0.008) / self.mass + (
        #         velocity[1] - velocity[0]) * self.damping
        # 中间的球
        for i_node in range(1, self.num_nodes - 1):
            accelerations[i_node, 0] = -accelerations[i_node - 1, 1]
            accelerations[i_node, 1] = self.fun_fx(locations[i_node + 1] - locations[i_node],
                                                   locations[i_node] - locations[i_node - 1]) / self.mass + (
                                               velocity[i_node + 1] - velocity[i_node]) * self.damping
        # 最后一个球
        accelerations_f = -self.fun_fx(locations[self.num_nodes - 1] - locations[self.num_nodes - 2],
                                       locations[self.num_nodes - 2] - locations[self.num_nodes - 3]) / self.mass
        accelerations_damping = -(velocity[self.num_nodes - 1] - velocity[self.num_nodes - 2]) * self.damping
        accelerations[self.num_nodes - 1, 0] = accelerations_f + accelerations_damping
        acceleration_new = np.reshape(np.sum(accelerations, axis=1), -1)
        friction = np.copy(velocity) / 0.01
        friction[friction <= -1] = -1.0
        friction[friction >= 1] = 1.0
        return np.concatenate((velocity, acceleration_new - friction * self.friction))

    def fun_closed_fig(self, para):
        self.is_closed_fig = True

    def main(self):
        locations_initial = np.linspace(0, (self.cons_len + self.u_min_potential) * (self.num_nodes - 1),
                                        self.num_nodes)
        # locations_initial[21:] -= self.cons_len
        velocity_initial = np.zeros(self.num_nodes)
        locations_velocity_initial = np.concatenate((locations_initial, velocity_initial))
        # print(dxdt(xv))
        time_array = np.linspace(0, 1.2, 600)

        file_name = "DynamicData/DynamicRes_{:d}.npy".format(self.num_nodes)
        if os.path.exists(file_name) and self.is_save_data:
            result_ode = np.load(file_name)
        else:
            res = sit.odeint(self.fun_ode, y0=locations_velocity_initial, t=time_array)
            result_ode = np.array(res)
            np.save(file_name, result_ode)
            print("Saved DynamicData")
        fig = plt.figure(figsize=(12, 1))
        # fig = plt.figure()
        fig.canvas.mpl_connect('close_event', self.fun_closed_fig)
        axes = plt.axes([0.0, 0.0, 1.0, 1.0])
        # axes = plt.axes([0.15, 0.15, 0.75, 0.75])
        for i in range(self.num_nodes):
            axes.plot(time_array, result_ode[:, i])
            plt.draw()
        # plt.show()
        # return 0
        plt.pause(3)
        while True:
            for i in range(result_ode.shape[0]):
                axes.cla()
                for j in range(self.num_nodes - 1):
                    axes.plot([result_ode[i, j], result_ode[i, j + 1]], [0, 0],
                              color=cm.jet((result_ode[i, j + 1] - result_ode[i, j] - 0.008) / 0.015))
                axes.scatter(result_ode[i, :self.num_nodes], np.zeros(self.num_nodes), color="k", marker="s", s=20)
                axes.set_xlim([-locations_initial[2], np.max(locations_initial) * 1.2])
                plt.draw()
                plt.pause(0.02)
                if self.is_closed_fig:
                    return 0
            return 0
        plt.show()


if __name__ == '__main__':
    dynamics = Dynamic()
    dynamics.main()
