# -*- coding: utf-8 -*-
"""
This code captures real-time data from an oscilloscope, uses a CNN network to identify states, then maps keys to
operate the “Calculator” application on a Windows computer.
"""
import os.path
from collections import defaultdict
import torchviz
import visualtorch
import pyautogui

from pyVisaGet5ChannelDataSave import Oscilloscope
import numpy as np
import serial
import serial.tools.list_ports
import threading
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.fft as fft
import scipy.signal as signal
import tqdm
import torch
import torch.nn as nn
import torch.utils.data.dataset as t_set
import torch.utils.data.dataloader as t_loader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=5),  # 输入406， 输出81
            nn.Conv1d(in_channels=1, out_channels=4,  # 输出 81
                      kernel_size=(11,), stride=(1,), padding=5, ),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2)  # 输出[1,40]
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=8,
                      kernel_size=(11,), stride=(1,), padding=5, ),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2)  # 输出[1,20]
        )

        self.liner = nn.Sequential(  # 输入[1,9]
            nn.Linear(in_features=1 * 20 * 8, out_features=32),
            nn.ELU(),
            nn.BatchNorm1d(num_features=32),
            nn.Linear(in_features=32, out_features=5)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # voltage_current = self.conv3(voltage_current)
        x = x.view(x.size(0), -1)  # 保留batch, 将后面的乘到一起 [batch, 32*7*7]
        output = self.liner(x)  # 输出[50,10]
        return output


def get_key(osc, axes):
    key_map = [["%", "E", "cap_s", "/"],
               ["7", "8", "9", "voltage_current"],
               ["4", "5", "6", "-"],
               ["1", "2", "3", "+"],
               ["N", "0", ".", "="]]
    scan = np.zeros((5, 5)) - 2  # -2 表示未扫描到
    color_view = np.zeros((5, 5, 4))
    y_view = np.linspace(0, 4, 5)
    while np.abs(np.sum(scan) - 2.0) > 0.1:
        scan = np.zeros((5, 5)) - 2  # -2 表示未扫描到
        while np.min(scan) < -1:  # 未完全扫描一遍
            channel, _, data_x = osc.get_fft_data()
            data_x = data_x * 1000
            tensor_x = torch.tensor(data_x.reshape([1, 1, -1]), dtype=torch.float32).cuda()
            test_output = cnn(torch.Tensor(tensor_x))
            test_out_np = test_output.cpu().data.numpy()

            index_open = (test_out_np > 0.5)
            test_out_np *= 0.0
            test_out_np[index_open] = 1.0
            color = cm.cool(test_out_np)[0]
            color_view[channel, :, :] = color
            axes.cla()
            for i in range(5):
                axes.scatter(np.ones(5) * (4 - i), 4 - y_view, c=color_view[i, :, :], s=4000, marker="s")
            for i in range(5):
                for j in range(4):
                    axes.text(4-j, 4-i, key_map[i][3-j], fontsize=20)
            axes.set_xlim([-1, 5])
            axes.set_ylim([-1, 5])
            plt.draw()
            plt.pause(0.01)
            for i in range(5):
                if test_out_np[0, i] > 0.5:
                    scan[i, channel] = 2  # 扫描到且张开
                else:
                    scan[i, channel] = 0  # 扫描到但未张开
    for i in range(5):
        for j in range(4):
            if np.abs(scan[i, j] - 2.0) < 0.1:
                return key_map[i][3-j]
    return "cap_s"


def click_calculator():
    my_osc = Oscilloscope("USB0::0x5656::0x0853::APO1423190111::INSTR")
    loc = {"%": np.array([1000, 416]),
           "E": np.array([1200, 416]),
           "cap_s": np.array([1302, 416]),
           "/": np.array([1502, 416]),
           "7": np.array([1000, 644]),
           "8": np.array([1200, 644]),
           "9": np.array([1302, 644]),
           "voltage_current": np.array([1502, 644]),
           "4": np.array([1002, 756]),
           "5": np.array([1202, 756]),
           "6": np.array([1302, 756]),
           "-": np.array([1502, 756]),
           "1": np.array([1002, 870]),
           "2": np.array([1202, 870]),
           "3": np.array([1302, 870]),
           "+": np.array([1502, 870]),
           "N": np.array([1002, 988]),
           "0": np.array([1202, 988]),
           ".": np.array([1302, 988]),
           "=": np.array([1502, 988])
           }

    def handle_close(event):
        my_osc.close()

    fig = plt.figure(figsize=(3, 3))
    fig.canvas.mpl_connect('close_event', handle_close)
    axes = plt.axes()
    while True:
        i_same = 0
        k2 = ""
        key1 = "cap_s"
        while i_same < 3:
            time.sleep(0.1)
            key1 = get_key(my_osc, axes)
            # print(i_same)
            if k2 == key1:
                i_same += 1
            else:
                i_same = 0
            k2 = key1
        loc_i = loc[key1]
        pyautogui.click(loc_i[0], loc_i[1])
        time.sleep(2)
        print("key: ", key1)


if __name__ == '__main__':
    cnn = torch.load("DynamicData/CNN/best_mode_oscilloscope_c5.pth", weights_only=False).cuda()
    click_calculator()
