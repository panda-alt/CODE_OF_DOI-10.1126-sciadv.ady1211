# -*- coding: utf-8 -*-
"""
This code uses datasets obtained via the SCPI protocol to train convolutional neural networks.
"""

import os.path

import numpy as np
import pandas
import serial
import serial.tools.list_ports
import threading
import time
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.signal as signal
import tqdm
import torch
import torch.nn as nn
import torch.utils.data.dataset as t_set
import torch.utils.data.dataloader as t_loader

global is_closed
is_closed = False


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


class Dataset:
    def __init__(self, path, random_array, is_train=True, num_per_type=0):
        self.path = path
        self.random_array = random_array
        self.is_train = is_train
        self.data_x = None
        self.data_y = None
        self.load_data()
        self.num_per_type = num_per_type

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, index):
        data = torch.Tensor(self.data_x[index, :, :])
        label = torch.Tensor(self.data_y[index, 0, :])
        return data, label

    def load_data(self):
        x_all = np.zeros((32 * 2000, 1, 406))
        y_all = np.zeros((32 * 2000, 1, 5))
        i_row_0 = 0
        i_row_1 = 0
        for i_status in range(32):
            data_i = np.load("{:s}/{:d}.npy".format(self.path, i_status))
            x_all[i_row_1:i_row_1 + data_i.shape[0], 0, :] = np.copy(data_i[:, 1:])
            i_row_1 += data_i.shape[0]
            str_bin_status = bin(i_status + 0xf000)
            for j in range(5):
                y_all[i_row_0:i_row_1, 0, j] = int(str_bin_status[j + 13])
            # print(y_all[i_row_0, 0, :])
            i_row_0 = i_row_1
        random_array = self.random_array[:i_row_1]
        x_all = x_all[:i_row_1, :, :] * 1000
        y_all = y_all[:i_row_1, :, :]
        index_train = (random_array < 0.7)
        index_test = (random_array >= 0.7)
        if self.is_train:
            self.data_x = x_all[index_train]
            self.data_y = y_all[index_train]
        else:
            self.data_x = x_all[index_test]
            self.data_y = y_all[index_test]
        # plt.plot(self.data_x[100, 0, :])
        # plt.show()


def on_close(event):
    global is_closed
    is_closed = True
    print('Figure closed!')


def train_net():
    global is_closed
    # 定义优化器和损失函数
    # 设置超参数
    epochs = 60
    learning_rate = 0.002
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
    random_array = np.random.random(2000 * 64)
    train_set = Dataset(path="DynamicData/SaveChannel_5", random_array=random_array, is_train=True)
    test_set = Dataset(path="DynamicData/SaveChannel_5", random_array=random_array, is_train=False)

    train_loader = t_loader.DataLoader(train_set, batch_size=50, shuffle=True)
    test_loader = t_loader.DataLoader(test_set, batch_size=1000000, shuffle=True)

    loss_train_list = []
    loss_test_list = []
    accu_test_list = []
    fig, axes = plt.subplots(1, 2)
    fig.canvas.mpl_connect('close_event', on_close)
    max_accu = 0.0
    f_history = open("DynamicData/CNN/history_l5_50_.csv", mode="w", encoding="UTF-8")
    f_history.write("epoc,loss_train, loss_test, accu\n")
    for epoch in range(epochs):
        print("进行第{}个epoch".format(epoch))
        for step, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = cnn(batch_x.cuda())  # batch_x=[50,1,28,28]
            loss = loss_function(output, batch_y.cuda())
            loss.backward()
            optimizer.step()
            # 为了实时显示准确率
            if step == 0:
                with torch.no_grad():
                    cnn.eval()
                    scheduler.step()
                    loss_train_list.append(loss.cpu().data.numpy())
                    for step_1, (test_x_tensor, test_y_tensor) in enumerate(test_loader):
                        test_output = cnn(torch.Tensor(test_x_tensor).cuda())
                        loss = loss_function(test_output, test_y_tensor.cuda())
                        loss_test_list.append(loss.cpu().data.numpy())

                        test_out_np = test_output.cpu().data.numpy()
                        index_open = (test_out_np > 0.5)
                        test_out_np *= 0.0
                        test_out_np[index_open] = 1.0
                        test_y = test_y_tensor.data.numpy()
                        compare = np.sum(np.abs(test_out_np - test_y[:, :]), axis=1)
                        index_true = (compare < 0.1)
                        accu_this = len(compare[index_true]) / test_out_np.shape[0]
                        # if accu_this > max_accu:
                        #     torch.save(cnn, "DynamicData/CNN/best_mode_oscilloscope_c5.pth")
                        # torch.save(cnn, "DynamicData/CNN/best_mode_oscilloscope_c5_ep{:d}.pth".format(epoch))
                        accu_test_list.append(accu_this)

                        n_step = len(loss_train_list)
                        steps = np.arange(1, n_step + 1, 1)
                        axes[0].cla()
                        axes[0].plot(np.array(loss_train_list), label="train")
                        axes[0].plot(np.array(loss_test_list), label="test")
                        axes[0].set_yscale("log")
                        axes[0].legend()
                        axes[1].cla()
                        axes[1].plot(np.array(accu_test_list), label="accu")
                        f_history.write("{:12.5E},{:12.5E},{:12.5E},{:12.5E}\n".format(
                            epoch, loss_train_list[-1], loss_test_list[-1], accu_test_list[-1]))
                        axes[1].legend()
                        if is_closed:
                            f_history.close()
                            return 0
                        plt.draw()
                        plt.pause(0.01)
                    cnn.train()


def set_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    set_seed(1234)
    cnn = CNN().cuda()
    train_net()
