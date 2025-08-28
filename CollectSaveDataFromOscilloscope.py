# -*- coding: utf-8 -*-
"""
This code uses the SCPI protocol to retrieve electromagnetic resonance signals from a UNI-T oscilloscope and saves the
data in an *.npy file.
"""

import time
import scipy.fft as fft
import scipy.signal as signal
import pyvisa as visa
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import tqdm
import re
from threading import Thread

global time_array_g, voltage_g, freq_cut_g, amp_cut_g, is_close, is_data_ok, channel_g
is_data_ok = False
is_close = False


class Oscilloscope:
    def __init__(self, device_name):
        self.time_scale = 100.0E-6  # s/div
        self.channel_scale = 100.0E-3  # V/Div
        self.n_data = 6999
        self.n_samples = 1000
        self.ResourceManager = visa.ResourceManager()
        self.instrument = self.ResourceManager.open_resource(device_name)
        self.is_close = False
        self.window = signal.windows.hann(self.n_data)
        self.time_array = np.linspace(0, self.time_scale * 14, self.n_data)
        self.freq = fft.fftfreq(n=self.n_data, d=self.time_array[1] - self.time_array[0])
        self.index_cut = (self.freq > 10E3) * (self.freq < 300E3)
        command_list = [":TRIGger:MODE EDGE",  # 边沿触发
                        ":TRIGger:SWEep NORMal",  # 一般触发
                        ":TRIGger:SOURce EXT",  # 外部触发
                        ":TRIGger:EDGE:SLOPe POSitive",  # 上升沿触发
                        ":CHANnel1:COUPling DC",  # 直流耦合
                        ":CHAN1:DISP ON",  # 打开通道1
                        ":CHAN1:OFFSet 0V",  # 通道1偏移0V
                        ":CHAN1:SCAL {:f}V".format(self.channel_scale),
                        ":TIMebase:SCALe {:f}".format(self.time_scale),
                        ":TIMebase:OFFSet {:f}".format(100.0E-6),  # 左偏移150us
                        ":WAVeform:SOURce CHAN1",  # 设置当前要查询波形数据的信号源为通道一
                        ":WAVeform:MODE RAW",  # 设置读取内存波形数据
                        ":ACQuire:MEMory:DEPTh 7K",  # 存储深度7K
                        ":WAVeform:STARt 1",  # 从第1个点开始读取
                        ":WAVeform:STOP 7000",  # 读取到7000个点（读取全部数据）
                        ":WAVeform:FORMat BYTE",  # 波形数据的返回格式为单字节模式
                        ":WAVeform:POINts 7000"  # 共采集7000个点（一次读取全部数据）
                        ]
        for command_i in command_list:  # 初始化设置
            self.instrument.write(command_i)
            command_query = re.sub(r"\s+.+", "?", command_i)
            print(("Set: {:<28s} ,Get: {}".format(command_i,
                                                  self.instrument.query(command_query))).replace("\n", ""))

    def read_waveform_data(self):
        data_merge = np.array([])
        star_t = 1
        self.instrument.write(":WAVeform:MODE RAW")  # 更新内存数据和重置START
        while star_t > 0:  # START为-1说明本帧数据已读取完
            self.instrument.write(":WAVeform:DATA?")  # 获取波形指令
            data = self.instrument.read_raw()  # 读取数据
            data_np = np.frombuffer(data,  # 转换到 ndarray
                                    dtype=np.uint8,  # 此处只用到相对大小，未进行电压值转换
                                    count=len(data))
            data_merge = np.concatenate((data_merge, data_np[6:-3]))
            star_t = int(self.instrument.query(":WAVeform:START?"))
            time.sleep(0.001)
        channel = get_channel_num(self.time_scale, data_merge)
        return channel, data_merge

    def close(self):
        self.instrument.close()

    def get_fft_data(self):
        channel = -1
        while channel < 0:
            channel, data_raw = self.read_waveform_data()
        data_raw[data_raw > 250] = 128
        voltage = (data_raw - 128) * self.channel_scale / 255
        amp = np.abs(fft.fft(voltage * self.window, norm="forward") * 4)
        return channel, voltage, amp[self.index_cut]

    def plot_save_data(self, i_type):
        global time_array_g, voltage_g, freq_cut_g, amp_cut_g, is_close, is_data_ok, channel_g

        freq_cut = self.freq[self.index_cut] / 1000
        data_save = np.zeros((self.n_samples, len(freq_cut) + 1))
        while True:
            # for i_get in tqdm.tqdm(range(self.n_samples)):
            if is_close:
                self.instrument.close()
                return 0
            channel, voltage, amp_cut = self.get_fft_data()
            # data_save[i_get, 1:] = np.copy(amp_cut)
            # data_save[i_get, 0] = channel * 0.1
            is_data_ok = False
            time_array_g, voltage_g, freq_cut_g, amp_cut_g, channel_g = \
                (self.time_array, voltage, freq_cut, amp_cut, channel)
            is_data_ok = True
        self.instrument.close()
        # np.save("DynamicData/SaveChannel_5_001/{:d}_addition.npy".format(i_type), data_save)


def handle_close(event):
    global is_close
    is_close = True
    print("Figure was closed!")


def view_data():
    global is_data_ok, is_close
    plt.style.use("one")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.canvas.manager.window.wm_geometry('+600+300')
    fig.canvas.mpl_connect('close_event', handle_close)
    line_channels = []
    x_temp = np.linspace(0, 1, 100)
    for i in range(5):
        line_channels.append(axes[0].plot(x_temp * 1.5, (x_temp - 0.5) * 110, label="cap_s-{:d}".format(i + 1)))
        line_channels.append(axes[1].plot(x_temp * 300, x_temp * 1.5, label="cap_s-{:d}".format(i + 1)))
    # print(line_channels)
    # axes[0].legend()
    axes[0].set_xlabel("Time/ms")
    axes[0].set_ylabel("Voltage/mV")
    axes[0].set_xlim([0.62, 0.82])
    axes[1].set_xlabel("Frequency/kHz")
    axes[1].set_ylabel("Amp/mV")
    axes[1].legend()
    plt.draw()
    plt.pause(0.1)
    while True:
        if is_close:
            return 0
        if is_data_ok:
            time_array, voltage, freq_cut, amp_cut, channel = \
                (time_array_g, voltage_g, freq_cut_g, amp_cut_g, channel_g)
            if channel == 0 or True:
                line_channels[channel * 2][0].set_data(time_array * 1000, voltage * 1000)
                line_channels[channel * 2 + 1][0].set_data(freq_cut, amp_cut * 1000)
            plt.draw()
        plt.pause(0.05)


@jit(nopython=True)
def get_channel_num(time_scale, data):
    bias_index = -100.0E-6 / (time_scale * 14) * 7000 + 142
    center_index = 7000 / 2
    width_index = 2.5E-6 / (time_scale * 14) * 7000
    for i in range(5):
        delay_time = i * 5.0E-6
        delay_index = delay_time / (time_scale * 14) * 7000
        detection_index_1 = int(center_index + bias_index + delay_index - width_index)
        detection_index_0 = int(center_index + bias_index + delay_index + width_index)
        if (data[detection_index_0] < 170) and (data[detection_index_1] > 200):
            return i
    return -1


def open_visa_save_data(i_type):
    my_osc = Oscilloscope("USB0::0x5656::0x0853::APO1423190111::INSTR")
    my_osc.plot_save_data(i_type)


def view_saved_data():
    fig = plt.figure(figsize=(16, 8))
    plt.style.use("chinese")
    axes = []
    for i in range(32):
        w = 0.09
        h = 0.14
        x = 0.05 + w * (i % 8) * 1.33
        y = 0.1 + h * (i // 8) * 1.7
        axes.append(plt.axes([x, y, w, h]))
        # axes[i].set_xlabel("Freq /kHz")
        # axes[i].set_ylabel("Amp /mV")
    freq = np.linspace(10, 300, 406)
    for i in range(32):
        data = np.load("DynamicData/SaveChannel_5/{:d}.npy".format(i))
        for j in range(100):  # DynamicData.shape[0]):
            axes[i].scatter(freq, data[j*10, 1:]*1000, alpha=0.01, marker=".", color="#0000aa", linewidths=0)
            if j == 99:
                axes[i].text(180, axes[i].get_ylim()[1]*0.8, "0x" + ("{:02x}".format(i).upper()))
    plt.savefig("Fig/数据集.png")
    plt.show()


if __name__ == '__main__':
    i_type = 4  # 总共32种状态，依次设置并采集
    str_bin = bin(i_type + 0xf000)
    print(str_bin[-5:])
    for i in range(2):
        time.sleep(1)
        print("\r{:d}".format(12 - i), end="")
    print("")
    thread1 = Thread(target=open_visa_save_data, args=(i_type,))
    thread1.start()
    # view_data()
    #
    # view_saved_data()
