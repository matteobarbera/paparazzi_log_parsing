from typing import List

import numpy as np
from matplotlib import pyplot as plt

from parselog import parselog
from spin_plot_tools import *
import log_names


def main_plot(filename):
    log_data = parselog(filename)
    ac_data = log_data.aircrafts[0].data
    # ac_msgs = log_data.msgs
    # print(ac_data.keys())
    # print(log_data.msgs.telemetry.GPS.fields.speed.unit)
    # print(ac_data.GPS.keys()); quit()

    gt, _, gp_alt, _, gq_alt, _, gr_alt = ac_data.IMU_GYRO.values()
    plt.figure("Gyro")
    plot_gyro(ac_data)

    plt.figure("Actuators")
    plot_actuators(ac_data)

    # plt.figure("GPS Climb")
    # plot_gps_climb(ac_data)
    #
    # plt.figure("GPS Speed")
    # plot_gps_speed(ac_data, axs[0, 1])
    #
    # plt.figure("GPS Altitude")
    # plot_gps_altitude(ac_data)

    plt.figure("IMU Accelerations")
    plot_accelerometer(ac_data)

    plt.figure("Attitude")
    plot_attitude(ac_data)

    plt.figure("RC")
    plot_rc(ac_data)


@savefig_decorator()
def plot_spin(filename: str, interval: tuple, fig_name=''):
    # =====================================
    # Only use to save the figs
    # =====================================

    log_data = parselog(filename)
    ac_data = log_data.aircrafts[0].data

    s_start, s_end = interval

    fig, axs = plt.subplots(4, 2)
    fig.canvas.set_window_title(fig_name)
    # fig.suptitle(fig_name)

    plot_actuators(ac_data, ax=axs[0, 0], interval=(s_start, s_end))

    plot_gyro(ac_data, ax=axs[1, 0], interval=(s_start, s_end))

    plot_gps_climb(ac_data, ax=axs[2, 0], interval=(s_start, s_end))
    plot_gps_climb(ac_data, ax=axs[3, 0], interval=(s_start, s_end))

    plot_magnetometer(ac_data, ax=axs[0, 1], interval=(s_start, s_end))

    plot_accelerometer(ac_data, ax=axs[1, 1], interval=(s_start, s_end))

    plot_attitude(ac_data, ax=axs[2, 1], interval=(s_start, s_end))

    plot_rc(ac_data, ax=axs[3, 1], interval=(s_start, s_end))


def plot_spins(filename: str, intervals: List[tuple]):
    spin_data = extract_spin_data(filename, intervals, single_revs=False)
    for num, ac_data in enumerate(spin_data):
        fig, axs = plt.subplots(4, 2)
        fig.canvas.set_window_title(f"Spin {num + 1}")
        fig.suptitle(f"Spin {num + 1}")

        plot_actuators(ac_data, axs[0, 0])
        plot_gyro(ac_data, axs[1, 0])

        act_t = ac_data.ACTUATORS.timestamp
        act_2 = (ac_data["ACTUATORS"]["values"][:, 2])
        defl_spikes_up = np.where(act_2 > 1730)
        defl_spikes_down = np.where(act_2 < 1001)
        axs[1, 0].vlines(act_t[defl_spikes_up], -1000, 1000)
        axs[1, 0].vlines(act_t[defl_spikes_down], -1000, 1000, color='r')

        plot_gps_climb(ac_data, axs[2, 0])
        plot_gps_altitude(ac_data, axs[3, 0])
        plot_gps_speed(ac_data, axs[0, 1])
        plot_accelerometer(ac_data, axs[1, 1])
        plot_attitude(ac_data, axs[2, 1])
        axs[2, 1].vlines(act_t[defl_spikes_up], -180, 180)
        axs[2, 1].vlines(act_t[defl_spikes_down], -180, 180, color='r')

        plot_rc(ac_data, axs[3, 1])


if __name__ == "__main__":
    main_plot(log_names.logs["fr_0016"])
    plt.show()
    quit()

    # 23-6
    fr_0009_spins = [(813, 826), (854, 860), (1015, 1029), (1062, 1078), (1130, 1157), (1199, 1222)]
    fr_0015_spins = [(423, 451), (484, 511), (568, 574), (595, 600), (649, 656), (703, 733), (767, 790)]
    fr_0016_spins = [(354, 370)]

    # for i in range(len(fr_0009_spins)):
    #     plot_spin(path_to_logs + logs["fr_0009"], fr_0009_spins[i])

    # for i in range(len(fr_0015_spins)):
    #     plot_spin(path_to_logs + logs["fr_0015"], fr_0015_spins[i])

    # for i in range(len(fr_0016_spins)):
    #     plot_spin(path_to_logs + logs["fr_0016"], fr_0016_spins[i])

    # 16-9
    fr_0004_spins = [(905, 928), (952, 982), (1020, 1043), (1086, 1115), (1173, 1177), (1196, 1217)]

    # for i in range(len(fr_0004_spins)):
    #     plot_spin(path_to_logs + logs["fr_0004"], fr_0004_spins[i])

    # plot_spins(path_to_logs + logs["fr_0004"], fr_0004_spins)

    plt.show()
