import copy
from functools import wraps
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from parselog import parselog


def extract_spin_data(filename: str, intervals: List[tuple]):
    log_data = parselog(filename)
    ac_data = log_data.aircrafts[0].data

    data = []
    for spin_n, interval in enumerate(intervals):
        tmp = copy.deepcopy(ac_data)
        s_start, s_end = interval
        for data_field in ac_data.keys():
            if 'timestamp' in ac_data[data_field].keys():
                mask = (ac_data[data_field].timestamp > s_start) & (ac_data[data_field].timestamp < s_end)
                tmp[data_field]['timestamp'] = ac_data[data_field]['timestamp'][mask]
                for key in ac_data[data_field].keys():
                    if key != 'timestamp':
                        tmp[data_field][key] = ac_data[data_field][key][mask]
        data.append(tmp)
    return data


def combine_masks(ms: np.ndarray):
    if len(ms) < 2:
        raise ValueError("There need to be at least two masks to combine")
    if len(ms) == 2:
        return np.ma.mask_or(ms[0], ms[1], shrink=False)
    else:
        comb_m = np.ma.mask_or(ms[0], ms[1], shrink=False)
        new_ms = np.vstack((comb_m, ms[2:]))
        return combine_masks(new_ms)


def savefig_decorator(fname):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            plt.rcParams.update({'font.size': 16})
            func(*args, **kwargs)
            fig = plt.gcf()
            fig.set_size_inches((20, 14), forward=False)
            fig.tight_layout()
            fig.subplots_adjust(top=0.95)

            plt.savefig(f"{fname}_Spin{kwargs['num']}.png", bbox_inches='tight', transparent=True, dpi=300)

        return wrapper

    return decorator


def plot_gq_gp_vs_psi(filename: str, intervals: List[tuple], fig_name=None):
    spin_data = extract_spin_data(filename, intervals)

    # spin1 = spin_data[0]
    # data logged at different frequencies!!
    # print(len(spin1["IMU_GYRO"]["timestamp"]) % len(spin1["ATTITUDE"]["timestamp"]))
    # print(len(spin1["IMU_GYRO"]["timestamp"][1:-1]) / len(spin1["ATTITUDE"]["timestamp"]))
    # print(len(spin1["IMU_GYRO"]["timestamp"][2::3]) == len(spin1["ATTITUDE"]["timestamp"]))
    #
    # print(spin1["IMU_GYRO"]["timestamp"][2:-1:3][:10])
    # print(spin1["ATTITUDE"]["timestamp"][:10])
    #
    # print(spin1["IMU_GYRO"]["timestamp"][2::3][-10:])
    # print(spin1["ATTITUDE"]["timestamp"][-10:])

    offset = len(spin_data[0]["IMU_GYRO"]["timestamp"]) % len(spin_data[0]["ATTITUDE"]["timestamp"])
    stride = len(spin_data[0]["IMU_GYRO"]["timestamp"]) // len(spin_data[0]["ATTITUDE"]["timestamp"])

    # TODO add check to make sure arrays match

    fig, axs = plt.subplots(3, 1)
    fig.canvas.set_window_title(fig_name)
    linestyle = {'ls': '--', 'marker': 'o', 'markevery': 1}
    linestyle2 = {'ls': 'None', 'marker': 'x', 'markevery': 1}

    # TODO Animate 3rd subplot
    # Have a look at the animate example on matplotlib documentation
    # init plot empty sets
    # store data in lists while plotting others
    # def function to replace the empty plots with the stored data
    # animate subplot

    for i, spin in enumerate(spin_data):
        # sampled at different rates
        _, _, gp_alt, _, gq_alt, _, _ = spin["IMU_GYRO"].values()
        gq_alt_reduced = gp_alt[offset::stride]
        gp_alt_reduced = gq_alt[offset::stride]
        psi = np.rad2deg(spin["ATTITUDE"]["psi"])

        # make sure lengths match
        if len(psi) < len(gp_alt_reduced):
            gp_alt_reduced = gp_alt_reduced[:-1]
            gq_alt_reduced = gq_alt_reduced[:-1]
        elif len(psi) < len(gp_alt_reduced):
            psi = psi[:-1]

        # sort arrays
        sort_mask = psi.argsort()
        psi_sorted = psi[sort_mask]
        gp_sorted = gp_alt_reduced[sort_mask]
        gq_sorted = gq_alt_reduced[sort_mask]

        # plot arrays
        axs[0].plot(psi_sorted, gp_sorted, label=f'rev {i + 1}', **linestyle)
        axs[1].plot(psi_sorted, gq_sorted, label=f'rev {i + 1}', **linestyle)
        axs[2].plot(gp_alt, gq_alt, label=f'rev {i + 1}', **linestyle2)
    axs[0].legend()
    axs[0].set_title('gq')
    axs[1].legend()
    axs[1].set_title('gp')
    axs[2].legend()
    axs[2].set_xlabel('gp')
    axs[2].set_ylabel('gq')


def plot_xy_vs_psi(filename: str, intervals: List[tuple], fig_name=None):
    spin_data = extract_spin_data(filename, intervals)

    plt.figure(fig_name)
    for i, spin in enumerate(spin_data):
        phi = np.squeeze(spin["ATTITUDE"]["phi"])
        theta = np.squeeze(spin["ATTITUDE"]["theta"])
        psi = np.squeeze(spin["ATTITUDE"]["psi"])

        ref_z = -1  # vec in body frame -> [0, 0, ref_z]
        earth_pos = ref_z * z_frame_transformation(phi, theta, psi)
        earth_pos = np.squeeze(earth_pos)  # remove single dimensional axis

        plt.plot(earth_pos[0, :], earth_pos[1, :], label=f"rev {i + 1}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")


def z_frame_transformation(phi: np.ndarray, theta: np.ndarray, psi: np.ndarray):
    # From Flight Dynamics Reader
    t_vec = np.array([[np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
                      [np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
                      [np.cos(phi) * np.cos(theta)]])
    return t_vec
