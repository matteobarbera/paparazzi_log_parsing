import copy
from functools import wraps

import numpy as np
from matplotlib import pyplot as plt

from parselog import parselog


def extract_spin_data(filename: str, interval: tuple, single_revs=True):
    log_data = parselog(filename)
    ac_data = log_data.aircrafts[0].data

    if single_revs:
        rev_intervals = get_rev_intervals(filename, *interval)
    else:
        rev_intervals = interval

    data = []
    for spin_n, t in enumerate(rev_intervals):
        tmp = copy.deepcopy(ac_data)
        s_start, s_end = t
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


def savefig_decorator(fname=None):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            plt.rcParams.update({'font.size': 16})
            func(*args, **kwargs)
            fig = plt.gcf()
            fig.set_size_inches((20, 14), forward=False)
            fig.tight_layout()
            fig.subplots_adjust(top=0.95)

            if fname is not None:
                plt.savefig(f"savefigs//{fname}_Spin{kwargs['num']}.png", bbox_inches='tight', transparent=True, dpi=300)
            else:
                plt.savefig("savefigs//" + fig.canvas.get_window_title() + ".png", bbox_inches='tight', transparent=True, dpi=300)

        return wrapper

    return decorator


def plot_gq_gp_vs_psi(filename: str, interval: tuple, *, max_spins: int = 6, fig_name=None):
    spin_data = extract_spin_data(filename, interval)

    if len(spin_data) > max_spins:
        spin_data = spin_data[:max_spins]

    offset = len(spin_data[0]["IMU_GYRO"]["timestamp"]) % len(spin_data[0]["ATTITUDE"]["timestamp"])
    stride = len(spin_data[0]["IMU_GYRO"]["timestamp"]) // len(spin_data[0]["ATTITUDE"]["timestamp"])

    assert (spin_data[0]["IMU_GYRO"]["timestamp"][offset] == spin_data[0]["ATTITUDE"]["timestamp"][0])

    fig, axs = plt.subplots(3, 1)
    fig.canvas.set_window_title(fig_name)
    linestyle = {'ls': '--', 'marker': 'o', 'markevery': 1}
    linestyle2 = {'ls': 'None', 'marker': 'o', 'fillstyle': 'none', 'markevery': 1}

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

        # plot arrays
        axs[0].plot(psi, gq_alt_reduced, label=f'rev {i + 1}', **linestyle)
        axs[1].plot(psi, gp_alt_reduced, label=f'rev {i + 1}', **linestyle)
        axs[2].plot(gp_alt, gq_alt, label=f'rev {i + 1}', **linestyle2)
    axs[0].legend()
    axs[0].set_title('gq')
    axs[1].legend()
    axs[1].set_title('gp')
    axs[2].legend()
    axs[2].set_xlabel('gp')
    axs[2].set_ylabel('gq')


def plot_xy_vs_psi(filename: str, interval: tuple, *, max_spins: int = 6, fig_name=None):
    spin_data = extract_spin_data(filename, interval)

    if len(spin_data) > max_spins:
        spin_data = spin_data[:max_spins]

    fig, axs = plt.subplots(3, 1)
    fig.canvas.set_window_title(fig_name)
    for i, spin in enumerate(spin_data):
        phi = np.squeeze(spin["ATTITUDE"]["phi"])
        theta = np.squeeze(spin["ATTITUDE"]["theta"])
        psi = np.squeeze(spin["ATTITUDE"]["psi"])

        ref_z = -1  # vec in body frame -> [0, 0, ref_z]
        earth_pos = ref_z * z_frame_transformation(phi, theta, psi)
        earth_pos = np.squeeze(earth_pos)  # remove single dimensional axis

        axs[0].plot(earth_pos[0, :], earth_pos[1, :], label=f"rev {i + 1}")
        axs[1].plot(psi, phi, label="phi")
        axs[2].plot(psi, theta, label="theta")
    axs[0].legend()
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[1].legend()
    axs[2].legend()


# @savefig_decorator()
def plot_spin_analysis(filename: str, interval: tuple, *, max_spins: int = 6, fig_name: str = None):
    spin_data = extract_spin_data(filename, interval)

    if len(spin_data) > max_spins:
        spin_data = spin_data[:max_spins]

    offset = len(spin_data[0]["IMU_GYRO"]["timestamp"]) % len(spin_data[0]["ATTITUDE"]["timestamp"])
    stride = len(spin_data[0]["IMU_GYRO"]["timestamp"]) // len(spin_data[0]["ATTITUDE"]["timestamp"])

    assert (spin_data[0]["IMU_GYRO"]["timestamp"][offset] == spin_data[0]["ATTITUDE"]["timestamp"][0])

    fig, axs = plt.subplots(3, 2)
    fig.canvas.set_window_title(fig_name + f'\tt: {interval}')
    linestyle = {'ls': '--', 'marker': 'o', 'markevery': 1}
    linestyle2 = {'ls': 'None', 'marker': 'o', 'fillstyle': 'none', 'markevery': 1}
    for i, spin in enumerate(spin_data):
        # sampled at different rates
        _, _, gp_alt, _, gq_alt, _, _ = spin["IMU_GYRO"].values()
        gq_alt_reduced = gp_alt[offset::stride]
        gp_alt_reduced = gq_alt[offset::stride]

        psi = np.squeeze(spin["ATTITUDE"]["psi"])
        phi = np.squeeze(spin["ATTITUDE"]["phi"])
        theta = np.squeeze(spin["ATTITUDE"]["theta"])

        ref_z = -1  # vec in body frame -> [0, 0, ref_z]
        earth_pos = ref_z * z_frame_transformation(phi, theta, psi)
        earth_pos = np.squeeze(earth_pos)  # remove single dimensional axis

        psi = np.rad2deg(psi)
        phi = np.rad2deg(phi)
        theta = np.rad2deg(theta)

        axs[0, 1].plot(earth_pos[0, :], earth_pos[1, :], label=f"rev {i + 1}")
        axs[1, 1].plot(psi, phi, label=f'phi rev {i + 1}')
        axs[2, 1].plot(psi, theta, label=f'theta rev {i + 1}')

        # make sure lengths match
        if len(psi) < len(gp_alt_reduced):
            gp_alt_reduced = gp_alt_reduced[:-1]
            gq_alt_reduced = gq_alt_reduced[:-1]
        elif len(psi) < len(gp_alt_reduced):
            psi = psi[:-1]

        # plot arrays
        axs[0, 0].plot(psi, gq_alt_reduced, label=f'rev {i + 1}', **linestyle)
        axs[1, 0].plot(psi, gp_alt_reduced, label=f'rev {i + 1}', **linestyle)
        axs[2, 0].plot(gp_alt, gq_alt, label=f'rev {i + 1}', **linestyle2)
    else:
        try:
            # check pilot input
            rc_roll, rc_pitch = spin["RC"]["values"][:, 1:3].T

            roll_mask = abs(rc_roll) > 7500
            roll_weight = 10 * np.ones(len(rc_roll)) * roll_mask
            roll_weight = np.where(roll_weight == 0, 1, roll_weight)
            pitch_mask = abs(rc_pitch) > 7500
            pitch_weight = 10 * np.ones(len(rc_pitch)) * pitch_mask
            pitch_weight = np.where(pitch_weight == 0, 1, pitch_weight)

            rc_roll_avg = np.average(rc_roll, weights=roll_weight)
            rc_pitch_avg = np.average(rc_pitch, weights=pitch_weight)

            threshold = 7000
            title = fig.canvas.get_window_title()
            if rc_roll_avg > threshold:
                fig.canvas.set_window_title(title + " Rp")
            elif rc_roll_avg < -threshold:
                fig.canvas.set_window_title(title + " Rn")
            elif rc_pitch_avg > threshold:
                fig.canvas.set_window_title(title + " Pp")
            elif rc_pitch_avg < -threshold:
                fig.canvas.set_window_title(title + " Pn")
        except NameError:  # if for loop empty
            pass

    axs[0, 0].set_title('gq')
    axs[1, 0].set_title('gp')
    axs[2, 0].set_xlabel('gp')
    axs[2, 0].set_ylabel('gq')

    axs[0, 1].set_title(f'x-y position of a point {abs(ref_z)}m above IMU')
    axs[0, 1].set_xlabel("x")
    axs[0, 1].set_ylabel("y")
    for ax in axs.reshape(-1):
        ax.grid()
        ax.legend()


def z_frame_transformation(phi: np.ndarray, theta: np.ndarray, psi: np.ndarray):
    # From Flight Dynamics Reader
    t_vec = np.array([[np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
                      [np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
                      [np.cos(phi) * np.cos(theta)]])
    return t_vec


def get_rev_intervals(filename: str, start_t: float, end_t: float):
    log_data = parselog(filename)
    ac_data = log_data.aircrafts[0].data

    interval_mask = (ac_data["ATTITUDE"]["timestamp"] > start_t) & (ac_data["ATTITUDE"]["timestamp"] < end_t)
    att_t = ac_data["ATTITUDE"]["timestamp"][interval_mask]
    att_psi = ac_data["ATTITUDE"]["psi"][interval_mask]
    sign_psi = np.sign(att_psi)
    zero_crossing = ((np.roll(sign_psi, 1) - sign_psi) != 0).astype(bool)
    zero_crossing[0] = 0  # if beginning/end of array has different sign it will detect a sign change at idx 0
    rev_crossing = np.where(np.abs(att_psi[zero_crossing]) > 1.5)  # ignore sign change occurring at half revolution

    # return revolution intervals in tuple format
    rev_times = att_t[zero_crossing][rev_crossing]
    return zip(rev_times[:-1], rev_times[1:])