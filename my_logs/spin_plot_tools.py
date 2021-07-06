import copy
from functools import wraps

import numpy as np
from matplotlib import pyplot as plt

from parselog import parselog, AttrDict


def extract_spin_data(filename: str, interval: tuple, *, single_revs=True, data_fields=None):
    """
    Extracts relevant spin data for the time interval specified

    @param filename: file string
    @param interval: tuple or list of tuples containing start_t and ent_t of time interval
    @param single_revs: whether to extract data for individual revolutions
    @param data_fields: only extract data from specified fields (list of dict keys of paparazzi log)
    @return: list of dicts containing data of each 'run'
    """
    log_data = parselog(filename)
    ac_data = log_data.aircrafts[0].data

    if single_revs:
        rev_intervals = get_rev_intervals(filename, *interval)
    else:
        # Extract data for single 'run' or for multiple runs
        if type(interval) == tuple:
            rev_intervals = [interval]
        else:
            rev_intervals = [i for i in interval]

    if data_fields is None:
        _data_fields = ac_data.keys()
    else:
        _data_fields = data_fields
        keys = list(ac_data.keys())
        for key in keys:
            if key not in _data_fields:
                del ac_data[key]

    data = []
    for spin_n, t in enumerate(rev_intervals):
        tmp = copy.deepcopy(ac_data)
        s_start, s_end = t
        for data_field in _data_fields:
            if 'timestamp' in ac_data[data_field].keys():
                mask = (ac_data[data_field].timestamp >= s_start) & (ac_data[data_field].timestamp < s_end)
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


def plot_gyro(ac_data: AttrDict, ax=None, *, interval: tuple = None):
    if ax is None:
        ax = plt.gca()

    gt, _, gp_alt, _, gq_alt, _, gr_alt = ac_data.IMU_GYRO.values()
    if interval is None:
        s_start = gt[0]
        s_end = gt[-1]
    else:
        s_start, s_end = interval

    gyro_mask = (gt > s_start) & (gt < s_end)
    ax.plot(gt[gyro_mask], gp_alt[gyro_mask], label='gp')
    ax.plot(gt[gyro_mask], gq_alt[gyro_mask], label='gq')
    ax.plot(gt[gyro_mask], gr_alt[gyro_mask], label='gr', zorder=1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Rotation [deg/s]")
    ax.legend(loc=1)
    ax.grid(which='both')

    return ax


def plot_actuators(ac_data: AttrDict, ax=None, *, interval: tuple = None):
    if ax is None:
        ax = plt.gca()

    act_t = ac_data.ACTUATORS.timestamp
    if interval is None:
        s_start = act_t[0]
        s_end = act_t[-1]
    else:
        s_start, s_end = interval

    act_0 = ac_data.ACTUATORS.values[:, 0]
    act_1 = ac_data.ACTUATORS.values[:, 1]
    act_2 = ac_data.ACTUATORS.values[:, 2]
    act_3 = ac_data.ACTUATORS.values[:, 3]

    act_mask = (act_t > s_start) & (act_t < s_end)

    ax.plot(act_t[act_mask], act_0[act_mask], label="thr l")
    ax.plot(act_t[act_mask], act_1[act_mask], label="ele l")
    ax.plot(act_t[act_mask], act_2[act_mask], label="ele r")
    ax.plot(act_t[act_mask], act_3[act_mask], label="thr r")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("PWM value")
    ax.legend(loc=1)
    ax.grid(which='both')

    return ax


def plot_magnetometer(ac_data: AttrDict, ax=None, *, interval: tuple = None):
    if ax is None:
        ax = plt.gca()

    mag_t = ac_data.IMU_MAG.timestamp
    if interval is None:
        s_start = mag_t[0]
        s_end = mag_t[-1]
    else:
        s_start, s_end = interval

    mag_mx = ac_data.IMU_MAG.mx
    mag_my = ac_data.IMU_MAG.my
    mag_mz = ac_data.IMU_MAG.mz
    mag_mask = (mag_t > s_start) & (mag_t < s_end)

    ax.plot(mag_t[mag_mask], mag_mx[mag_mask])
    ax.plot(mag_t[mag_mask], mag_my[mag_mask])
    ax.plot(mag_t[mag_mask], mag_mz[mag_mask])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mag")
    ax.grid(which='both')

    return ax


def plot_accelerometer(ac_data: AttrDict, ax=None, *, interval: tuple = None):
    if ax is None:
        ax = plt.gca()

    imu_t, imu_ax, imu_ay, imu_az = ac_data.IMU_ACCEL.values()
    if interval is None:
        s_start = imu_t[0]
        s_end = imu_t[-1]
    else:
        s_start, s_end = interval

    imu_mask = (imu_t > s_start) & (imu_t < s_end)
    ax.plot(imu_t[imu_mask], imu_ax[imu_mask], label='ax')
    ax.plot(imu_t[imu_mask], imu_ay[imu_mask], label='ay')
    ax.plot(imu_t[imu_mask], imu_az[imu_mask], label='az')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration [m/s^2]")
    ax.legend()
    ax.grid(which='both')

    return ax


def plot_attitude(ac_data: AttrDict, ax=None, *, interval: tuple = None, plot_psi: bool = False):
    if ax is None:
        ax = plt.gca()

    att_t = ac_data.ATTITUDE.timestamp
    if interval is None:
        s_start = att_t[0]
        s_end = att_t[-1]
    else:
        s_start, s_end = interval

    att_phi = ac_data.ATTITUDE.phi
    att_theta = ac_data.ATTITUDE.theta
    att_psi = ac_data.ATTITUDE.psi
    att_mask = (att_t > s_start) & (att_t < s_end)

    ax.plot(att_t[att_mask], np.rad2deg(att_phi)[att_mask], label="phi")
    ax.plot(att_t[att_mask], np.rad2deg(att_theta)[att_mask], label="theta")
    if plot_psi:
        ax.plot(att_t[att_mask], np.rad2deg(att_psi)[att_mask], label="psi")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle [deg]")
    ax.legend()
    ax.grid(which='both')

    return ax


def plot_gps_climb(ac_data: AttrDict, ax=None, *, interval: tuple = None):
    if ax is None:
        ax = plt.gca()

    gps_t = ac_data.GPS.timestamp
    if interval is None:
        s_start = gps_t[0]
        s_end = gps_t[-1]
    else:
        s_start, s_end = interval

    gps_cl = ac_data.GPS.climb * 0.01  # to m/s
    gps_mask = (gps_t > s_start) & (gps_t < s_end)
    ax.plot(gps_t[gps_mask], gps_cl[gps_mask])
    ax.set(xlabel="Time [s]", ylabel="Climb rate [m/s]")
    ax.grid(which='both')

    return ax


def plot_gps_altitude(ac_data: AttrDict, ax=None, *, interval: tuple = None):
    if ax is None:
        ax = plt.gca()

    gps_t = ac_data.GPS.timestamp
    if interval is None:
        s_start = gps_t[0]
        s_end = gps_t[-1]
    else:
        s_start, s_end = interval

    gps_alt = ac_data.GPS.alt * 0.001  # to m
    gps_mask = (gps_t > s_start) & (gps_t < s_end)
    ax.plot(gps_t[gps_mask], gps_alt[gps_mask])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude [m]")
    ax.grid(which='both')

    return ax


def plot_gps_speed(ac_data: AttrDict, ax=None, *, interval: tuple = None):
    if ax is None:
        ax = plt.gca()

    gps_t = ac_data.GPS.timestamp
    if interval is None:
        s_start = gps_t[0]
        s_end = gps_t[-1]
    else:
        s_start, s_end = interval

    gps_spd = ac_data["GPS"]["speed"] * 0.01  # to m/s
    gps_mask = (gps_t > s_start) & (gps_t < s_end)
    ax.plot(gps_t[gps_mask], gps_spd[gps_mask])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude [m]")
    ax.grid(which='both')

    return ax


def plot_rc(ac_data: AttrDict, ax=None, *, interval: tuple = None):
    if ax is None:
        ax = plt.gca()

    rc_t = ac_data.RC.timestamp
    if interval is None:
        s_start = rc_t[0]
        s_end = rc_t[-1]
    else:
        s_start, s_end = interval

    rc_thr = ac_data.RC.values[:, 0]
    rc_roll = ac_data.RC.values[:, 1]
    rc_pitch = ac_data.RC.values[:, 2]
    rc_yaw = ac_data.RC.values[:, 3]
    rc_aux2 = ac_data.RC.values[:, 6]
    rc_mask = (rc_t > s_start) & (rc_t < s_end)
    ax.plot(rc_t[rc_mask], rc_thr[rc_mask], label="thr")
    ax.plot(rc_t[rc_mask], rc_roll[rc_mask], label="roll")
    ax.plot(rc_t[rc_mask], rc_pitch[rc_mask], label="pitch")
    ax.plot(rc_t[rc_mask], rc_yaw[rc_mask], label="yaw")
    ax.plot(rc_t[rc_mask], rc_aux2[rc_mask], label="aux2")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Channel value")
    ax.legend()
    ax.grid(which='both')

    return ax


def plot_commands(ac_data: AttrDict, ax=None, *, interval: tuple = None):
    if ax is None:
        ax = plt.gca()

    comm_t = ac_data["COMMANDS"]["timestamp"]
    if interval is None:
        s_start = comm_t[0]
        s_end = comm_t[-1]
    else:
        s_start, s_end = interval

    comm_thr = ac_data["COMMANDS"]["values"][:, 0]
    comm_roll = ac_data["COMMANDS"]["values"][:, 1]
    comm_pitch = ac_data["COMMANDS"]["values"][:, 2]
    comm_yaw = ac_data["COMMANDS"]["values"][:, 3]
    comm_fs = ac_data["COMMANDS"]["values"][:, 4]
    comm_mask = (comm_t > s_start) & (comm_t < s_end)

    ax.plot(comm_t[comm_mask], comm_thr[comm_mask], label="thr")
    ax.plot(comm_t[comm_mask], comm_roll[comm_mask], label="roll")
    ax.plot(comm_t[comm_mask], comm_pitch[comm_mask], label="pitch")
    ax.plot(comm_t[comm_mask], comm_yaw[comm_mask], label="yaw")
    ax.plot(comm_t[comm_mask], comm_fs[comm_mask], label="fs_landing")
    ax.grid(which='both')
    ax.set(xlabel="Time [s]", ylabel="Commands value")
    ax.legend()

    return ax


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
def plot_spin_analysis(filename: str, interval: tuple, *, max_spins: int = 6, fig_name: str = ''):
    data_fields = ["IMU_GYRO", "ATTITUDE", "RC"]
    spin_data = extract_spin_data(filename, interval, data_fields=data_fields)

    if len(spin_data) > max_spins:
        spin_data = spin_data[:max_spins]

    offset = len(spin_data[0]["IMU_GYRO"]["timestamp"]) % len(spin_data[0]["ATTITUDE"]["timestamp"])
    stride = len(spin_data[0]["IMU_GYRO"]["timestamp"]) // len(spin_data[0]["ATTITUDE"]["timestamp"])
    if spin_data[0]["IMU_GYRO"]["timestamp"][offset] != spin_data[0]["ATTITUDE"]["timestamp"][0]:
        offset = stride - 1
    assert (spin_data[0]["IMU_GYRO"]["timestamp"][offset] == spin_data[0]["ATTITUDE"]["timestamp"][0])

    fig, axs = plt.subplots(3, 2)
    fig.canvas.set_window_title(fig_name + f' t: {interval}')
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

        ref_z = -1
        earth_pos = z_axis_plot(phi, theta, psi, ref_z=ref_z)

        psi = np.rad2deg(psi)
        phi = np.rad2deg(phi)
        theta = np.rad2deg(theta)

        axs[0, 1].plot(earth_pos[0, :], earth_pos[1, :], label=f"rev {i + 1}")
        axs[1, 1].plot(psi, phi, label=f'rev {i + 1}')
        axs[2, 1].plot(psi, theta, label=f'rev {i + 1}')

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

    # check sign of pilot input
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

    axs[0, 0].set_ylabel('gq [deg/s]')
    axs[0, 0].set_xlabel('psi [deg]')
    axs[1, 0].set_ylabel('gp [deg/s]')
    axs[1, 0].set_xlabel('psi [deg]')
    axs[2, 0].set_xlabel('gp [deg/s]')
    axs[2, 0].set_ylabel('gq [deg/s]')

    axs[0, 1].set_title(f'x-y position of a point {abs(ref_z)}m above IMU')
    axs[0, 1].set_xlabel('x [m]')
    axs[0, 1].set_ylabel('y [m]')
    axs[1, 1].set_ylabel('phi [deg]')
    axs[1, 1].set_xlabel('psi [deg]')
    axs[2, 1].set_ylabel('theta [deg]')
    axs[2, 1].set_xlabel('psi [deg]')
    for ax in axs.reshape(-1):
        ax.grid()
        ax.legend()


def z_axis_plot(phi: np.ndarray, theta: np.ndarray, psi: np.ndarray, *, ref_z=-1):
    # vec in body frame -> [0, 0, ref_z]
    earth_pos = ref_z * z_frame_transformation(phi, theta, psi)
    earth_pos = np.squeeze(earth_pos)  # remove single dimensional axis
    return earth_pos


def z_frame_transformation(phi: np.ndarray, theta: np.ndarray, psi: np.ndarray):
    # From Flight Dynamics Reader
    t_vec = np.array([[np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
                      [np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
                      [np.cos(phi) * np.cos(theta)]])
    return t_vec


def get_rev_intervals(filename: str, start_t: float, end_t: float, *, rec_run=False):
    """
    Calculates time intervals of individual revolutions for specified spin data time interval

    @param filename: file string
    @param start_t: time interval start
    @param end_t: time interval end
    @param rec_run: helper variable for recursive usage
    @return: zip of interval times for each individual spin
    """
    log_data = parselog(filename)
    ac_data = log_data.aircrafts[0].data

    interval_mask = (ac_data["ATTITUDE"]["timestamp"] >= start_t) & (ac_data["ATTITUDE"]["timestamp"] < end_t)
    att_t = ac_data["ATTITUDE"]["timestamp"][interval_mask]
    att_psi = ac_data["ATTITUDE"]["psi"][interval_mask]
    sign_psi = np.sign(att_psi)
    zero_crossing = ((np.roll(sign_psi, 1) - sign_psi) != 0).astype(bool)
    zero_crossing[0] = 0  # if beginning/end of array has different sign it will detect a sign change at idx 0
    rev_crossing = np.where(np.abs(att_psi[zero_crossing]) > 1.5)[0]  # ignore sign change occurring at half revolution

    # return revolution intervals in tuple format
    rev_times = (att_t[zero_crossing])[rev_crossing]

    zipped_rev_times = zip(rev_times[:-1], rev_times[1:])
    if len(rev_times) % 2 == 1 and rev_crossing[0] == 1 and not rec_run:
        padded_rev_times = get_rev_intervals(filename, start_t, end_t + 1, rec_run=True)
        previous_len = len(list(zipped_rev_times))
        zipped_rev_times = list(padded_rev_times)[:previous_len + 1]
    return zipped_rev_times


# @savefig_decorator()
def plot_axis_projection(filename: str, interval: tuple, *, fig_name: str = ''):
    spin_data = extract_spin_data(filename, interval, single_revs=False)[0]
    spin_ints = get_rev_intervals(filename, *interval)

    _, _, gp_alt, _, gq_alt, _, _ = spin_data["IMU_GYRO"].values()  # TODO use/add translation from GPS data?

    psi = np.squeeze(spin_data["ATTITUDE"]["psi"])
    phi = np.squeeze(spin_data["ATTITUDE"]["phi"])
    theta = np.squeeze(spin_data["ATTITUDE"]["theta"])

    earth_pos = z_axis_plot(phi, theta, psi)

    rc_roll, rc_pitch = spin_data["RC"]["values"][:, 1:3].T

    rot_c = []
    z_c = []
    rot_c_el = []
    z_c_el = []
    for t in spin_ints:
        s_start, s_end = t
        imu_mask = (spin_data["IMU_GYRO"]["timestamp"] > s_start) & (spin_data["IMU_GYRO"]["timestamp"] < s_end)
        att_mask = (spin_data["ATTITUDE"]["timestamp"] > s_start) & (spin_data["ATTITUDE"]["timestamp"] < s_end)
        rc_mask = (spin_data["RC"]["timestamp"] > s_start) & (spin_data["RC"]["timestamp"] < s_end)

        gp_avg = np.mean(gp_alt[imu_mask])
        gq_avg = np.mean(gq_alt[imu_mask])
        earth_pos_avg = np.mean(earth_pos[:, att_mask], axis=1)
        rot_c.append([gp_avg, gq_avg])
        z_c.append(earth_pos_avg)

        if any(abs(rc_roll[rc_mask]) > 5500) or any(abs(rc_pitch[rc_mask]) > 5500):
            rot_c_el.append([gp_avg, gq_avg])
            z_c_el.append(earth_pos_avg)

    rot_c = np.array(rot_c)
    rot_c_el = np.array(rot_c_el)
    z_c = np.array(z_c)
    z_c_el = np.array(z_c_el)

    linestyle = {'ls': '--', 'marker': 'o', 'fillstyle': 'none', 'markevery': 1}
    linestyle2 = {'marker': 'o', 'color': 'red'}
    fig, axs = plt.subplots(1, 2)
    fig.canvas.set_window_title(fig_name + f" t: {interval}")
    axs[0].plot(rot_c[:, 0], rot_c[:, 1], label="rot_c", **linestyle)
    axs[1].plot(z_c[:, 0], z_c[:, 1], label="TPP proj", **linestyle)
    axs[1].plot(z_c[0, 0], z_c[0, 1], label="start spin", marker='o', color='k')
    if len(rot_c_el) != 0:
        axs[0].scatter(rot_c_el[:, 0], rot_c_el[:, 1], label="elevons active", **linestyle2)
        axs[1].scatter(z_c_el[:, 0], z_c_el[:, 1], label="elevons active", **linestyle2)
    axs[0].set_xlabel("gp avg [deg/s]")
    axs[0].set_ylabel("gq avg [deg/s]")
    axs[1].set_xlabel("x avg [m]")
    axs[1].set_ylabel("y avg [m]")
    for ax in axs.reshape(-1):
        ax.grid()
        ax.legend()


if __name__ == "__main__":
    pass
