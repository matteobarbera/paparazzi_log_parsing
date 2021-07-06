from matplotlib import pyplot as plt
import numpy as np

from parselog import parselog
from spin_plot_tools import extract_spin_data
import log_names


def f(filename, interval=None):
    log_data = parselog(filename)
    ac_data = log_data.aircrafts[0].data
    if interval is not None:
        ac_data = extract_spin_data(filename, interval, single_revs=False)[0]
        # print(ac_data)
        # quit()
    # print(ac_data.keys())
    # print(ac_data.ACTUATORS.keys())
    # print(ac_data.COMMANDS.keys())
    # print(ac_data.FBW_STATUS.keys())
    # print(ac_data.RC.keys())
    print(ac_data.PPRZ_MODE.keys())

    # s_start = float("-inf")
    # s_end = float("inf")
    # if interval is not None:
    #     s_start, s_end = interval

    fig, axs = plt.subplots(2, 2)
    # fig.canvas.set_window_title(f"")
    # fig.suptitle(f"")

    # print(ac_data["ACTUATORS"])
    # quit()
    act_t = ac_data["ACTUATORS"]["timestamp"]
    # act_v = ac_data.ACTUATORS.values[:, :4]
    act_0 = ac_data["ACTUATORS"]["values"][:, 0]
    act_1 = ac_data["ACTUATORS"]["values"][:, 1]
    act_2 = ac_data["ACTUATORS"]["values"][:, 2]
    act_3 = ac_data["ACTUATORS"]["values"][:, 3]

    axs[0, 0].plot(act_t, act_0, label="thr l")
    axs[0, 0].plot(act_t, act_1, label="ele l")
    axs[0, 0].plot(act_t, act_2, label="ele r")
    axs[0, 0].plot(act_t, act_3, label="thr r")
    axs[0, 0].set(xlabel="Time [s]", ylabel="PWM Value")
    axs[0, 0].legend(loc=1)
    axs[0, 0].grid(which='both')

    comm_t = ac_data["COMMANDS"]["timestamp"]
    comm_thr = ac_data["COMMANDS"]["values"][:, 0]
    comm_roll = ac_data["COMMANDS"]["values"][:, 1]
    comm_pitch = ac_data["COMMANDS"]["values"][:, 2]
    comm_yaw = ac_data["COMMANDS"]["values"][:, 3]
    comm_fs = ac_data["COMMANDS"]["values"][:, 4]

    axs[0, 1].plot(comm_t, comm_thr, label="thr")
    axs[0, 1].plot(comm_t, comm_roll, label="roll")
    axs[0, 1].plot(comm_t, comm_pitch, label="pitch")
    axs[0, 1].plot(comm_t, comm_yaw, label="yaw")
    axs[0, 1].plot(comm_t, comm_fs, label="fs_landing")
    axs[0, 1].grid(which='both')
    axs[0, 1].set(xlabel="Time [s]", ylabel="Commands value")
    axs[0, 1].legend()

    rc_t = ac_data["RC"]["timestamp"]
    rc_thr = ac_data["RC"]["values"][:, 0]
    rc_roll = ac_data["RC"]["values"][:, 1]
    rc_pitch = ac_data["RC"]["values"][:, 2]
    rc_yaw = ac_data["RC"]["values"][:, 3]
    rc_aux2 = ac_data["RC"]["values"][:, 6]
    axs[1, 0].plot(rc_t, rc_thr, label="thr")
    axs[1, 0].plot(rc_t, rc_roll, label="roll")
    axs[1, 0].plot(rc_t, rc_pitch, label="pitch")
    axs[1, 0].plot(rc_t, rc_yaw, label="yaw")
    axs[1, 0].plot(rc_t, rc_aux2, label="aux2")
    axs[1, 0].set(xlabel="Time [s]", ylabel="Channel value")
    axs[1, 0].legend()
    axs[1, 0].grid(which='both')

    # sta_t = ac_data["FBW_STATUS"]["timestamp"]
    # sta_r = ac_data["FBW_STATUS"]["rc_status"]
    # axs[1, 1].plot(sta_t, sta_r, label="rc_status")
    # axs[1, 1].legend()

    sta_t = ac_data["PPRZ_MODE"]["timestamp"]
    sta_r = ac_data["PPRZ_MODE"]["ap_mode"]
    axs[1, 1].plot(sta_t, sta_r, label="ap_mode")
    axs[1, 1].legend()

    plt.show()


def b():
    log_data = parselog(log_names.cz_logs["fr_0168"])
    ac_data = log_data.aircrafts[0].data

    # print(ac_data.keys())
    # quit()

    gt, _, gp_alt, _, gq_alt, _, gr_alt = ac_data["IMU_GYRO"].values()

    plt.figure()
    # plt.plot(gt, gp_alt, label='gp')
    # plt.plot(gt, gq_alt, label='gq')
    plt.plot(gt, gr_alt, label='gr')
    plt.legend()

    mag_t = ac_data.IMU_MAG.timestamp
    mag_mx = ac_data.IMU_MAG.mx
    mag_my = ac_data.IMU_MAG.my
    mag_mz = ac_data.IMU_MAG.mz

    plt.figure()
    plt.plot(mag_t, mag_mx, label="mx")
    plt.plot(mag_t, mag_my, label="my")
    plt.plot(mag_t, mag_mz, label="mz")
    plt.legend()

    imu_t, imu_ax, imu_ay, imu_az = ac_data.IMU_ACCEL.values()
    plt.figure()
    plt.plot(imu_t, imu_ax, label='ax')
    plt.plot(imu_t, imu_ay, label='ay')
    plt.plot(imu_t, imu_az, label='az')
    plt.legend()

    act_t = ac_data.ACTUATORS.timestamp
    act_0 = ac_data.ACTUATORS.values[:, 0]
    act_1 = ac_data.ACTUATORS.values[:, 1]
    act_2 = ac_data.ACTUATORS.values[:, 2]
    act_3 = ac_data.ACTUATORS.values[:, 3]

    plt.figure("Actuators")
    plt.plot(act_t, act_0, label="ch 0")
    plt.plot(act_t, act_1, label="ch 1")
    plt.plot(act_t, act_2, label="ch 2")
    plt.plot(act_t, act_3, label="ch 3")
    plt.xlabel("Time [s]")
    plt.ylabel("PWM value")

    att_t = ac_data["ATTITUDE"]["timestamp"]
    # att_phi = ac_data["ATTITUDE"]["phi"]
    att_theta = ac_data["ATTITUDE"]["theta"]
    att_psi = ac_data["ATTITUDE"]["psi"]
    # att_psi[att_psi < 0] += np.pi * 2

    plt.figure()
    # plt.plot(att_t, np.rad2deg(att_phi))
    # plt.plot(att_t, np.rad2deg(att_theta))
    plt.plot(att_t, np.rad2deg(att_psi))
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # f(log_names.logs["fr_0131"], (185, 220))
    b()
