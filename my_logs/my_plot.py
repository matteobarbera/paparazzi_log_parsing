import numpy as np
from matplotlib import pyplot as plt

from parselog import parselog


def main_plot(filename):
    log_data = parselog(filename)
    ac_data = log_data.aircrafts[0].data
    # ac_msgs = log_data.msgs
    print(ac_data.keys())
    # print(ac_data.IMU_GYRO.keys()); quit()

    # gt, _, gp_alt, _, gq_alt, _, gr_alt = ac_data.IMU_GYRO.values()
    # plt.figure("Gyro")
    # plt.plot(gt, gp_alt, label='gp')
    # plt.plot(gt, gq_alt, label='gq')
    # plt.plot(gt, gr_alt, label='gr')
    # plt.xlabel("Time [s]")
    # plt.ylabel("Rotation [deg/s]")
    # plt.legend(loc=1)
    #
    # act_t = ac_data.ACTUATORS.timestamp
    # act_v = ac_data.ACTUATORS.values[:, :4]
    # act_0 = ac_data.ACTUATORS.values[:, 0]
    # act_1 = ac_data.ACTUATORS.values[:, 1]
    # act_2 = ac_data.ACTUATORS.values[:, 2]
    # act_3 = ac_data.ACTUATORS.values[:, 3]
    #
    # plt.figure("Actuators")
    # plt.plot(act_t, act_0, label="ch 0")
    # plt.plot(act_t, act_1, label="ch 1")
    # plt.plot(act_t, act_2, label="ch 2")
    # plt.plot(act_t, act_3, label="ch 3")
    # plt.xlabel("Time [s]")
    # plt.ylabel("PWM value")
    # plt.legend(loc=1)
    #
    gps_t = ac_data.GPS.timestamp
    gps_cl = ac_data.GPS.climb * 0.01  # to m/s
    gps_alt = ac_data.GPS.alt * 0.001
    plt.figure("GPS Climb")
    plt.plot(gps_t, gps_cl)
    plt.xlabel("Time [s]")
    plt.ylabel("Climb rate [m/s]")

    plt.figure("GPS Altitude")
    plt.plot(gps_t, gps_alt)
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude [m]")
    #
    # imu_t, imu_ax, imu_ay, imu_az = ac_data.IMU_ACCEL.values()
    # plt.figure("IMU Accelerations")
    # plt.plot(imu_t, imu_ax, label='ax')
    # plt.plot(imu_t, imu_ay, label='ay')
    # plt.plot(imu_t, imu_az, label='az')
    # plt.xlabel("Time [s]")
    # plt.ylabel("Acceleration [m/s^2]")
    # plt.legend()
    #
    # att_t = ac_data.ATTITUDE.timestamp
    # att_phi = ac_data.ATTITUDE.phi
    # att_theta = ac_data.ATTITUDE.theta
    # att_psi = ac_data.ATTITUDE.psi
    #
    # plt.figure("Attitude")
    # plt.plot(att_t, np.rad2deg(att_phi), label="phi")
    # plt.plot(att_t, np.rad2deg(att_theta), label="theta")
    # plt.plot(att_t, np.rad2deg(att_psi), label="psi")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Angle [deg]")
    # plt.legend()

    rc_t = ac_data.RC.timestamp
    rc_val = ac_data.RC.values[:, 6]
    plt.figure("RC")
    plt.plot(rc_t, rc_val, label="val")
    plt.xlabel("Time [s]")
    plt.ylabel("Channel value")
    plt.legend()


def plot_spin(filename: str, interval: tuple, num: int = 0):
    log_data = parselog(filename)
    ac_data = log_data.aircrafts[0].data

    s_start, s_end = interval

    fig, axs = plt.subplots(4, 2)
    fig.suptitle(f"Spin {num}")

    gt, _, gp_alt, _, gq_alt, _, gr_alt = ac_data.IMU_GYRO.values()
    gyro_mask = (gt > s_start) & (gt < s_end)
    axs[0, 0].plot(gt[gyro_mask], gp_alt[gyro_mask], label='gp')
    axs[0, 0].plot(gt[gyro_mask], gq_alt[gyro_mask], label='gq')
    axs[0, 0].plot(gt[gyro_mask], gr_alt[gyro_mask], label='gr')
    axs[0, 0].set(xlabel="Time [s]", ylabel="Rotation [deg/s]")
    axs[0, 0].legend(loc=1)

    act_t = ac_data.ACTUATORS.timestamp
    # act_v = ac_data.ACTUATORS.values[:, :4]
    act_0 = ac_data.ACTUATORS.values[:, 0]
    act_1 = ac_data.ACTUATORS.values[:, 1]
    act_2 = ac_data.ACTUATORS.values[:, 2] + 120  # add trim
    act_3 = ac_data.ACTUATORS.values[:, 3]

    act_mask = (act_t > s_start) & (act_t < s_end)

    # plt.figure(f"Actuators - S: {num}")
    # plt.subplot(422)
    axs[1, 0].plot(act_t[act_mask], act_0[act_mask], label="thr l")
    axs[1, 0].plot(act_t[act_mask], act_1[act_mask], label="ele l")
    axs[1, 0].plot(act_t[act_mask], act_2[act_mask], label="ele r")
    axs[1, 0].plot(act_t[act_mask], act_3[act_mask], label="thr r")
    axs[1, 0].set(xlabel="Time [s]", ylabel="PWM Value")
    axs[1, 0].legend(loc=1)

    gps_t = ac_data.GPS.timestamp
    gps_cl = ac_data.GPS.climb * 0.01  # to m/s
    gps_alt = ac_data.GPS.alt * 0.001  # to m
    gps_spd = ac_data.GPS.speed * 0.01  # to m/s
    gps_mask = (gps_t > s_start) & (gps_t < s_end)
    axs[2, 0].plot(gps_t[gps_mask], gps_cl[gps_mask])
    axs[2, 0].set(xlabel="Time [s]", ylabel="Climb rate [m/s]")

    axs[3, 0].plot(gps_t[gps_mask], gps_alt[gps_mask])
    axs[3, 0].set(xlabel="Time [s]", ylabel="Altitude [m]")

    axs[0, 1].plot(gps_t[gps_mask], gps_spd[gps_mask])
    axs[0, 1].set(xlabel="Time [s]", ylabel="2D Speed [m/s]")

    imu_t, imu_ax, imu_ay, imu_az = ac_data.IMU_ACCEL.values()
    imu_mask = (imu_t > s_start) & (imu_t < s_end)
    axs[1, 1].plot(imu_t[imu_mask], imu_ax[imu_mask], label='ax')
    axs[1, 1].plot(imu_t[imu_mask], imu_ay[imu_mask], label='ay')
    axs[1, 1].plot(imu_t[imu_mask], imu_az[imu_mask], label='az')
    axs[1, 1].set(xlabel="Time [s]", ylabel="Acceleration [m/s^2]")
    axs[1, 1].legend()

    att_t = ac_data.ATTITUDE.timestamp
    att_phi = ac_data.ATTITUDE.phi
    att_theta = ac_data.ATTITUDE.theta
    att_psi = ac_data.ATTITUDE.psi
    att_mask = (att_t > s_start) & (att_t < s_end)

    axs[2, 1].plot(att_t[att_mask], np.rad2deg(att_phi)[att_mask], label="phi")
    axs[2, 1].plot(att_t[att_mask], np.rad2deg(att_theta)[att_mask], label="theta")
    # plt.plot(att_t[att_mask], np.rad2deg(att_psi)[att_mask], label="psi")
    axs[2, 1].set(xlabel="Time [s]", ylabel="Angle [deg]")
    axs[2, 1].legend()

    rc_t = ac_data.RC.timestamp
    rc_thr = ac_data.RC.values[:, 0]
    rc_roll = ac_data.RC.values[:, 1]
    rc_pitch = ac_data.RC.values[:, 2]
    rc_yaw = ac_data.RC.values[:, 3]
    rc_aux2 = ac_data.RC.values[:, 6]
    rc_mask = (rc_t > s_start) & (rc_t < s_end)
    axs[3, 1].plot(rc_t[rc_mask], rc_thr[rc_mask], label="thr")
    axs[3, 1].plot(rc_t[rc_mask], rc_roll[rc_mask], label="roll")
    axs[3, 1].plot(rc_t[rc_mask], rc_pitch[rc_mask], label="pitch")
    axs[3, 1].plot(rc_t[rc_mask], rc_yaw[rc_mask], label="yaw")
    axs[3, 1].plot(rc_t[rc_mask], rc_aux2[rc_mask], label="aux2")
    axs[3, 1].set(xlabel="Time [s]", ylabel="Channel value")
    axs[3, 1].legend()


if __name__ == "__main__":
    path_to_logs = "//home//matteo//Documents//MSc//Thesis//logs//23-6//decoded//"
    logs = {"fr_0009": "20_06_30__14_59_00_SD.log", "fr_0015": "20_06_30__16_49_48_SD.log",
            "fr_0016": "20_06_30__17_18_44_SD.log"}

    # main_plot(path_to_logs + logs["fr_0016"])
    fr_0009_spins = [(813, 826), (854, 860), (1015, 1029), (1062, 1078), (1130, 1157), (1199, 1222)]
    fr_0015_spins = [(423, 451), (484, 511), (568, 574), (595, 600), (649, 656), (703, 733), (767, 790)]
    fr_0016_spins = [(354, 370)]

    # for i in range(len(fr_0009_spins)):
    #     plot_spin(path_to_logs + logs["fr_0009"], fr_0009_spins[i], num=i)
    #
    # for i in range(len(fr_0015_spins)):
    #     plot_spin(path_to_logs + logs["fr_0015"], fr_0015_spins[i], num=i)

    for i in range(len(fr_0016_spins)):
        plot_spin(path_to_logs + logs["fr_0016"], fr_0016_spins[i], num=i+1)

    plt.show()
