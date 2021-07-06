import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy.signal import butter, filtfilt

import log_names
from spin_plot_tools import extract_spin_data, get_rev_intervals


def my_tsa(filename: str, interval: tuple, *, fig_name: str = ''):
    spin_data = extract_spin_data(filename, interval, single_revs=False)[0]
    spin_ints = get_rev_intervals(filename, *interval)

    _, _, gp_alt, _, gq_alt, _, _ = spin_data["IMU_GYRO"].values()

    psi = np.squeeze(spin_data["ATTITUDE"]["psi"])
    phi = np.squeeze(spin_data["ATTITUDE"]["phi"])
    theta = np.squeeze(spin_data["ATTITUDE"]["theta"])

    filt_ord = 1
    cutoff_freq = 0.1
    b, a = butter(filt_ord, cutoff_freq, btype='low', output='ba')

    p_t_series = []
    q_t_series = []
    phi_t_series = []
    theta_t_series = []
    for t in spin_ints:
        s_start, s_end = t
        imu_mask = (spin_data["IMU_GYRO"]["timestamp"] > s_start) & (spin_data["IMU_GYRO"]["timestamp"] < s_end)
        att_mask = (spin_data["ATTITUDE"]["timestamp"] > s_start) & (spin_data["ATTITUDE"]["timestamp"] < s_end)

        # p_t_series.append(gp_alt[imu_mask])
        # q_t_series.append(gq_alt[imu_mask])
        # phi_t_series.append(phi[att_mask])
        # theta_t_series.append((theta[att_mask]))

        p_t_series.append(filtfilt(b, a, gp_alt[imu_mask]))
        # q_t_series.append(filtfilt(b, a, gq_alt[imu_mask]))
        # phi_t_series.append(phi[att_mask])
        # theta_t_series.append((theta[att_mask]))

        # plt.plot(np.arange(0, .1 * len(q_t_series[0]), 0.1), q_t_series[0])
        # plt.plot(np.arange(0, .1 * len(q_t_series[0]), 0.1), filtfilt(b, a, q_t_series[0]))
        # plt.show()

    # From telemetry/fixedwing_flight_recorder.xml
    period_attitude = 0.1
    period_gyro = 0.6
    fs_att = 1 / period_attitude
    fs_gyro = 1 / period_gyro

    tsa(p_t_series)


def tsa(time_series, max_n=8):
    filt_ord = 3
    cutoff_freq = 0.01
    b, a = butter(filt_ord, cutoff_freq, btype='low', output='ba')

    min_l = len(time_series[0])
    f_ser = []
    idx_ser = []
    for i, t_ser in enumerate(time_series[:-1]):
        if i > max_n:
            break
        if (l := len(t_ser)) < min_l:
            # min_l = l
            continue
        t_ser = filtfilt(b, a, t_ser) / l
        # cfft = fft(t_ser)
        # real_fft = (2 / len(cfft)) * np.abs(cfft[:len(t_ser) // 2])
        # f_ser.append(real_fft)
        f_ser.append(fft(t_ser))
        idx_ser.append(i)
    # min_l = 1e8
    # for s in f_ser:
    #     if len(s) < min_l:
    #         min_l = len(s)
    f_ser = np.asarray([ser[:min_l] for ser in f_ser])
    ser_avg = np.mean(f_ser, axis=0)
    ser_avg_t = ifft(ser_avg) * len(ser_avg)

    # ser_var = np.mean(( - ser_avg_t) ** 2, axis=0)
    # print(f_ser.shape)
    # print(ser_avg.shape)
    # print(ser_var.shape)
    # ser_dev = np.sqrt(ser_var)

    # plt.plot(np.arange(0, 0.1 * len(ser_avg), 0.1), ser_avg)
    # plt.plot(np.arange(0, 0.1 * len(ser_avg), 0.1), ser_avg + 1.96 * ser_dev)
    # plt.plot(np.arange(0, 0.1 * len(ser_avg), 0.1), ser_avg - 1.96 * ser_dev)
    diff2 = np.zeros(len(ser_avg_t))
    plt.figure()
    for idx in idx_ser:
        t_ser = np.asarray(time_series[idx])
        # print(t_ser)
        # print(ser_avg_t)
        # diff2 += (t_ser - ser_avg_t) ** 2
        plt.plot(np.arange(0, .1 * len(t_ser), 0.1), t_ser, color='r')
    # var = (1 / (len(idx_ser))) * diff2
    # dev = np.sqrt(var)
    plt.plot(np.arange(0, .1 * len(ser_avg_t), 0.1), ser_avg_t, color='k')
    # plt.plot(np.arange(0, 0.1 * len(ser_avg), 0.1), ser_avg + 1.96 * dev, color='k', ls='--')
    # plt.plot(np.arange(0, 0.1 * len(ser_avg), 0.1), ser_avg - 1.96 * dev, color='k', ls='--')
    plt.show()


if __name__ == "__main__":
    # path_to_logs = "//home//matteo//Documents//MSc//Thesis//logs"
    # # logs = {"fr_0009": "20_06_30__14_59_00_SD.log", "fr_0015": "20_06_30__16_49_48_SD.log"}
    # logs = {"fr_0004": "//16-9//decoded//20_09_23__16_35_53_SD.log",
    #         "fr_0015": "//23-6//decoded//20_06_30__16_49_48_SD.log"}

    fr_0015_rough_spins = [(423, 451), (484, 511), (568, 574), (595, 600), (649, 656), (703, 733), (767, 790)]
    fr_0004_rough_spins = [(905, 928), (952, 982), (1020, 1043), (1086, 1115), (1173, 1177), (1196, 1217)]

    # fr_0004_spins = [(912, 921), (960, 977), (1028, 1036), (1093, 1108), (1204, 1212)]
    # fr_0015_spins = [(427, 449), (488, 519), (707, 731), (772, 787)]
    fr_0004_spins = [(912, 919), (960, 977), (1028, 1036), (1093, 1108), (1204, 1212)]
    fr_0015_spins = [(427, 449), (488, 519), (707, 731), (772, 787)]

    my_tsa(log_names.logs["fr_0004"], fr_0004_spins[0])
