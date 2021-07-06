from matplotlib import pyplot as plt

from spin_plot_tools import plot_axis_projection
from my_plot import plot_spin

if __name__ == "__main__":
    path_to_logs = "//home//matteo//Documents//MSc//Thesis//logs"
    # logs = {"fr_0009": "20_06_30__14_59_00_SD.log", "fr_0015": "20_06_30__16_49_48_SD.log"}
    logs = {"fr_0004": "//16-9//decoded//20_09_23__16_35_53_SD.log",
            "fr_0015": "//23-6//decoded//20_06_30__16_49_48_SD.log"}

    fr_0015_rough_spins = [(423, 451), (484, 511), (568, 574), (595, 600), (649, 656), (703, 733), (767, 790)]
    fr_0004_rough_spins = [(905, 928), (952, 982), (1020, 1043), (1086, 1115), (1173, 1177), (1196, 1217)]

    fr_0004_spins = [(912, 921), (960, 977), (1028, 1036), (1093, 1108), (1204, 1212)]
    fr_0015_spins = [(427, 449), (488, 519), (707, 731), (772, 787)]

    # doesn't take translation of IMU into account
    # gps data logged 4 times slower than attitude
    for i, t in enumerate(fr_0004_spins):
        plot_axis_projection(path_to_logs + logs["fr_0004"], t, fig_name=f"S{i + 1}F04 Proj")
    for i, t in enumerate(fr_0015_spins):
        plot_axis_projection(path_to_logs + logs["fr_0015"], t, fig_name=f"S{i + 1}F15 Proj")

    # for i, t in enumerate(fr_0004_rough_spins):
    #     plot_spin(path_to_logs + logs["fr_0004"], t, fig_name=f"S{i + 1}F04 Overview")
    # for i, t in enumerate(fr_0015_rough_spins):
    #     plot_spin(path_to_logs + logs["fr_0015"], t, fig_name=f"S{i + 1}F15 Overview")
    # plt.show()

    # TODO
    # any way to improve readability of graph?
    # https://stackoverflow.com/questions/50360567/python-errorbar-with-varied-marker-size

