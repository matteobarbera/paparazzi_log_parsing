from matplotlib import pyplot as plt

from spin_plot_tools import plot_xy_vs_psi, plot_gq_gp_vs_psi

if __name__ == "__main__":
    path_to_logs = "//home//matteo//Documents//MSc//Thesis//logs//16-9//decoded//"
    # logs = {"fr_0009": "20_06_30__14_59_00_SD.log", "fr_0015": "20_06_30__16_49_48_SD.log",
    #         "fr_0016": "20_06_30__17_18_44_SD.log"}
    logs = {"fr_0004": "20_09_23__16_35_53_SD.log"}

    fr_0004_rough_spins = [(905, 928), (952, 982), (1020, 1043), (1086, 1115), (1173, 1177), (1196, 1217)]
    # plot_spins(path_to_logs + logs["fr_0004"], fr_0004_rough_spins)  # check timings

    # ========= Spin analysis 16-9 ===============
    fr_0004_spins = [(905, 928)]

    # Stable spin portion (spin 1)
    fr_0004_spins_red1 = [(916.17, 916.52), (916.53, 916.86), (916.86, 917.2), (917.19, 917.57), (917.54, 917.9)]
    # Elevon right cyclic deflection (spin 1)
    fr_0004_spins_red2 = [(920.74, 921.17), (921.17, 921.5), (921.5, 921.835), (921.835, 922.17), (922.17, 922.5),
                          (922.5, 922.82)]

    # for i in range(len(fr_0004_spins_red1)):
    #     intervals = fr_0004_spins_red1[:i+1]
    #     plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], intervals, fig_name="Stable S1")
    #
    # for i in range(len(fr_0004_spins_red2)):
    #     intervals = fr_0004_spins_red2[:i+1]
    #     plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], intervals, fig_name="ELE R S1")

    plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], fr_0004_spins_red1, fig_name="Stable S1")
    plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], fr_0004_spins_red2, fig_name="ELE R S1")

    plot_xy_vs_psi(path_to_logs + logs["fr_0004"], fr_0004_spins_red1, fig_name="Stable S1")
    plot_xy_vs_psi(path_to_logs + logs["fr_0004"], fr_0004_spins_red2, fig_name="ELE R S1")

    plt.show()
