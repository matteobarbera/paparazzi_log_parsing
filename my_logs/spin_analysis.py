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
    # fr_0004_spins = [(905, 928)]
    fr_0004_spins = [(912, 914.8)]

    # --- Spin 1 ---
    spin1_init = (909, 912)
    spin1_stable_period1 = (912, 915)
    spin1_stable_period2 = (914, 918)
    spin1_stable_cutoff = (913, 915)
    spin1_elevon = (921, 922.9)

    # plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin1_init, fig_name="Init S1")
    # plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin1_stable_period1, fig_name="Stable p1 S1")
    # plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin1_stable_period2, fig_name="Stable p2 S1")
    # plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin1_stable_cutoff, fig_name="Stable c1 S1")
    # plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin1_elevon, fig_name="ELE R S1")
    #
    # plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin1_init, fig_name="Init S1")
    # plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin1_stable_period1, fig_name="Stable p1 S1")
    # plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin1_stable_period2, fig_name="Stable p2 S1")
    # plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin1_stable_cutoff, fig_name="Stable c1 S1")
    # plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin1_elevon, fig_name="ELE R S1")

    # --- Spin 2 ---
    spin2_stable_1 = (960, 963)
    spin2_stable_2 = (963, 966)
    spin2_elevon_1 = (965, 970)
    spin2_elevon_2 = (971, 974)
    spin2_elevon_3 = (974, 977)

    # plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin2_stable_1, max_spins=5, fig_name="Stable 1 S2")
    # plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin2_stable_2, max_spins=5, fig_name="Stable 2 S2")
    # plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin2_elevon_1, max_spins=5, fig_name="Elevon 1 S2")
    # plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin2_elevon_2, max_spins=5, fig_name="Elevon 2 S2")
    # plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin2_elevon_3, max_spins=5, fig_name="Elevon 3 S2")
    #
    # plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin2_stable_1, max_spins=5, fig_name="Stable 1 S2")
    # plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin2_stable_2, max_spins=5, fig_name="Stable 2 S2")
    # plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin2_elevon_1, max_spins=5, fig_name="Elevon 1 S2")
    # plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin2_elevon_2, max_spins=5, fig_name="Elevon 2 S2")
    # plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin2_elevon_3, max_spins=5, fig_name="Elevon 3 S2")

    # --- Spin 3 ---
    spin3_stable_1 = (1029, 1031)
    spin3_stable_2 = (1031, 1032.5)
    spin3_elevon_1 = (1032, 1034)  # low elevon deflection
    spin3_elevon_2 = (1034, 1037)

    # plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin3_stable_1, fig_name="Stable 1 S3")
    # plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin3_stable_2, fig_name="Stable 2 S3")
    # plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin3_elevon_1, fig_name="Elevon 1 S3")
    # plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin3_elevon_2, fig_name="Elevon 2 S3")
    #
    # plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin3_stable_1, fig_name="Stable 1 S3")
    # plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin3_stable_2, fig_name="Stable 2 S3")
    # plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin3_elevon_1, fig_name="Elevon 1 S3")
    # plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin3_elevon_2, fig_name="Elevon 2 S3")

    # --- Spin 4 ---
    spin4_large_angle = (1095, 1098)
    spin4_large_defl = (1100, 1102)
    spin4_stable_defl = (1104, 1107)
    spin4_weird_sect = (1107, 1109)

    # plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin4_large_angle, max_spins=6, fig_name="Large ang S4")
    # plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin4_large_defl, max_spins=6, fig_name="Large defl S4")
    # plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin4_stable_defl, max_spins=6, fig_name="Stable defl S4")
    # plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin4_weird_sect, max_spins=6, fig_name="Weird sec S4")
    #
    # plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin4_large_angle, max_spins=6, fig_name="Large ang S4")
    # plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin4_large_defl, max_spins=6, fig_name="Large defl S4")
    # plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin4_stable_defl, max_spins=6, fig_name="Stable defl S4")
    # plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin4_weird_sect, max_spins=6, fig_name="Weird sec S4")

    # -- Spin 6 ---
    spin6_stable_1 = (1203, 1205)
    spin6_stable_2 = (1205, 1207)
    spin6_stable_3 = (1207, 1209)
    spin6_elevon_1 = (1209, 1211)
    spin6_elevon_2 = (1211, 1213)

    plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin6_stable_1, max_spins=6, fig_name="Stable 1 S6")
    plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin6_stable_2, max_spins=6, fig_name="Stable 2 S6")
    plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin6_stable_3, max_spins=6, fig_name="Stable 3 S6")
    plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin6_elevon_1, max_spins=6, fig_name="Elevon 1 S6")
    plot_gq_gp_vs_psi(path_to_logs + logs["fr_0004"], spin6_elevon_2, max_spins=6, fig_name="Elevon 2 S6")

    plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin6_stable_1, max_spins=6, fig_name="Stable 1 S6")
    plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin6_stable_2, max_spins=6, fig_name="Stable 2 S6")
    plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin6_stable_3, max_spins=6, fig_name="Stable 3 S6")
    plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin6_elevon_1, max_spins=6, fig_name="Elevon 1 S6")
    plot_xy_vs_psi(path_to_logs + logs["fr_0004"], spin6_elevon_2, max_spins=6, fig_name="Elevon 2 S6")

    plt.show()
