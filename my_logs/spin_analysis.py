from matplotlib import pyplot as plt

from my_plot import plot_spin, plot_spins
from spin_plot_tools import plot_spin_analysis

if __name__ == "__main__":
    path_to_logs = "//home//matteo//Documents//MSc//Thesis//logs"
    # logs = {"fr_0009": "//23-6//decoded//20_06_30__14_59_00_SD.log",
    # "fr_0015": "//23-6//decoded//20_06_30__16_49_48_SD.log"}
    logs = {"fr_0004": "//16-9//decoded//20_09_23__16_35_53_SD.log",
            "fr_0015": "//23-6//decoded//20_06_30__16_49_48_SD.log"}

    fr_0015_rough_spins = [(423, 451), (484, 511), (568, 574), (595, 600), (649, 656), (703, 733), (767, 790)]
    fr_0004_rough_spins = [(905, 928), (952, 982), (1020, 1043), (1086, 1115), (1173, 1177), (1196, 1217)]
    # plot_spins(path_to_logs + logs["fr_0004"], fr_0004_rough_spins)  # check timings
    spin_n = 1
    spin04 = True
    spin15 = False
    if spin_n is not None and spin04:
        plot_spin(path_to_logs + logs["fr_0004"], fr_0004_rough_spins[spin_n - 1], fig_name=spin_n)
    if spin_n is not None and spin15:
        plot_spin(path_to_logs + logs["fr_0015"], fr_0015_rough_spins[spin_n - 1], fig_name=spin_n)
    # ========= Spin analysis 16-9 ===============
    plt.show()
    quit()
    # --- Spin 1 ---
    spin1_init = (909, 912)
    spin1_stable_period1 = (912, 915)
    spin1_stable_period2 = (915, 918)
    spin1_stable_cutoff = (913, 915)
    spin1_elevon = (921, 923)

    if (spin_n is None or spin_n == 1) and spin04:
        # plot_spin_analysis(path_to_logs + logs["fr_0004"], spin1_init, fig_name="Init S1")
        plot_spin_analysis(path_to_logs + logs["fr_0004"], spin1_stable_period1, fig_name="S1F04 St1")
        plot_spin_analysis(path_to_logs + logs["fr_0004"], spin1_stable_period2, fig_name="S1F04 St1")
        # plot_spin_analysis(path_to_logs + logs["fr_0004"], spin1_stable_cutoff, fig_name="Stable c1 S1")
        plot_spin_analysis(path_to_logs + logs["fr_0004"], spin1_elevon, fig_name="S1F04 El1")

    # --- Spin 2 ---
    spin2_stable_1 = (960, 963)
    spin2_stable_2 = (963, 966)
    spin2_elevon_1 = (966, 968)
    spin2_elevon_2 = (971, 974)
    spin2_elevon_3 = (974, 977)

    if (spin_n is None or spin_n == 2) and spin04:
        plot_spin_analysis(path_to_logs + logs["fr_0004"], spin2_stable_1, max_spins=6, fig_name="S2F04 St1")
        # plot_spin_analysis(path_to_logs + logs["fr_0004"], spin2_stable_2, max_spins=6, fig_name="Stable 2 S2")
        plot_spin_analysis(path_to_logs + logs["fr_0004"], spin2_elevon_1, max_spins=6, fig_name="S2F04 El1")
        # plot_spin_analysis(path_to_logs + logs["fr_0004"], spin2_elevon_2, max_spins=6, fig_name="Elevon 2 S2")
        # plot_spin_analysis(path_to_logs + logs["fr_0004"], spin2_elevon_3, max_spins=6, fig_name="Elevon 3 S2")

    # --- Spin 3 ---
    spin3_stable_1 = (1029, 1031)
    spin3_stable_2 = (1031, 1032.5)
    spin3_elevon_1 = (1032, 1034)  # low elevon deflection
    spin3_elevon_2 = (1034, 1037)

    if (spin_n is None or spin_n == 3) and spin04:
        pass
        plot_spin_analysis(path_to_logs + logs["fr_0004"], spin3_stable_1, fig_name="16-9 Spin 3 No Elevon")
        plot_spin_analysis(path_to_logs + logs["fr_0004"], spin3_stable_2, fig_name="16-9 Spin 3 No Elevon")
        plot_spin_analysis(path_to_logs + logs["fr_0004"], spin3_elevon_1, fig_name="16-9 Spin 3 Elevon Active")
        plot_spin_analysis(path_to_logs + logs["fr_0004"], spin3_elevon_2, fig_name="16-9 Spin 3 Elevon Active")

    # --- Spin 4 ---
    spin4_unstable_defl = (1096, 1099)
    spin4_large_defl = (1100, 1102)
    spin4_stable_defl = (1104, 1107)
    spin4_weird_sect = (1107, 1109)

    if (spin_n is None or spin_n == 4) and spin04:
        plot_spin_analysis(path_to_logs + logs["fr_0004"], spin4_unstable_defl, max_spins=6, fig_name="S4F04 El1 unst rrate")
        # plot_spin_analysis(path_to_logs + logs["fr_0004"], spin4_large_defl, max_spins=6, fig_name="Large defl S4")
        plot_spin_analysis(path_to_logs + logs["fr_0004"], spin4_stable_defl, max_spins=6, fig_name="S4F04 El2 st rrate")
        # plot_spin_analysis(path_to_logs + logs["fr_0004"], spin4_weird_sect, max_spins=6, fig_name="Weird sec S4")

    # -- Spin 6 ---
    spin6_stable_1 = (1203, 1205)
    spin6_stable_2 = (1205, 1207)
    spin6_stable_3 = (1207, 1209)
    spin6_elevon_1 = (1209, 1211.5)
    spin6_elevon_2 = (1211, 1213)

    if (spin_n is None or spin_n == 6) and spin04:
        pass
        # plot_spin_analysis(path_to_logs + logs["fr_0004"], spin6_stable_1, max_spins=6, fig_name="Stable 1 S6")
        plot_spin_analysis(path_to_logs + logs["fr_0004"], spin6_stable_2, max_spins=6, fig_name="16-9 Spin 6 No Elevon")
        plot_spin_analysis(path_to_logs + logs["fr_0004"], spin6_stable_3, max_spins=6, fig_name="16-9 Spin 6 No Elevon")
        plot_spin_analysis(path_to_logs + logs["fr_0004"], spin6_elevon_1, max_spins=6, fig_name="16-9 Spin 6 Elevon Active")
        plot_spin_analysis(path_to_logs + logs["fr_0004"], spin6_elevon_2, max_spins=6, fig_name="16-9 Spin 6 Elevon Active")

    # Data from 23-6 for comparison
    # -- Spin 1 --
    spin1_el1 = (428, 430)
    spin1_el2 = (430, 432.8)
    spin1_el3 = (434, 436)
    spin1_el4 = (440, 442.8)
    spin1_st1 = (433, 434.4)
    spin1_st2 = (436, 437.5)
    if (spin_n is None or spin_n == 1) and spin15:
        plot_spin_analysis(path_to_logs + logs["fr_0015"], spin1_el1, max_spins=6, fig_name="S1F15 El1")
        # plot_spin_analysis(path_to_logs + logs["fr_0015"], spin1_el2, max_spins=6, fig_name="El 2")
        plot_spin_analysis(path_to_logs + logs["fr_0015"], spin1_el3, max_spins=6, fig_name="S1F15 El3")
        # plot_spin_analysis(path_to_logs + logs["fr_0015"], spin1_el4, max_spins=6, fig_name="El 4")
        plot_spin_analysis(path_to_logs + logs["fr_0015"], spin1_st1, max_spins=6, fig_name="S1F15 St1")
        # plot_spin_analysis(path_to_logs + logs["fr_0015"], spin1_st2, max_spins=6, fig_name="St 1")

    # -- Spin 2 --
    spin2_el1 = (494.5, 497)
    spin2_el2 = (498, 501)
    spin2_el3 = (508, 510)
    spin2_st1 = (488, 490)
    spin2_st2 = (501.3, 503.6)
    if (spin_n is None or spin_n == 2) and spin15:
        plot_spin_analysis(path_to_logs + logs["fr_0015"], spin2_el1, max_spins=6, fig_name="S2F15 El1")
        # plot_spin_analysis(path_to_logs + logs["fr_0015"], spin2_el2, max_spins=6, fig_name="El 2")
        # plot_spin_analysis(path_to_logs + logs["fr_0015"], spin2_el3, max_spins=6, fig_name="El 3")
        plot_spin_analysis(path_to_logs + logs["fr_0015"], spin2_st1, max_spins=6, fig_name="S2F15 St1")
        # plot_spin_analysis(path_to_logs + logs["fr_0015"], spin2_st2, max_spins=6, fig_name="St 2")

    # -- Spin 6 --
    spin6_el1 = (714, 717)
    spin6_el2 = (717, 719)
    spin6_el3 = (722, 724.5)
    spin6_st1 = (707.3, 709.3)
    spin6_st2 = (724.5, 725.6)
    if (spin_n is None or spin_n == 6) and spin15:
        plot_spin_analysis(path_to_logs + logs["fr_0015"], spin6_el1, max_spins=6, fig_name="S6F15 El1")
        plot_spin_analysis(path_to_logs + logs["fr_0015"], spin6_el2, max_spins=6, fig_name="S6F15 El2")
        # plot_spin_analysis(path_to_logs + logs["fr_0015"], spin6_el3, max_spins=6, fig_name="El 3")
        plot_spin_analysis(path_to_logs + logs["fr_0015"], spin6_st1, max_spins=6, fig_name="S6F15 St1")
        # plot_spin_analysis(path_to_logs + logs["fr_0015"], spin6_st2, max_spins=6, fig_name="St 2")

    # -- Spin 7 --
    spin7_el1 = (778.6, 780.6)
    spin7_el2 = (783, 785)
    spin7_st1 = (776, 778)
    spin7_st2 = (780.6, 782.5)
    spin7_st3 = (785, 786.5)
    if (spin_n is None or spin_n == 7) and spin15:
        plot_spin_analysis(path_to_logs + logs["fr_0015"], spin7_el1, max_spins=6, fig_name="S7F15 El1")
        plot_spin_analysis(path_to_logs + logs["fr_0015"], spin7_el2, max_spins=6, fig_name="S7F15 El2")
        plot_spin_analysis(path_to_logs + logs["fr_0015"], spin7_st1, max_spins=6, fig_name="S7F15 St1 pre defl")
        plot_spin_analysis(path_to_logs + logs["fr_0015"], spin7_st2, max_spins=6, fig_name="S7F15 St2 post defl")
        plot_spin_analysis(path_to_logs + logs["fr_0015"], spin7_st3, max_spins=6, fig_name="S7F15 St3")
    plt.show()
