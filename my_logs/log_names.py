path_to_flight_logs = "//home//matteo//Documents//MSc//Thesis//logs//Flight"
path_to_cz_logs = "//home//matteo//Documents//MSc//Thesis//logs//CyberZoo"
logs = {"fr_0004": "//16-9//decoded//20_09_23__16_35_53_SD.log",
        "fr_0009": "//23-6//decoded//20_06_30__14_59_00_SD.log",
        "fr_0015": "//23-6//decoded//20_06_30__16_49_48_SD.log",
        "fr_0016": "//23-6//decoded//20_06_30__17_18_44_SD.log",
        "fr_0076": "//29-4//decoded//80_01_10__16_48_35_SD.log",
        "fr_0077": "//3-5//decoded//80_01_07__15_15_48_SD.log",
        "fr_0130": "//9-6-21//decoded//21_06_16__16_03_57_SD.log",
        "fr_0131": "//9-6-21//decoded//21_06_16__16_12_04_SD.log",
        "fr_0132": "//9-6-21//decoded//21_06_16__16_47_17_SD.log",
        "fr_0133": "//9-6-21//decoded//21_06_16__16_31_57_SD.log",
        "fr_0134": "//9-6-21//decoded//21_06_16__16_22_54_SD.log",
        }

cz_logs = {"fr_0140": "//10-6-21//decoded//80_01_10__17_20_27_SD.log",
           "fr_0143": "//16-6-21//decoded//80_01_09__09_45_09_SD.log",  # CZ GPS heading test
           "fr_0145": "//16-6-21//decoded//80_01_09__10_27_27_SD.log",  # CZ GPS heading test
           "fr_0148": "//16-6-21//decoded//21_06_16__10_29_08_SD_no_GPS.log",  # CZ GPS heading test
           "fr_0149": "//16-6-21//decoded//80_01_09__13_53_27_SD.log",  # CZ GPS heading test
           "fr_0168": "//21-6-21//decoded//80_01_07__18_20_03_SD.log",  # weird heading shit
           "fr_0170": "//2-7-21//decoded//80_01_08__19_03_30_SD.log",
           "fr_0173": "//2-7-21//decoded//80_01_11__15_10_14_SD.log",  # complementary filter test, not good
           "fr_0174": "//2-7-21//decoded//80_01_11__15_40_35_SD.log",  # complementary filter test, not good
           }

for key, val in logs.items():
    logs[key] = path_to_flight_logs + val

for key, val in cz_logs.items():
    cz_logs[key] = path_to_cz_logs + val
