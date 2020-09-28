import copy
from typing import List

import numpy as np

from parselog import parselog


def extract_spin_data(filename: str, intervals: List[tuple]):
    log_data = parselog(filename)
    ac_data = log_data.aircrafts[0].data

    data = [copy.deepcopy(ac_data)] * len(intervals)
    for spin_n, interval in enumerate(intervals):
        s_start, s_end = interval
        for data_field in ac_data.keys():
            if 'timestamp' in ac_data[data_field].keys():
                mask = (ac_data[data_field].timestamp > s_start) & (ac_data[data_field].timestamp < s_end)
                data[spin_n][data_field]['timestamp'] = ac_data[data_field]['timestamp'][mask]
                for key in ac_data[data_field].keys():
                    if key != 'timestamp':
                        data[spin_n][data_field][key] = ac_data[data_field][key][mask]
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
