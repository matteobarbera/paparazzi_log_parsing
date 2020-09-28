import copy
from functools import wraps
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from parselog import parselog


def extract_spin_data(filename: str, intervals: List[tuple]):
    log_data = parselog(filename)
    ac_data = log_data.aircrafts[0].data

    data = []
    for spin_n, interval in enumerate(intervals):
        tmp = copy.deepcopy(ac_data)
        s_start, s_end = interval
        for data_field in ac_data.keys():
            if 'timestamp' in ac_data[data_field].keys():
                mask = (ac_data[data_field].timestamp > s_start) & (ac_data[data_field].timestamp < s_end)
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


def savefig_decorator(fname):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            plt.rcParams.update({'font.size': 16})
            func(*args, **kwargs)
            fig = plt.gcf()
            fig.set_size_inches((20, 14), forward=False)
            fig.tight_layout()
            fig.subplots_adjust(top=0.95)

            plt.savefig(f"{fname}_Spin{kwargs['num']}.png", bbox_inches='tight', transparent=True, dpi=300)

        return wrapper

    return decorator
