"""
Created on %(25/09/2017)
Function to compute vitals, i.e., heart rate and respiration.
"""

import numpy as np
from detect_peaks import detect_peaks
from beat_to_beat import compute_rate


def vitals(t1, t2, win_size, window_limit, sig):
    all_rate = []
    for j in range(0, window_limit):
        sub_signal = sig[t1:t2]
        indices=detect_peaks(sub_signal,mpd=1)
        rate = compute_rate(indices)
        all_rate.append(rate)
        t1 = t2
        t2 += win_size
    all_rate = np.vstack(all_rate).flatten()
    return all_rate
