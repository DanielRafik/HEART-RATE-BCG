import numpy as np

from detect_peaks import detect_peaks


def compute_rate(peaks):

    diff_sample = peaks[-1] - peaks[0] + 1
    t_N = diff_sample / 50
    heartRate = (len(peaks) - 1) / t_N * 60
    return heartRate
    
