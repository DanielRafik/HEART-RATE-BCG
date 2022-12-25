# Import required libraries
import math
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter,resample
import pyfftw
import matplotlib.pyplot as plt
from band_pass_filtering import band_pass_filtering
from compute_vitals import vitals
from detect_apnea_events import apnea_events
from detect_body_movements import detect_patterns
from detect_peaks import detect_peaks
from modwt_matlab_fft import modwt
from modwt_mra_matlab_fft import modwtmra
from remove_nonLinear_trend import remove_nonLinear_trend
from data_subplot import data_subplot
from heartpy import process
import neurokit2 as nk
# ======================================================================================================================

# Main program starts here
print('\nstart processing ...')

file = 'DATA ANALYTICS PROJECT\data\sample_data.csv'
file_name_newdata_BCG='DATA ANALYTICS PROJECT\data\FileName_LC_BCG3.csv'
file_name_newdata_ECG="DATA ANALYTICS PROJECT\data\FileName_ECG.csv"

if file.endswith(".csv"):
    fileName = os.path.join(file)
    if os.stat(fileName).st_size != 0:
        rawData = pd.read_csv(fileName, sep=",", header=None, skiprows=1).values

        data_BCG=pd.read_csv(file_name_newdata_BCG)
        data_ECG=pd.read_csv(file_name_newdata_ECG)

        utc_time = rawData[:, 0]
        data_stream = rawData[:, 1]



        # start_point, end_point, window_shift, fs = 0, 500, 500, 50
        # ==========================================================================================================
        # data_stream, utc_time = detect_patterns(start_point, end_point, window_shift, data_stream, utc_time, plot=1)
        # ==========================================================================================================
        # BCG signal extraction
        # movement = band_pass_filtering(data_stream, fs, "bcg")
        # time=np.linspace(0,5000,5000,endpoint=False)

        movement_BCG=data_BCG.values
        print('------------------------------')
        print("============ BCG DATA ===============")
        print(movement_BCG)
        
        resampled_BCG = resample(movement_BCG,int(50/1000*len(movement_BCG)))
        print("---------------------------------------------")
        print("============ Resampled BCG DATA ===============")
        print(resampled_BCG)
        


        movement_ECG=data_ECG.values
        print('------------------------------')
        print("============ ECG DATA ===============")
        print(movement_ECG)
        
        resampled_ECG=resample(movement_ECG,int(50/1000*len(movement_ECG)))
        print("---------------------------------------------")
        print("============ Resampled ECG DATA ===============")
        print(resampled_ECG)
        

        wd, m = process(resampled_ECG.flatten(), 50)
        ECG_Heart_Rate=m["bpm"]
        print("---------------------------------------------")
        print("============ ECG HEART RATE ===============")
        print(ECG_Heart_Rate)



        w = modwt(resampled_BCG, 'bior3.9', 4)
        dc = modwtmra(w, 'bior3.9')
        wavelet_cycle = dc[4]

        t1, t2, window_length, window_shift = 0, 500, 500, 500
        sub_signal=wavelet_cycle[t1:t2]

        def heart_rate(peaks):
            diff_sample = peaks[-1] - peaks[0] + 1
            t_N = diff_sample / 50
            heartRate = (len(peaks) - 1) / t_N * 60
            return heartRate
        

        def vitals(t1, t2, win_size, window_limit, sig):
            all_rate = []
            for j in range(0, window_limit):
                sub_signal = sig[t1:t2]
                indices=detect_peaks(sub_signal,mpd=1)
                rate = heart_rate(indices)
                all_rate.append(rate)
                t1 = t2
                t2 += win_size
            all_rate = np.vstack(all_rate).flatten()
            return all_rate
        

        
        limit = int(math.floor(resampled_BCG.size / window_shift))

        beats = vitals(t1, t2, window_shift, limit,   wavelet_cycle)
        print("---------------------------------------------")
        print("============ BCG HEART RATE ===============")
        print('\nHeart Rate Information')
        print('Minimum pulse : ', np.around(np.min(beats)))
        print('Maximum pulse : ', np.around(np.max(beats)))
        print('Average pulse : ', np.around(np.mean(beats)))
        
        

        
        
        
        # print(Heart_rate_BCG)
        

        

        # movement_ECG=resample(movement_ECG,5000)
        # ==========================================================================================================
        # Respiratory signal extraction
        # breathing = band_pass_filtering(data_stream, fs, "breath")
        # breathing = remove_nonLinear_trend(breathing, 3)
        # breathing = savgol_filter(breathing, 11, 3)
        # # ==========================================================================================================
        
        # ==========================================================================================================
        # # Vital Signs estimation - (10 seconds window is an optimal size for vital signs measurement)
        # t1, t2, window_length, window_shift = 0, 500, 500, 500
        # hop_size = math.floor((window_length - 1) / 2)
        # limit = int(math.floor(breathing.size / window_shift))
        # # ==========================================================================================================
        # # Heart Rate
        # beats = vitals(t1, t2, window_shift, limit, wavelet_cycle, utc_time, mpd=1, plot=0)
        # print('\nHeart Rate Information')
        # print('Minimum pulse : ', np.around(np.min(beats)))
        # print('Maximum pulse : ', np.around(np.max(beats)))
        # print('Average pulse : ', np.around(np.mean(beats)))
        # # Breathing Rate
        # beats = vitals(t1, t2, window_shift, limit, breathing, utc_time, mpd=1, plot=0)
        # print('\nRespiratory Rate Information')
        # print('Minimum breathing : ', np.around(np.min(beats)))
        # print('Maximum breathing : ', np.around(np.max(beats)))
        # print('Average breathing : ', np.around(np.mean(beats)))
        # # ==============================================================================================================
        # thresh = 0.3
        # events = apnea_events(breathing, utc_time, thresh=thresh)
        # # ==============================================================================================================
        # # Plot Vitals Example
        # t1, t2 = 2500, 2500 * 2
        # data_subplot(data_stream, movement_BCG, breathing, wavelet_cycle, t1, t2)
        

        # ==============================================================================================================
    print('\nEnd processing ...')
    # ==================================================================================================================