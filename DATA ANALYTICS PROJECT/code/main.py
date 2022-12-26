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

file_Patient_1 = 'DATA ANALYTICS PROJECT\data\X1001.csv'
file_Patient_2 = "DATA ANALYTICS PROJECT\data\X1002.csv"

def calculate_ECG_and_BCG_Heart_Rate(file):
    if file.endswith(".csv"):
        fileName = os.path.join(file)
        if os.stat(fileName).st_size != 0:
            
            all_data=pd.read_csv(fileName,sep=",",header=None,skiprows=1).values

            data_BCG=all_data[:,10]
            data_ECG=all_data[:,2]

            resampling_ratio = int(50/1000*len(data_BCG))

            resampled_BCG = resample(data_BCG,resampling_ratio)
            
            resampled_ECG = resample(data_ECG,resampling_ratio)
            
            wd, m = process(resampled_ECG.flatten(), 50)
            ECG_Heart_Rate=np.around(m["bpm"])

            print("\t---------------------------------------------")
            print("\t============ ECG HEART RATE ===============")
            print("\t\t",ECG_Heart_Rate)
            
            w = modwt(resampled_BCG, 'bior3.9', 4)
            dc = modwtmra(w, 'bior3.9')
            wavelet_cycle = dc[4]

            t1, t2,  window_shift = 0, 500, 500
            
            
            limit = int(math.floor(resampled_BCG.size / window_shift))

            beats = vitals(t1, t2, window_shift, limit,   wavelet_cycle)
           
            print("\t---------------------------------------------")
            print("\t============ BCG HEART RATE ===============")
            print('\n\t\tHeart Rate Information')
            print('\t\tMinimum pulse : ', np.around(np.min(beats)))
            print('\t\tMaximum pulse : ', np.around(np.max(beats)))
            print('\t\tAverage pulse : ', np.around(np.mean(beats)))
            return ECG_Heart_Rate,np.around(np.mean(beats))
            
            
# ================================================== Patient 1 ===================================================================
print("\n=================================================================")
print("===================== Patient 1 =================================") 
print("=================================================================")
ECG_HR_P1,BCG_HR_P1=calculate_ECG_and_BCG_Heart_Rate(file_Patient_1)    
print("\n=================================================================")
print("===================== Patient 2 =================================") 
print("=================================================================")
ECG_HR_P2,BCG_HR_P2=calculate_ECG_and_BCG_Heart_Rate(file_Patient_2)    



print('\nEnd processing ...')     
        # print(Heart_rate_BCG)
        

        

       