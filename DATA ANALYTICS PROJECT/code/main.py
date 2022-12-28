# Import required libraries
import math
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter,resample
import pyfftw
import matplotlib.pyplot as plt
from band_pass_filtering import band_pass_filtering
from compute_vitals import vitals,vitals_ECG
from detect_apnea_events import apnea_events
from detect_body_movements import detect_patterns
from detect_peaks import detect_peaks
from modwt_matlab_fft import modwt
from modwt_mra_matlab_fft import modwtmra
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,mean_absolute_error
from remove_nonLinear_trend import remove_nonLinear_trend
from data_subplot import data_subplot
from heartpy import process
import neurokit2 as nk
# ======================================================================================================================

# Main program starts here
print('\nstart processing ...')
files=['DATA ANALYTICS PROJECT\data\X1001.csv',
'DATA ANALYTICS PROJECT\data\X1002.csv',
'DATA ANALYTICS PROJECT\data\X1003.csv',
'DATA ANALYTICS PROJECT\data\X1004.csv',
'DATA ANALYTICS PROJECT\data\X1005.csv',
'DATA ANALYTICS PROJECT\data\X1006.csv',
'DATA ANALYTICS PROJECT\data\X1007.csv',
'DATA ANALYTICS PROJECT\data\X1008.csv',
'DATA ANALYTICS PROJECT\data\X1009.csv',
'DATA ANALYTICS PROJECT\data\X1010.csv',
'DATA ANALYTICS PROJECT\data\X1011.csv',
'DATA ANALYTICS PROJECT\data\X1012.csv',
'DATA ANALYTICS PROJECT\data\X1013.csv',
'DATA ANALYTICS PROJECT\data\X1014.csv',
'DATA ANALYTICS PROJECT\data\X1019.csv',
'DATA ANALYTICS PROJECT\data\X1020.csv',
'DATA ANALYTICS PROJECT\data\X1021.csv',
'DATA ANALYTICS PROJECT\data\X1022.csv',
'DATA ANALYTICS PROJECT\data\X1023.csv',
'DATA ANALYTICS PROJECT\data\X1024.csv',
'DATA ANALYTICS PROJECT\data\X1025.csv',
'DATA ANALYTICS PROJECT\data\X1026.csv',
'DATA ANALYTICS PROJECT\data\X1027.csv',
'DATA ANALYTICS PROJECT\data\X1028.csv',
'DATA ANALYTICS PROJECT\data\X1029.csv',
'DATA ANALYTICS PROJECT\data\X1030.csv',
'DATA ANALYTICS PROJECT\data\X1031.csv',
'DATA ANALYTICS PROJECT\data\X1033.csv',
'DATA ANALYTICS PROJECT\data\X1034.csv',
'DATA ANALYTICS PROJECT\data\X1035.csv',
'DATA ANALYTICS PROJECT\data\X1037.csv',
'DATA ANALYTICS PROJECT\data\X1038.csv',
'DATA ANALYTICS PROJECT\data\X1039.csv',
'DATA ANALYTICS PROJECT\data\X1040.csv',
'DATA ANALYTICS PROJECT\data\X1042.csv',
'DATA ANALYTICS PROJECT\data\X1043.csv',
'DATA ANALYTICS PROJECT\data\X1044.csv',
'DATA ANALYTICS PROJECT\data\X1046.csv',
'DATA ANALYTICS PROJECT\data\X1047.csv',
'DATA ANALYTICS PROJECT\data\X0132.csv']
 



def calculate_ECG_and_BCG_Heart_Rate(file):
    if file.endswith(".csv"):
        fileName = os.path.join(file)
        if os.stat(fileName).st_size != 0:
            
            all_data=pd.read_csv(fileName,sep=",",header=None,skiprows=1).values

            data_BCG=all_data[:,5]
            data_ECG=all_data[:,2]

            resampling_ratio = int(50/1000*len(data_BCG))
            resampled_BCG = resample(data_BCG,resampling_ratio)
            resampled_ECG = resample(data_ECG,resampling_ratio)
            
        
            t1, t2,  window_shift = 0, 500, 500
            w_ECG = modwt(resampled_ECG, 'bior3.9', 5)
            dc_ECG = modwtmra(w_ECG, 'bior3.9')
            wavelet_cycle_ECG = dc_ECG[5]
            limit_ECG = int(math.floor(resampled_ECG.size / window_shift))
            ECG_Heart_Rates=np.around(vitals_ECG(t1,t2,window_shift,limit_ECG,wavelet_cycle_ECG))
            print("\t\t\t---------------------------------------------")
            print("\t\t\t============ ECG HEART RATE ===============")
            print("\t\t\t\tHeart Rate : ",ECG_Heart_Rates)



            w_BCG = modwt(resampled_BCG, 'bior3.9', 4)
            dc_BCG = modwtmra(w_BCG, 'bior3.9')
            wavelet_cycle_BCG = dc_BCG[4]
            t1, t2,  window_shift = 0, 500, 500
            limit_BCG = int(math.floor(resampled_BCG.size / window_shift))
            beats_BCG = np.around(vitals(t1, t2, window_shift, limit_BCG,   wavelet_cycle_BCG))
            print("\t\t\t---------------------------------------------")
            print("\t\t\t============ BCG HEART RATE ===============")
            print('\t\t\t\tHeart Rate : ', beats_BCG)
            print("\t\t\t---------------------------------------------")


            return ECG_Heart_Rates,beats_BCG
            
            
# ================================================== Patients Heart Rates ===================================================================

def calculate_errors(ecg,bcg):
    n = len(ecg)
    sum = 0
    for i in range(n):
        sum += abs(ecg[i] - bcg[i])
    Mean_Absolute_error = sum/n

    Mean_Squared_Error = np.square(np.subtract(ecg,bcg)).mean()

    sum=0
    for i in range(n):
        sum+=abs(ecg[i] - bcg[i])/ecg[i]
    Mean_Absolute_Percentage_Error=sum/n

    return Mean_Absolute_error,Mean_Squared_Error,Mean_Absolute_Percentage_Error


for i in range(0,len(files)):
    print("\n\t\t=================================================================")
    print("\t\t===================== Patient",i+1, "=================================") 
    print("\t\t=================================================================")
    ecg,bcg=calculate_ECG_and_BCG_Heart_Rate(files[i])
    Mean_Absolute_Error,Mean_Squared_Error,Mean_Absolute_Percentage_Error=calculate_errors(ecg,bcg)
    print("\t\tMean absolute error : " + str(Mean_Absolute_Error))
    print("\t\tMean absolute persentage error : ", Mean_Absolute_Percentage_Error)
    print("\t\tMean squared error : ", Mean_Squared_Error)

    

print('\nEnd processing ...')     
        # print(Heart_rate_BCG)
        

        

       