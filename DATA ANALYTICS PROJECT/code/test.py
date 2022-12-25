import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter,resample
from detect_peaks import detect_peaks
from beat_to_beat import compute_rate


file_name_newdata_BCG='DATA ANALYTICS PROJECT\data\FileName_LC_BCG3.csv'
file_name_newdata_ECG="DATA ANALYTICS PROJECT\data\FileName_ECG.csv"

data_BCG=pd.read_csv(file_name_newdata_BCG)
data_ECG=pd.read_csv(file_name_newdata_ECG)

movement_BCG=data_BCG.values
movement_ECG=data_ECG.values

movement_BCG=resample(movement_BCG,int(50/1000*len(movement_BCG)))
movement_ECG=resample(movement_BCG,int(50/1000*len(movement_ECG)))


# movement=resample(movement,5000)
# print(compute_rate(movement_BCG, 1))
# time=np.linspace(0,5000,5000,endpoint=False)
plt.figure("BCG")
plt.plot(movement_BCG)
plt.figure("ECG")
plt.plot(movement_ECG)
plt.show()