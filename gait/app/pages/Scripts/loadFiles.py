import pandas as pd
import numpy as np
from . commonFunctions import *
import os
import sys
sys.path.append('calibration')
# from . calibration import *

class loadCsv():
    def __init__(self, data, resample):
        '''
        This function is used to create an object of the sensor data that contains the
        sample frequency, raw acceleration and corrected gyroscope ('rad/s'). This
        object will be further used in functions to store variables.
        It uses the working directory 'owd' that is set above and assumes you placed
        the files in the correct map: 'owd'/test_files/gait.
        The first 100 and last 100 samples are always skipped.
        The gyroscope is converted to rad/s.
        The gyroscope constant bias error is corrected using values determined in a
        static test. To add a new calibration file go to: main_files/add_calibration
        Samplefrequenty is calculated based on the IMU timestamps
        Possibility to plot the raw acceleration adn gyroscope signal.
        '''
        # serial              = pd.read_csv(owd,nrows = 1).loc[0][0]

        # print('data:')
        # print(owd)

        # print(data)
        # print(len(data))

        #
        # if resample:
        #     resampleFrequecy             = 0.01
        #     tmpsampleFreq                = (1/( 10000/np.mean(np.diff(data.iloc[:,0])) ))
        #     tmptime                      = np.array(data['T'])
        #     tmptime                      = (tmptime - tmptime[0] ) /10000
        #     newTime                      = np.arange(0,round(len(tmptime) * tmpsampleFreq,2),resampleFrequecy)
        #
        #     lindx = []
        #     # Resample to 100 HZ
        #     for i in newTime:
        #         idx = np.abs(i - tmptime).argmin()
        #         lindx.append(idx)
        #
        #     self.gyroscope               =  np.radians(np.array(data.iloc[lindx,4:7]))
        #     self.acceleration            = np.array(data.iloc[lindx,1:4])
        #     self.sampleFreq              = 0.01

        # else:
        self.sampleFreq              = 1 / 104
        self.acceleration            =  np.array(data.iloc[:,1:4])
        self.gyroscope               =  np.radians(np.array(data.iloc[:,4:7]))



        self.gyroscope          = self.gyroscope[200:-200]
        self.acceleration       = self.acceleration[200:-200]
