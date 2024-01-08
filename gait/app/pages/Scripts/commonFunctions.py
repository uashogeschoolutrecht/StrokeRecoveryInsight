import numpy as np
from scipy import signal
import statsmodels.tsa.api as smt

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

def filt_band(input_signal,  samplefreq, cutoff_h,cutoff_l , order =1, n_input = 1 ):
    '''
    Butterworth bandpass filter
    '''
    filtacceleration = np.zeros((len(input_signal),n_input))
    try:
        for k in range(0,n_input):                                                 
            b, a = signal.butter(order, (2*cutoff_h)/(1/samplefreq), 'highpass');
            filtacceleration[:,k] = signal.filtfilt(b,a, input_signal[:,k])
            b, a = signal.butter(order, (2*cutoff_l)/(1/samplefreq), 'lowpass');
            filtacceleration[:,k] = signal.filtfilt(b,a, input_signal[:,k])
        return filtacceleration
    except IndexError:
        filtacceleration = np.zeros(len(input_signal))
        for k in range(0,n_input):                                                 
            b, a = signal.butter(order, (2*cutoff_h)/(1/samplefreq), 'highpass');
            filtacceleration[:] = signal.filtfilt(b,a, input_signal[:])
            b, a = signal.butter(order, (2*cutoff_l)/(1/samplefreq), 'lowpass');
            filtacceleration[:] = signal.filtfilt(b,a, input_signal[:])
        return filtacceleration

def filt_high(input_signal, samplefreq, cutoff, order =1, n_input = 1):
    '''
    Butterworth highpass filter
    '''                              
    filtacceleration = np.zeros((len(input_signal),n_input))
    try: 
        for k in range(0,n_input):
            b, a = signal.butter(order, (2*cutoff)/(1/samplefreq), 'highpass');
            filtacceleration = signal.filtfilt(b,a, input_signal[:,k])
        return filtacceleration
    except IndexError: 
         for k in range(0,n_input):
            b, a = signal.butter(order, (2*cutoff)/(1/samplefreq), 'highpass');
            filtacceleration[:,k] = signal.filtfilt(b,a, input_signal[:])
         return filtacceleration
     
def filt_low(input_signal, samplefreq, cutoff, order = 1,  n_input = 1): 
    '''
    Butterworth lowpass filter
    '''                                 
    filtacceleration = np.zeros((len(input_signal),n_input))
    try: 
        for k in range(0,n_input):
            b, a = signal.butter(order, (2*cutoff)/(1/samplefreq), 'lowpass');
            filtacceleration[:,k] = signal.filtfilt(b,a, input_signal[:,k])
        return filtacceleration
    except IndexError: 
        for k in range(0,n_input):
            b, a = signal.butter(order, (2*cutoff)/(1/samplefreq), 'lowpass');
            filtacceleration[:,k] = signal.filtfilt(b,a, input_signal[:])
        return filtacceleration
    
def plot1(signal, title = '', xlabel = '', ylabel = ''):                                       # 3 subplots
    fig1, ax2 = plt.subplots(3)
    ax2[0].plot(np.array(range(0,len(signal))), signal[:,0]) 
    ax2[1].plot(np.array(range(0,len(signal))), signal[:,1]) 
    ax2[2].plot(np.array(range(0,len(signal))), signal[:,2]) 
    ax2[0].set_title(title)
    ax2[2].set_xlabel(xlabel)
    ax2[0].set_ylabel(ylabel)
    ax2[1].set_ylabel(ylabel)
    ax2[2].set_ylabel(ylabel)

def plot2(signal1, signal2, title, xlabel, ylabel1, ylabel2):                   # 2 subplots 
    fig1, ax2 = plt.subplots(2)
    ax2[0].plot(np.array(range(0,len(signal1))), signal1) 
    ax2[1].plot(np.array(range(0,len(signal2))), signal2) 
    ax2[0].set_title(title)
    ax2[1].set_xlabel(xlabel)
    ax2[0].set_ylabel(ylabel1)
    ax2[1].set_ylabel(ylabel2)
    
def plot3(signal1, signal2, title, xlabel, ylabel, legend1, legend2):           # 1 plot 2 signals
    fig1, ax2 = plt.subplots()
    ax2.plot(np.array(range(0,len(signal1))), signal1 ) 
    ax2.plot(np.array(range(0,len(signal2))), signal2) 
    ax2.legend((legend1, legend2), loc='upper right')
    ax2.set_title(title)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    
def plot4(signal, title = '', xlabel = '', ylabel = ''):                                       # 1 plot 
    fig1, ax2 = plt.subplots()
    ax2.plot(np.array(range(0,len(signal))), signal) 
    ax2.set_title(title)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)

def plot5(signal1, signal2, signal3, title, xlabel, ylabel, legend1, legend2, legend3):           # 1 plot 3 signals
    fig1, ax2 = plt.subplots()
    ax2.plot(np.array(range(0,len(signal1))), signal1 ) 
    ax2.plot(np.array(range(0,len(signal2))), signal2) 
    ax2.plot(np.array(range(0,len(signal3))), signal3) 

    ax2.legend((legend1, legend2, legend3), loc='upper right')
    ax2.set_title(title)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    
def plot6(signalx, signaly, title = '', xlabel = '', ylabel = ''):                                       # 1 plot 
    fig1, ax2 = plt.subplots()
    ax2.plot(signalx, signaly) 
    ax2.set_title(title)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)    
    
def desciptives_acc(obj, signal, peaks, therapistinput, firststride ):                                                           # Calculates the acceleration range
    '''
    add descriptives to object
    Range, std and rms
    '''     
    count = int(np.floor(len(peaks[firststride:-firststride])/therapistinput))
    accrangeX = []
    accstdX= []
    accrmsX= []
    accrangeY= []
    accstdY= []
    accrmsY= []
    accrangeZ= []
    accstdZ= []
    accrmsZ= []
    for i in range(count):
        select = signal[peaks[firststride+therapistinput*i]:peaks[therapistinput+firststride+therapistinput*i+1]]
        accrangeX.append(np.max(select[:,0]) - np.min(select[:,0]))
        accstdX.append(np.std(select[:,0]))
        accrmsX.append(np.sqrt(np.mean(select[:,0]**2)))
        accrangeY.append(np.max(select[:,1]) - np.min(select[:,1])             )
        accstdY.append(np.std(select[:,1]))
        accrmsY.append(np.sqrt(np.mean(select[:,1]**2)))
        accrangeZ.append(np.max(select[:,2]) - np.min(select[:,2])         )
        accstdZ.append( np.std(select[:,2]))
        accrmsZ.append(np.sqrt(np.mean(select[:,2]**2)))
        
        
    obj.accrangeX = np.mean(  accrangeX )                              
    obj.accstdX =  np.mean( accstdX)
    obj.accrmsX =  np.mean( accrmsX)
    obj.accrangeY =       np.mean(  accrangeY )                 
    obj.accstdY = np.mean( accstdY  )  
    obj.accrmsY =  np.mean( accrmsY  )  
    obj.accrangeZ =    np.mean(   accrangeZ)                         
    obj.accstdZ = np.mean(  accstdZ )  
    obj.accrmsZ =  np.mean(  accrmsZ )  

def desciptives_gyr(obj, signal, peaks, therapistinput, firststride):            
    '''
    add descriptives to object
    Range, std and rms
    '''     
                                                 
    count = int(np.floor(len(peaks[firststride:-firststride])/therapistinput))
    gyrrangeX = []
    gyrstdX= []
    gyrrmsX= []
    gyrrangeY= []
    gyrstdY= []
    gyrrmsY= []
    gyrrangeZ= []
    gyrstdZ= []
    gyrrmsZ= []
    for i in range(count):
        select = signal[peaks[firststride+therapistinput*i]:peaks[therapistinput+firststride+therapistinput*i+1]]
        gyrrangeX.append(np.max(select[:,0]) - np.min(select[:,0]))
        gyrstdX.append(np.std(select[:,0]))
        gyrrmsX.append(np.sqrt(np.mean(select[:,0]**2)))
        gyrrangeY.append(np.max(select[:,1]) - np.min(select[:,1])             )
        gyrstdY.append(np.std(select[:,1]))
        gyrrmsY.append(np.sqrt(np.mean(select[:,1]**2)))
        gyrrangeZ.append(np.max(select[:,2]) - np.min(select[:,2])         )
        gyrstdZ.append( np.std(select[:,2]))
        gyrrmsZ.append(np.sqrt(np.mean(select[:,2]**2)))
        
        
    obj.gyrrangeX = np.mean(  gyrrangeX )                              
    obj.gyrstdX =  np.mean( gyrstdX)
    obj.gyrrmsX =  np.mean( gyrrmsX)
    obj.gyrrangeY =       np.mean(  gyrrangeY )                 
    obj.gyrstdY = np.mean( gyrstdY  )  
    obj.gyrrmsY =  np.mean( gyrrmsY  )  
    obj.gyrrangeZ =    np.mean(   gyrrangeZ)                         
    obj.gyrstdZ = np.mean(  gyrstdZ )  
    obj.gyrrmsZ =  np.mean(  gyrrmsZ )  
    
    
    
def desciptives_jerk(obj, signal):                                                           # Calculates the acceleration range
    obj.jerkrange = np.max(signal) - np.min(signal)                                  # standard deviation and rms
    obj.jerkstd = np.std(signal)
    obj.jerkrms = np.sqrt(np.mean(signal**2))


def RMS(signal):                                                                # Root mean square
    return np.sqrt(np.mean(signal**2))

def RMSE(signal):                                                               # Root mean square error
    return np.sqrt((sum((np.mean(signal) - signal)**2)/(len(signal)-1)))


def magnitude(signal):                                                          # Magnitude of a x-axis signal 
     return np.sqrt(np.sum(signal * signal, axis = 1))
 

    
def rotsig(signal, rotmatrix):

    '''
    Takes an nx3 matrix and rotates it using an 3x3 rotation matrix
    '''
    x = ( signal[:,0] * rotmatrix[0,0]) + ( signal[:,0] * rotmatrix[0,1])
    y = ( signal[:,1] * rotmatrix[1,0]) + ( signal[:,1] * rotmatrix[1,1])
    z = signal[:,2]
    rotsignal = np.transpose(np.array((x,y,z)))
    return rotsignal


def startEnd(obj, objtype, minDiff, maxDiff):
    '''
    We defined t he start and the end of the as a period of time (5 secondes) in 
    which the signal stayed below a specified accelerometer and gyroscope 
    threshold. If this was not found we included the part between the first
    and the last signal. Then we evaluate if the lenth of the signal 
    corresponds to the expected length. The function requires a 'foot' or 
    'lowback' argument as these require slightly different thresholds.
    '''

    print("nieuwe code")
    accMag             = magnitude(obj.acceleration)                           
    gyrMag             = magnitude(obj.gyroscope)                              
    threshAcc          = np.mean(accMag) + np.std(accMag)                       # Sets threshold
    if objtype == 'foot':
        threshGyr      = np.mean(gyrMag) 
    else:
        threshGyr      = np.mean(gyrMag) + np.std(gyrMag) * 0.2

    
    standphases        = []
    tmpCounter         = 0
    for counter, value in enumerate(accMag):                                    # Standphase is defined as >30 samples with a 
        if ((value < threshAcc) and (gyrMag[counter] < threshGyr)):             # acceleration maginitude and gyroscope magnitude low than
            tmpCounter += 1                                                     # This function can most likely be improved (!)
        else:
            if (tmpCounter > 30):
                standphases.append(np.array(range(counter-tmpCounter,counter)))
            tmpCounter = 0                
        if ((value == accMag[-1]) & (tmpCounter > 0)):
            standphases.append(np.array(range(counter-tmpCounter,counter)))
    
    try:
        statBegin = []
        for i in standphases[0:25]:
            statBegin.append(len(i))
            
        sortStatBegin = np.sort(statBegin)
        revStatBegin = sortStatBegin[::-1]
        nullStart = False
    except: 
        nullStart = True        
        start = np.array([0])
        
    try:
        statEnd = []
        for i in standphases[-25:len(standphases)]:         
            statEnd.append(len(i))
            
        sortStatEnd = np.sort(statEnd)
        revStatEnd = sortStatEnd[::-1]
        nullStop = False        

    except:
        stop = np.array([len(accMag)-1])
        nullStop = True       
    
    try:
        if ((nullStart == False ) & (nullStop == False )):
            try:   
                for j in revStatEnd:
                        for i in revStatBegin:
                            probableStart = standphases[np.where(statBegin == i)[0][0]]
                            probableStop = standphases[np.where(statEnd == j)[0][0] - 25 + len(standphases)] 
                            signalLength = len(np.arange(probableStart[-1],probableStop[0]))
                            if (len(probableStart)>100):
                                if ((signalLength > minDiff) & (signalLength < maxDiff)):
                                    start = probableStart
                                    stop = probableStop
                                    raise Error
            except Error:           
                if ((start[-1] - 100) > 0):
                    beginSample = start[-1] - 100
                elif ((start[-1] - 50) > 0):
                    beginSample = start[-1] - 50       
                elif ((start[-1] - 10) > 0):
                    beginSample = start[-1] - 10       
                else:
                    beginSample = start[-1]    
            
                if ((stop[0] + 100) < len(accMag)-1):
                    endSample = stop[0] + 100
                elif ((stop[0] + 50) < len(accMag)-1):
                    endSample = stop[0] + 50    
                elif ((stop[0] - 10) < len(accMag)-1):
                    endSample = stop[0] + 10
                else:
                    endSample = stop[0]          
        
        elif ((nullStart == True ) & (nullStop == False )):
           try:
               for i in revStatEnd:
                probableStop = standphases[np.where(statEnd == j)[0][0] - 25 + len(standphases)] 
                signalLength = len(np.arange(start[0],probableStop[0]))
                if ((signalLength > minDiff) & (signalLength < maxDiff)):
                    stop = probableStop
                    beginSample = start
                    raise Error       
           except Error: 
               if ((stop[0] + 100) < len(accMag)-1):
                    endSample = stop[0] + 100
               elif ((stop[0] + 50) < len(accMag)-1):
                    endSample = stop[0] + 50    
               elif ((stop[0] - 10) < len(accMag)-1):
                    endSample = stop[0] + 10
               else:
                    endSample = stop[0]      
                
        elif ((nullStart == False ) & (nullStop == True )):
            try:
                for i in statBegin:
                    probableStart = standphases[np.where(statBegin == i)[0][0]]
                    signalLength = len(np.arange(probableStart[-1],stop[0]))
                    if ((signalLength > minDiff) & (signalLength < maxDiff)):
                        start = probableStart
                        endSample = stop
                        raise ValueError        
            except Error:    
                if ((start[-1] - 100) > 0):
                    beginSample = start[-1] - 100
                elif ((start[-1] - 50) > 0):
                    beginSample = start[-1] - 50       
                elif ((start[-1] - 10) > 0):
                    beginSample = start[-1] - 10       
                else:
                    beginSample = start[-1]     
    
        else:
            beginSample = start
            endSample = stop
    except: 
        pass
        
    try:                
        walkPart = np.arange(beginSample, endSample) 
        startEnd = [start,stop]
    except:
        walkPart = np.arange(0, len(accMag)-1) 
        startEnd = [0,0]
        
    print('Included signal length is: ', len(walkPart))

    return walkPart, startEnd

def resample(obj, peakarray, n_strides, firststride, new_sf, sig):
    ''' 
    Insert object
    insert amount of strides/steps that you want to include
    Insert first stride
    Insert new sample frequency
    '''
    
    laststride = firststride + n_strides 
    begin = int((peakarray[firststride-1] + peakarray[firststride]) / 2)
    end = int((peakarray[laststride-1] + peakarray[laststride]) / 2)
    n_samples = n_strides * new_sf
    resampled = np.zeros((n_samples,3))
    if sig == 'acceleration':
        for i in range(3):
            sig = obj.acceleration[begin:end,i]
            y_new = np.zeros(n_samples)
            f = signal.decimate(sig, 2)
            y_new = signal.resample(f, n_samples)
            resampled[:,i] = y_new
    else:
        for i in range(3):
            sig = obj.gyroscope[begin:end,i]
            y_new = np.zeros(n_samples)
            f = signal.decimate(sig, 2)
            y_new = signal.resample(f, n_samples)
            resampled[:,i] = y_new
            
    return resampled, new_sf    

def autocovariance(obj, sig, plotje = None):
    '''
    Calculates the total autocovariance and the autocovariance max value
    '''
    try:
        lag = int(obj.dominantFreq * 3)
    except AttributeError:
        lag = int((1 / obj.fftsignalVT.stepfreq) / obj.sampleFreq * 3)
        
    if sig == 'acc':
        sig = obj.acceleration
    else:
        sig = obj.globalacceleration
        
    obj.autocovx_acc_ser = smt.acovf(sig[obj.walkSig,0], 
                                     demean = True, nlag = lag, fft = True)
    obj.autocovx_acc = np.max(obj.autocovx_acc_ser[10:])
    
    obj.autocovy_acc_ser = smt.acovf(sig[obj.walkSig,1], 
                                 demean = True,  nlag = lag, fft = True)
    obj.autocovy_acc = np.max(obj.autocovy_acc_ser[10:])

    
    obj.autocovz_acc_ser = smt.acovf(sig[obj.walkSig,2], 
                                 demean = True, nlag = lag, fft = True)
    obj.autocovz_acc = np.max(obj.autocovz_acc_ser[10:])

    obj.autocovx_gyr_ser = smt.acovf(obj.gyroscope[obj.walkSig,0], 
                                 demean = True,  nlag = lag, fft = True)
    obj.autocovy_gyr = np.max(obj.autocovx_gyr_ser[10:])
   
    
    obj.autocovy_gyr_ser = smt.acovf(obj.gyroscope[obj.walkSig,1], 
                                 demean = True,  nlag = lag, fft = True)
    obj.autocovy_gyr = np.max(obj.autocovy_gyr_ser[10:])
   
    
    obj.autocovz_gyr_ser = smt.acovf(obj.gyroscope[obj.walkSig,2], 
                                 demean = True,  nlag = lag, fft = True)
    obj.autocovz_gyr = np.max(obj.autocovz_gyr_ser[10:])
   
    if plotje:
        array = np.transpose(np.array([obj.autocovx_acc_ser, obj.autocovy_acc_ser,
                        obj.autocovz_acc_ser]))
        plot1(array,'Autocovariance acceleration', 'Samples', 'Overlap')
        array = np.transpose(np.array([obj.autocovx_gyr_ser, obj.autocovy_gyr_ser,
                        obj.autocovz_gyr_ser]))
        plot1(array,'Autocovariance Gyroscope', 'Samples', 'Overlap')
        
  

def autocorrelation(obj, sig, plotje = None):
    '''
    Calculates the total autocorrelation and the autocorrelation max value
    '''
    try:
        lag = int(obj.dominantFreq * 3)
    except AttributeError:
        lag = int((1 / obj.fftsignalVT.stepfreq) / obj.sampleFreq * 3)
        
    if sig == 'acc':
        sig = obj.acceleration
    else:
        sig = obj.globalacceleration
            
    obj.autocorx_acc_ser = smt.acf(sig[obj.walkSig,0], 
                                      nlags = lag, fft = True)
    obj.autocorx_acc = np.max(obj.autocovx_acc_ser[10:])
    
    obj.autocory_acc_ser = smt.acf(sig[obj.walkSig,1], 
                                  nlags = lag, fft = True)
    obj.autocory_acc = np.max(obj.autocovy_acc_ser[10:])

    
    obj.autocorz_acc_ser = smt.acf(sig[obj.walkSig,2], 
                                  nlags = lag, fft = True)
    obj.autocorz_acc = np.max(obj.autocovz_acc_ser[10:])

    obj.autocorx_gyr_ser = smt.acf(obj.gyroscope[obj.walkSig,0], 
                                   nlags = lag, fft = True)
    obj.autocory_gyr = np.max(obj.autocovx_gyr_ser[10:])
   
    
    obj.autocory_gyr_ser = smt.acf(obj.gyroscope[obj.walkSig,1], 
                                   nlags = lag, fft = True)
    obj.autocory_gyr = np.max(obj.autocovy_gyr_ser[10:])
   
    
    obj.autocorz_gyr_ser = smt.acf(obj.gyroscope[obj.walkSig,2], 
                                   nlags = lag, fft = True)
    obj.autocorz_gyr = np.max(obj.autocovz_gyr_ser[10:])
   
    if plotje:
        array = np.transpose(np.array([obj.autocorx_acc_ser, obj.autocory_acc_ser,
                        obj.autocorz_acc_ser]))
        plot1(array, 'Autocorrelation acceleration', 'Samples', 'Overlap in percentage')
        array = np.transpose(np.array([obj.autocorx_gyr_ser, obj.autocory_gyr_ser,
                        obj.autocorz_gyr_ser]))
        plot1(array, 'Autocorrelation Gyroscope', 'Samples', 'Overlap in percentage')
        

def normalised(self, leftFoot, rightFoot, lowBack, lengte) :
    bodyHeight = lengte
    meantotdist = np.mean([leftFoot.totdist, rightFoot.totdist])
    self.normDistance = meantotdist/bodyHeight
    self.normCadence = np.mean([leftFoot.cadence, rightFoot.cadence]) / bodyHeight
    self.normDistancePStride = np.mean([leftFoot.meanStrideDistperstep,  
                                  rightFoot.meanStrideDistperstep])
    TBH = np.sqrt(bodyHeight / 9.81)
    self.normTimePerStep = lowBack.stridetimemean / TBH
    