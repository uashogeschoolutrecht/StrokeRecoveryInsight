import numpy as np
from . import commonFunctions

class fastfourierTransform:
    def __init__(self, obj, direction, position = None, 
                 lineartaper = None, plotje = None):
        ''' DIRECTION CORRESPOND TO THE FOLLOWING 
        Foot sensor:      
        forward/backward:       acceleration[:,0]
        left/right:             acceleration[:,1]
        up/down:                acceleration[:,2]
        
        Lower back sensor:
        forward/backward:       acceleration[:,2]
        left/right:             acceleration[:,1]
        up/down:                acceleration[ ,0]
        '''
        if position == 'lowBack':
            if direction == 'VT':
                direction = 0
            elif direction == 'ML':
                direction = 1
            elif direction == 'AP':
                direction = 2
                
        if ((position == 'Foot')):
            if direction == 'AP':
                direction = 0
            elif direction == 'ML':
                direction = 1
            elif direction == 'AP':
                direction = 2
                
        self.direction  = direction
        signal = obj.accFT[:,direction]
        signal = (signal - signal.mean()) / signal.std()
        signalLength = len(signal)                

        # Apply linear taper of %5 at each end
        if lineartaper :
            int(signalLength * 0.05 )
            signal[0:int(signalLength * 0.05 )] = (signal[0:int(signalLength * 0.05 )] / 20)
            signal[signalLength - int(signalLength * 0.05): signalLength ] = signal[signalLength - int(signalLength * 0.05): signalLength ] / 20
            
        # add x amount of zeros so that the signal is 2^x
        check = True
        zeroscounter = 512
        while (check == True):
            if (signalLength > zeroscounter):
                 zeroscounter = 2 * zeroscounter
            else:
                tot = zeroscounter - signalLength
                check = False
        signal = np.concatenate([signal, np.zeros(tot)]) 
        ncor = len(signal)
        
        # Creates the necessairy frequenciees
        fftsignal           = np.fft.fft(signal)
        freqenties           = np.fft.fftfreq(ncor, obj.sampleFreq)
        power               = (abs(fftsignal) / ncor)
        mask                = freqenties > 0
        

        # Mask array to be used for power spectra
        # ignoring half the values, as they are complex conjucates of the other
        self.freqs      = freqenties
        self.power      = power
        self.fftTime   = freqenties[mask]
        self.fftValues = power[mask]
        
        if plotje:
            commonFunctions.plot6(freqenties[mask], 
                                  power[mask], 
                                  title = 'Spectral Density plot',
                                  xlabel = 'Frequency [Hz]',
                                  ylabel = 'Power'
                                  )

    def FFTGait(self, hzMax, hzMin, printje = None):
        
        fftTime = self.fftTime
        fftValues = self.fftValues
            
        
        dominantFreqInphase = np.empty((9,3),float)
        frequencyRange = np.where((fftTime < hzMax) & (fftTime > hzMin))
        maxOfRange = max(fftValues[frequencyRange])
        result = np.where(maxOfRange == fftValues)
        hzOfMaxValue = fftTime[result[0][0]]
        dominantFreqInphase[0,0:3] = [maxOfRange, result[0][0], hzOfMaxValue]
        dominantfreqvalues = np.empty((8,20), int) 
        
        # Inphase

        for i in range(2,10):
            if (hzMax > 1.5):
                var = np.array(range((int(dominantFreqInphase[0][1]) * i - 10),
                                     (int(dominantFreqInphase[0][1]) * i + 10),1))
            else:
                fftsignal = int(dominantFreqInphase[0][1]) * 2
                var = np.array(range(((fftsignal * i) - 
                                      int(dominantFreqInphase[0][1]) - 10),
                    ((fftsignal * i)- int(dominantFreqInphase[0][1]) + 10),1))
            j = i - 2
            dominantfreqvalues[j,0:20] = var 
        
     
        for i in range(0,len(dominantfreqvalues)):    
            tmpMax = max(fftValues[dominantfreqvalues[i,:]])
            tmpMaxHz = np.where(tmpMax == fftValues)
            Hzval = fftTime[tmpMaxHz[0][0]]
            varlist = [tmpMax, tmpMaxHz[0][0], Hzval] 
            j = i + 1
            dominantFreqInphase[j,0:3] = varlist
        
        # Outphase
        
        dominantfreqoutphase = np.empty((9,3),float) 
        dominantfreqvalues2 = np.empty((9,20), int)
        if (hzMax > 1.5):
            dominantfreqvalues2[1:9,:] = dominantfreqvalues - (int(dominantFreqInphase[0][1] / 2))
            dominantfreqvalues2[0,:] = dominantfreqvalues2[1,:] - int(dominantFreqInphase[0][1])
        else: 
            dominantfreqvalues2[1:9,:] = dominantfreqvalues + (int(dominantFreqInphase[0][1]))
            dominantfreqvalues2[0,:] = dominantfreqvalues2[1,:] - (int(dominantFreqInphase[0][1] * 2))
    
            
        for i in range(0,len(dominantfreqvalues2)):    
            tmpMax = max(fftValues[dominantfreqvalues2[i,:]])
            tmpMaxHz = np.where(tmpMax == fftValues)
            Hzval = fftTime[tmpMaxHz[0][0]]
            varlist = [tmpMax, tmpMaxHz[0][0], Hzval] 
            dominantfreqoutphase[i,0:3] = varlist
        
        # width
        half = dominantFreqInphase[0][0] / 2 
        i = True
        num = int(dominantFreqInphase[0][1])
        while (i == True):
            num = num + 1
            value = fftValues[num]
            if (value <= half):
                first = num
                i = False
        
        i = True
        num = int(dominantFreqInphase[0][1])
        while (i == True):
            num = num - 1 
            value = fftValues[num]
            if (value <= half):
                second = num
                i = False        
        
        width = fftTime[first] - fftTime[second]   
        
        # Slope
        slope1 = (dominantFreqInphase[0][0] - fftValues[first])   / (first - (int(dominantFreqInphase[0][1])))
        slope2 = (dominantFreqInphase[0][0] - fftValues[second])  / ((int(dominantFreqInphase[0][1])) - second)
        slope = np.mean([slope1, slope2])
        
        # Density dominant peak, the peak +/- 2 samples
        SDPrange = range((int(dominantFreqInphase[0][1])) -2, (int(dominantFreqInphase[0][1])) + 3)
        SDP = sum(fftValues[SDPrange])
        
        self.inphase             = dominantFreqInphase                                              
        self.outphase            = dominantfreqoutphase                                                
        self.width               = width                                    # Width of dominant peak
        self.slope               = slope                                   # Slope of dominant peak 
        self.SDP                 = SDP                                   # Density of dominant peak
        self.harmonicratio       = sum(dominantFreqInphase [:,0]) / sum(dominantfreqoutphase [:,0]) # harmonicratio
        self.harmonicindex       = dominantFreqInphase [0,0] / sum(dominantFreqInphase [0:5,0]) # harmonic index
        self.dominantpeak        = self.inphase [0,0]
        self.hzMax               = dominantFreqInphase [0][2] * 2 + 1                          # Determine new HZ for stepfreq for AP and VT
        if self.direction == 1:
            self.stridefreq          = dominantFreqInphase[0][2]
            self.hzMin                 = self.stridefreq + 0.5 * self.stridefreq 
        else :
            self.stepfreq = dominantFreqInphase[0][2]
        
        if printje:
            if self.direction == 0:
                print("Mediolateral")
            elif self.direction == 1:
                print("Mediolateral")
            else :
                print("Mediolateral")
                
            if self.direction == 1:
                print("stridefreq: ", self.inphase[0][2] )
            else:
                print("stepfreq: ", self.inphase[0][2] )

            print("harmonicratio: ", self.harmonicratio)                                         
            print("harmonic index: ", self.harmonicindex )                                        
            print("Width of the dominant peak (50%)", self.width )
            print("Slope of dominant peak (50%)", self.slope )
            print("Density of dominant peak", self.SDP )
            print("Amplitude of dominant peak", self.dominantpeak )
            print('\n')


    def fft_balance(obj):
        '''
        Input: FFT variables 
        Output: Tot frequency, median frequency, 95% frequency, 
        dominant frequency, frequency dispersion
        '''
        fftTime    = obj.fftTime
        fftValues  = obj.fftValues
        fft_tot     =  np.sum(fftValues)
        obj.fft_tot = fft_tot
        
        F50_Sum     = 0
        for i in fftValues:
            F50_Sum += i
            if ((F50_Sum / fft_tot) >= 0.5):
                F50_idx = np.where(fftValues == i)[0]
                F50 = fftTime[F50_idx]
                break
        obj.F50_Sum = F50
        
        F95_Sum     = 0
        for i in fftValues:
            F95_Sum += i
            if ((F95_Sum / fft_tot) >= 0.95):
                F95_idx = np.where(fftValues == i)[0]
                F95 = fftTime[F95_idx]
                break  
         
        obj.F95_Sum = F95  
        
        dominantfreq = np.max(obj.fftValues)
        obj.dominantfreq = dominantfreq
        dominantfreqS = obj.fftTime[np.where(obj.fftValues == dominantfreq)[0]]
        mask = ((obj.freqs > dominantfreqS - 0.1) & (obj.freqs < dominantfreqS + 0.1))
        
        obj.freqdispersion = np.sum(obj.power[mask]) / fft_tot
        
