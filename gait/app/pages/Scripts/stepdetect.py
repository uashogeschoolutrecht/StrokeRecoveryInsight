
'''
This function is used to detect steps in a footsensor. We first assess the 
main frequency of the sensor in the ML direction. We assume that the signal 
contains some regularity and the stride frequency to be between 0.2 and 1 HZ.
With a stride length of 1.4m this is equal to 1 km/h > and < 5 km/h. In stroke
patients the stride length is ofter smaller (0.7m) meaning that we can include 
patients that walk only 0.5 km/h. This allows us to calculate the minimum and
maximum time between steps.
       
This function can be used to calculate the peaks in the VT direction
corresponding to the Foottouch. 
We first make the assumption of regularity that we can find the next 
step at (0.75* 'average step length' -> ∞ .
After this function we do a false negative peak detection. We calculate
the distance between the found peaks. In case this distance is too 
large another peak detection will be done with lower threshold values.
Then we do a false positive peak detection based on the fact that 
between two peaks should be a standphase. If not such a standphase can 
be detected one of the peaks is incorrect. The most dominant peak will 
be selected. 
Based on this method we can calculate:
Peaks in VT and standphases. 
mean, std, rms, normalised time per step     
'''

import numpy as np
import matplotlib.pyplot as plt
from . import commonFunctions
from scipy import signal   
from . frequencyFunctions import fastfourierTransform
from scipy import integrate
from scipy.stats import pearsonr

    
class Stepdetect:
    def stepFrequency(footObject, maxStrideFreq, plotje = False):               # Create object with the relevant components
        '''
        We estimate the main frequency of the sensor in the ML direction. 
        We assume that the signal contains some regularity and the stride 
        frequency to be between 0.2 and 1 HZ.
        '''
        footObject.accFT              = np.array(footObject.acceleration)      # The fourier transform                               
        footObject.fftFoot            = fastfourierTransform(footObject,
                                                              direction = 'ML', 
                                                              position = 'Foot',
                                                              plotje = plotje)

        footObject.fftFoot.FFTGait(hzMax = maxStrideFreq, hzMin = 0.15)
        
        
        footObject.dominantFreq       = ((1 / footObject.fftFoot.stridefreq)
                                            / footObject.sampleFreq)            # Average time between steps
        footObject.accMag             = commonFunctions.magnitude(footObject.acceleration)
        footObject.gyrMag             = commonFunctions.magnitude(footObject.gyroscope)
        footObject.threshAcc          = np.mean(footObject.accMag) + np.std(footObject.accMag) 
        footObject.threshGyr          = np.mean(footObject.gyrMag) 
        
        if footObject.dominantFreq    < 100:    # Fast        
            multiplyr       = 0.25
        elif footObject.dominantFreq  < 150:    # Normal
              multiplyr     = 0.3
        else :    # Slow
              multiplyr     = 0.4
        footObject.threshT            = np.int(np.floor(footObject.dominantFreq * multiplyr)) # Minimu time between steps
        if    footObject.threshT  > 50:
            footObject.threshT  = 50
            
    
    def stepdetection_foot(footObject,  devideInparts, plotje = None, 
                           printje = None):
        '''
        This function can be used to calculate the peaks in the VT direction
        corresponding to the Foottouch. 
        We first make the assumption of regularity that we can find the next 
        step at (0.75* 'average step length' -> ∞ .
        After this function we do a false negative peak detection. We calculate
        the distance between the found peaks. In case this distance is too 
        large another peak detection will be done with lower threshold values.
        Then we do a false positive peak detection based on the fact that 
        between two peaks should be a standphase. If not such a standphase can 
        be detected one of the peaks is incorrect. The most dominant peak will 
        be selected. 
        Based on this method we can calculate:
        Peaks in VT and standphases. 
        mean, std, rms, normalised time per step
        '''
        relPeriod           = footObject.walkSig
        acceleration        = footObject.acceleration
        threshAcc           = footObject.threshAcc
        threshGyr           = footObject.threshGyr
        threshT             = footObject.threshT 
        dominantFreq        = footObject.dominantFreq
        accMagnitude        = footObject.accMag
        gyrMagnitude        = footObject.gyrMag
        min_distance        = int(dominantFreq * 0.75)                          # The minimum distance between steps
        accVT               = acceleration[:,2]
        '''
        accVT               = np.abs(acceleration[:,2])
        accVTINT            = integrate.cumtrapz(accVT, 
                                                 accVT, 
                                                 footObject.sampleFreq, 
                                                 axis = 0)
        accVT = accVTINT
        '''

        min_height          =  np.mean(accVT[relPeriod]) + np.std(accVT[relPeriod])   # The minimum height of a peak to be a peak
        peaksVT             = signal.find_peaks(accVT[relPeriod], 
                                     distance = min_distance,                   # Each peak in VT == heel contact
                                     height = min_height)
        peaks               = peaksVT[0] + relPeriod[0]   
        difPeaks            = np.diff(peaks) 
        difPeaks            = np.concatenate((difPeaks, np.array([0])))
        tmpPeaks            =  []                                               # By default between these peaks there is a standphase
        

        # Negative peak detection 
        #breakpoint()
        for i in range(0,len(peaks)):
            tmpPeaks.append(peaks[i])
            if difPeaks[i] > (dominantFreq*1.5):
                try:
                   
                    tmpPeak = signal.find_peaks(accVT[peaks[i]+threshT
                                                     :peaks[i+1]-threshT], 
                                     height = np.mean(accVT[relPeriod]) + np.std(accVT[relPeriod])*0.75)[0][0] + peaks[i]+threshT
                    if (( tmpPeak - tmpPeaks[-1] ) < threshT*0.75) or ((peaks[i+1] - tmpPeak) < threshT*0.75):
                        pass
                    else:
                        #print('peak appended')
                        tmpPeaks.append(tmpPeak)
                except: 
                    pass
                    
                    
        peaks = np.array(tmpPeaks)   
        try:
            # False positive peak detection
            standphases = []
            swingphases = []
            standphases.append(np.arange(0,relPeriod[0]))
            for i in range(0,len(peaks)-1):
                stationaryCounter           = []
                counter         = 0
                counter2        = 0
                check           = False
                tmpThreshAcc  = threshAcc
                tmpThreshGyr  = threshGyr
                tmpThreshTime    = threshT
                while (check == False):
                    for j in range(peaks[i],peaks[i+1]):                                       #                Standphase is defined as >20 samples with a 
                        if ((accMagnitude[j] < tmpThreshAcc) and (gyrMagnitude[j] < tmpThreshGyr)):       # acceleration maginitude and gyroscope magnitude low than
                            stationaryCounter.append(j)                                                 # 1.8 and 1.7 respectively
                            counter  += 1        
                            if ((j == (peaks[i+1]-1)) & (counter > threshT)): 
                                standphases.append(np.array(stationaryCounter[:-1]))
                                check = True
                                break
                        else: 
                            if (counter > threshT):
                                standphases.append(np.array(stationaryCounter))
                                check = True
                                break
                            else:
                                counter = 0
                                stationaryCounter   = []
                    if (check == False):
                        tmpThreshAcc  *= 1.2 
                        tmpThreshGyr  *= 1.2
                        tmpThreshTime    *= 0.5
                        counter2 += 1 
                    if (counter2 == 4):
                        #print('We cannot find a standfase between the peak at: ', 
                        #     peaks[i], 'and', peaks[i+1])
                        arr         = np.array([peaks[i], peaks[i+1]])
                        lowestPeak  = np.where(accVT[arr] == np.min(accVT[arr]))[0]
                        lowestPeak  = arr[lowestPeak]
                        delPeak     = np.where(peaks == lowestPeak)[0]
                        peaks = np.delete(peaks, delPeak)
                        check == True
                        break
        except IndexError:
            pass
            
        standphases.append(np.arange(relPeriod[-1],len(accVT) ))
        
        for i in range(0,len(standphases)-1):
            swingphases.append(np.array(range(standphases[i][-1], standphases[i+1][0])))

        if plotje:                                                              # Plot the magnitude in case something is wrong
            fig1, ax2 = plt.subplots()
            ax2.plot(np.array(range(0,len(accVT))), accVT)
            ax2.plot(peaks, accVT[peaks], 'o')            
        

        # Devide in parts: 
        stridenum = len(swingphases)
        footObject.stridenum           = stridenum 
        footObject.standphases       = standphases
        footObject.swingphases       = swingphases 
        footObject.peaksVT          = peaks
        

        count = int(np.floor(len(peaks[5:-5]) / devideInparts))
        timepersteride = []
        stridetimemean = []
        stridetimeSTD = []
        normstridetime = []
        for i in range(count):
            tmpVal          = np.diff(peaks[devideInparts*i+5:devideInparts*i+6+devideInparts])*footObject.sampleFreq
            timepersteride.append(tmpVal)
            stridetimemean.append(np.mean(tmpVal))
            stridetimeSTD.append(np.std(tmpVal))
            normstridetime.append( np.std(tmpVal) / tmpVal)
            
        footObject.timepersteride      = timepersteride
        footObject.stridetimemean      = np.mean(stridetimemean)
        footObject.stridetimeSTD       = np.mean(stridetimeSTD)
        footObject.normstridetime      = np.mean(normstridetime)  
        footObject.cadence              = int(len(peaks)/2)
        if printje:
            print("Amount of stride in trial: ", footObject.stridenum)
            print("Mean time per stride: ", footObject.stridetimemean, 's')
            print("Step time STD: ", footObject.stridetimeSTD)
            print("Normalised step time: ",  footObject.normstridetime )
            print("\n")
    

    def stepdetection_lowback_HC(lowBack, leftFoot, rightFoot, devideInparts,
                                 plotje = None, printje = None):
        from scipy import signal 
        '''
        This function can be used to determine the gait events (initial contact) 
        during normal walking. 
        
        First the signal is filtered with a lowpass filter of 2Hz. From this 
        signal the IC can be determined by a minimum in the vertical direction.
        The start is defined as the place where acceleration magnitude is 
        > 1.3. This peak will not be in the calculations.
        
        The method is based on the article of:
            Bugané, F., Benedetti, M. G., Casadio, G., Attala, S., Biagi, F., Manca, 
        M., & Leardini, A. (2012). Estimation of spatial-temporal gait parameters 
        in level walking based on a single accelerometer: Validation on normal 
        subjects by standard gait analysis. Computer Methods and Programs in 
        Biomedicine, 108(1), 129–137. doi:10.1016/j.cmpb.2012.02.003 
                
        Pham MH, Elshehabi M, Haertner L, Del Din S, Srulijes K, Heger T, Synofzik M, Hobert MA, Faber GS, Hansen C, Salkovic D, Ferreira JJ, Berg D, Sanchez-Ferro Á, van Dieën JH, Becker C,
        Rochester L, Schmidt G and Maetzler W (2017) Validation of a Step Detection Algorithm during Straight Walking and Turning in Patients with Parkinson’s Disease and Older Adults Using an Inertial Measurement
        Unit at the Lower Back. Front. Neurol. 8:457. doi: 10.3389/fneur.2017.00457
        '''
        def detectstride(footobj, ORAP, step, deviation):
            peaks = np.array(footobj.peaksVT )
            peaks -= peaks[0] -  step
            peaksstridelowback = []
            
            # First stride 
            peaksstridelowback.append(peaks[0])
            sampleNumb = peaks[1]
            diffpeaks = np.diff(peaks)
            
            # All but first and last stride
            try:
                for j in range(len(peaks)-2):
                    searchWindow = np.array(range(sampleNumb-deviation,
                                                  sampleNumb+deviation))
                    sampleNumb = (np.where(np.max(ORAP[searchWindow]) == 
                                          ORAP[searchWindow])[0][0] + 
                                          (sampleNumb - deviation) )
                    peaksstridelowback.append(sampleNumb)
                    sampleNumb += diffpeaks[j+1]
            except IndexError:
                print('search window out of bound.')
                   
            # Last stride 
            try:
                searchWindow    = np.array(range(sampleNumb-deviation,
                                              sampleNumb+deviation))
                sampleNumb      = (np.where(np.max(ORAP[searchWindow]) == 
                                            ORAP[searchWindow])[0][0] + 
                                          (sampleNumb - deviation) )
                peaksstridelowback.append(sampleNumb)
                peaksstridelowback = np.array(peaksstridelowback)
            except IndexError:
                print('search window out of bound, last step not found')
           
            return peaksstridelowback
        
        def detectSecondFoot(peaks, foundLowBackPeaks, MinPeakDifference,
                                 dominantFreq):
                peaksstridelowback = []
                # Finds the seconds step  
                try:
                    for i in range(0,len(peaks)-1):  
                        start = int(foundLowBackPeaks[i] + MinPeakDifference)
                        stop = int(foundLowBackPeaks[i+1] - MinPeakDifference)
                        peaksstridelowback.append(np.where(np.max(ORAP[start:stop])  == 
                                                           ORAP[start:stop])[0][0] + start)
                except ValueError:
                    pass
                except IndexError: 
                    pass 
                try:
                    if len(peaksstridelowback) < len(peaks):
                        difference = len(peaks) - len(peaksstridelowback) 
                        for i in range(0,difference):
                            start += int(dominantFreq)
                            stop += int(dominantFreq) 
                            peaksstridelowback.append(np.where(np.max(ORAP[start:stop])  == 
                                           ORAP[start:stop])[0][0] + start)
                except ValueError: 
                    pass 
                return peaksstridelowback
            
        def correlationFootLowback(Foot, LowBack, counter, peaks):
            corr = []
            for i in range(0,counter-1):
                defsig = np.arange(i,i+len(Foot))
                corr.append(pearsonr(Foot,oR[defsig])[0])
            maxCorr = np.max(corr)
            mx = np.where(maxCorr == corr)[0]   
            firstPeak = mx + peaks[0]
            return firstPeak + 100, maxCorr 
        
        peaksLeft   = np.array(leftFoot.peaksVT) 
        peaksLeft   -= leftFoot.walkSig[0]
        lF          = (leftFoot.acceleration[leftFoot.walkSig,2] - 
                       np.mean(leftFoot.acceleration[leftFoot.walkSig,2]))
        peaksRight  = np.array(rightFoot.peaksVT) 
        peaksRight  -= rightFoot.walkSig[0]
        rF          = (rightFoot.acceleration[rightFoot.walkSig,2] - 
                       np.mean(rightFoot.acceleration[rightFoot.walkSig,2]))
        oR          = (lowBack.acceleration[lowBack.walkSig,0] - 
                       np.mean(lowBack.acceleration[lowBack.walkSig,0]))
        oR = np.concatenate((np.zeros(100),oR,np.zeros(2000)))
        counterL =  len(oR) - len(lF)
        if counterL > 400:
            counterL = 400
        counterR =  len(oR) - len(rF)
        if counterR > 400:
            counterR = 400        
        # Checks where the correlation between lowback and Foot is 
        # highest and gives an estimation of where the first peak should be.
        # commonFunctions.plot4(oR)
      
            
        firstPeakLeft,  corrL  = correlationFootLowback(lF, oR, counterL, 
                                                        peaksLeft)
        firstPeakRight, corrR  = correlationFootLowback(rF, oR, counterR, 
                                                        peaksRight)

        filtAbove = lowBack.fftsignalVT.stepfreq / 4
        # We find peaks using the int, filt, int, filt of lowBack AP
        accelerationORAP    = lowBack.acceleration[lowBack.walkSig,2]
        ORAPInt             = integrate.cumtrapz(accelerationORAP, accelerationORAP, 
                                                 lowBack.sampleFreq, axis = 0)

        ORAPfilt            = commonFunctions.filt_band(ORAPInt, 
                                                lowBack.sampleFreq, 
                                                filtAbove, 
                                                15, 
                                                2, 
                                                1)

        ORAPfiltInt         = integrate.cumtrapz(ORAPfilt, ORAPfilt, 
                                                 lowBack.sampleFreq, axis = 0)
                                                 
        ORAPfiltfilt        = commonFunctions.filt_band(ORAPfiltInt, 
                                                        lowBack.sampleFreq, 
                                                        filtAbove, 
                                                        15, 
                                                        2, 
                                                        1)
        ORAP = ORAPfiltfilt


        
        # commonFunctions.plot4(ORAP)
        # If the correlation between signals > 0.2 this is used to find first 
        # peak. Else we use the peak find method
        if (corrL > 0.3) | (corrR > 0.3):
            if corrL > corrR:  
                LowBackRange = np.arange(firstPeakLeft - 50, 
                                         firstPeakLeft + 50)
            else:
                LowBackRange = np.arange(firstPeakRight - 50, 
                                         firstPeakRight + 50)  
            firstpeak = np.where(np.max(ORAP[LowBackRange]) == 
                                 ORAP[LowBackRange])[0] + LowBackRange[0]
        
        else:
            min_height =  np.mean(ORAP[:500]) + (np.std(ORAP[:500]))
            samplesperstep = int(leftFoot.dominantFreq / 2)
            min_distance = samplesperstep * 0.5
            peaks = signal.find_peaks(ORAP[:500], height = min_height,                 
                                      distance = min_distance)
            firstpeak = peaks[0][0]
        
        # We evaluate if the first step is left/right based on ML
        ORML = commonFunctions.filt_low(lowBack.acceleration[lowBack.walkSig,1],
                                        0.01, 
                                        cutoff = 1, 
                                        order = 1)
        firststepdir = np.mean(ORML[np.arange(firstpeak-10, firstpeak+10)])
        # commonFunctions.plot4(ORML)
        if firststepdir < np.mean(ORML):
            direction = 'Right'
        else:
            direction = 'Left'
        
        lowBack.direction = direction
        deviation = 15
        samplesPerStep = int(leftFoot.dominantFreq / 2)
        MinPeakDifference = int(0.4 * samplesPerStep) 

        ORAP = np.concatenate((ORAP,np.zeros(1000)))

        for i in range(4):
            deviation += 5
            if direction == 'Right':
                peaksstridelowbackRight = detectstride(rightFoot, 
                                                       ORAP, 
                                                       firstpeak, 
                                                       deviation = deviation)         
                peaksstridelowbackLeft = detectSecondFoot(peaks = peaksLeft, 
                                                          foundLowBackPeaks = peaksstridelowbackRight, 
                                                          MinPeakDifference = MinPeakDifference, 
                                                          dominantFreq = leftFoot.dominantFreq)        
            else:
                peaksstridelowbackLeft = detectstride(leftFoot, 
                                                       ORAP, 
                                                       firstpeak, 
                                                       deviation = deviation)
                peaksstridelowbackRight = detectSecondFoot(peaks = peaksRight, 
                                                          foundLowBackPeaks = peaksstridelowbackLeft, 
                                                          MinPeakDifference = MinPeakDifference, 
                                                          dominantFreq = rightFoot.dominantFreq)
                    
            peakslowback = np.concatenate((peaksstridelowbackRight, peaksstridelowbackLeft))
            peakslowback = np.delete(peakslowback, np.where(ORAP[peakslowback] == 0)[0])
            peakslowback = np.array(peakslowback[np.argsort(peakslowback)])

            if (np.abs(len(peakslowback) - (leftFoot.stridenum + rightFoot.stridenum)) < 10):
                break
            else:
                if direction == 'Right':
                    peaksstridelowbackLeft = detectstride(leftFoot, 
                                                           ORAP, 
                                                           firstpeak, 
                                                           deviation = deviation)
                    peaksstridelowbackRight = detectSecondFoot(peaks = peaksRight, 
                                                              foundLowBackPeaks = peaksstridelowbackLeft, 
                                                              MinPeakDifference = MinPeakDifference, 
                                                              dominantFreq = rightFoot.dominantFreq)      
                else:
                    peaksstridelowbackRight = detectstride(rightFoot, 
                                       ORAP, 
                                       firstpeak, 
                                       deviation = deviation)         
                    peaksstridelowbackLeft = detectSecondFoot(peaks = peaksLeft, 
                                                              foundLowBackPeaks = peaksstridelowbackRight, 
                                                              MinPeakDifference = MinPeakDifference, 
                                                              dominantFreq = leftFoot.dominantFreq)  
                peakslowback = np.concatenate((peaksstridelowbackRight, peaksstridelowbackLeft))
                peakslowback = np.delete(peakslowback, np.where(ORAP[peakslowback] == 0)[0])
                peakslowback = np.array(peakslowback[np.argsort(peakslowback)])
    
                if (np.abs(len(peakslowback) - (leftFoot.stridenum + rightFoot.stridenum)) < 10):
                    break

        if plotje:
            fig1, ax2 = plt.subplots()
            ax2.plot(np.array(range(0,len(ORAP))), ORAP) 
            ax2.plot(peakslowback, ORAP[peakslowback],'o') 
            
        lowBack.stepnum     = len(peakslowback)
    
        print("Amount of stride found in trial: ", lowBack.stepnum )
        count = int(np.floor(len(peakslowback[10:-10]) / devideInparts))
        timepersteride = []
        stridetimemean = []
        stridetimeSTD = []
        normstridetime = []
        
        for i in range(count):
            tmpVal          = np.diff(peakslowback[devideInparts*i+10:devideInparts*i+11+devideInparts])*lowBack.sampleFreq
            timepersteride.append(tmpVal)
            stridetimemean.append(np.mean(tmpVal))
            stridetimeSTD.append(np.std(tmpVal))
            normstridetime.append( np.std(tmpVal) / tmpVal)
              
        lowBack.peaks               = peakslowback
        lowBack.timepersteride      = timepersteride  
        lowBack.stridetimemean      = np.mean(stridetimemean)
        lowBack.stridetimeSTD       = np.mean(stridetimeSTD)
        lowBack.normstridetime      = np.mean(normstridetime  )      
       
        if printje:
            print("Mean time per stride: ", lowBack.stridetimemean, 's')
            print("Step time STD: ", lowBack.stridetimeSTD)
            print("Normalised step time: ",  lowBack.normstridetime )
            print("\n")

      
    