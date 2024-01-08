import streamlit as st
import numpy as np
import pandas as pd

from gait.app.pages.Scripts.loadFiles import loadCsv
from gait.app.pages.Scripts import commonFunctions
from gait.app.pages.Scripts import advancedFunctions
from gait.app.pages.Scripts.stepdetect import Stepdetect
from gait.app.pages.Scripts.sensorfusion import *
from gait.app.pages.Scripts.frequencyFunctions import * 
from gait.app.pages.Scripts.quaternionfunction import * 
from gait.app.pages.Scripts.lyapunov import *
from gait.app.pages.Scripts.asymmetry import *
from gait.app.pages.Scripts.entropy import *







# Main code
def main():
    strides =  10                               
    steps = 20                               
    MinExpSigLen = 11000 

    st.title("Processing your own collected data")

    # Introduction
    st.markdown("""
    On this page, we enable you to upload your own collected gait data, walking test instructions:
    """)
    
    # Walking Test Instructions
    st.markdown("""
    The walking test consists of the following part:
    
    - Walk continuously for 2 minutes (with or without an aid) on a 14-meter-long course with two cones at each end.
    
    A measurement cannot be included if the participant loses their balance and has to correct to avoid falling, or if the participant makes a non-walking-related movement (e.g., due to distraction). If a measurement fails, the patient may attempt again. Prior to the test, instructions are given for the task. The therapist may give an indication of the elapsed time, provided it does not distract the participant.
    
    Three motion sensors are used. Two sensors are placed midway on the left foot and right foot. Ensure that the sensor cannot slide off the foot during the measurement. The third sensor is placed on the lower back at the level of L5. The time is measured with a stopwatch. Before the motion sensor and the stopwatch are started, the participant first stands in the starting position. Once a stable position is reached, the sensors are activated before the stopwatch starts. When two minutes have elapsed, the therapist signals the participant to stop walking and remain still until the sensors are deactivated.
    """)

    # Displaying the image
    st.image('gait/app/walk.png', caption='Example of data collection')
    parFoot = st.radio("Select the Paretic Foot", ('Left', 'Right'))
    lengte = st.number_input('Enter the length of the participant in cm', min_value=0, max_value=300, value=180, step=1)
    # File upload names
    # Define the upload names
    upload_names = ["Low Back Sensor", "Right Ankle", "Left Ankle"]
    uploaded_files = {name: None for name in upload_names}

    # Create file uploaders with specific names
    all_files_uploaded = True  # Flag to check if all files are uploaded
    for name in upload_names:
        uploaded_files[name] = st.file_uploader(f"Upload {name}", type=["csv"])
        if uploaded_files[name] is None:
            all_files_uploaded = False  # If any file is missing, set the flag to False
    
    

    # Only process files if all files have been uploaded
    if all_files_uploaded:
        # Based on the selection, assign the corresponding DataFrames
        
        # Read each uploaded file into a DataFrame with specific column names and skipping rows
        if uploaded_files["Left Ankle"]:
            data = pd.read_csv(uploaded_files["Left Ankle"], names=['T', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'Time'], sep=',', skiprows=10)
            leftFoot        = loadCsv(data=data, 
                            resample = False,
                            )
        if uploaded_files["Right Ankle"]:
            data = pd.read_csv(uploaded_files["Right Ankle"], names=['T', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'Time'], sep=',', skiprows=10)
            rightFoot        = loadCsv(data=data, 
                            resample = False,
                            )
        if uploaded_files["Low Back Sensor"]:
            data = pd.read_csv(uploaded_files["Low Back Sensor"], names=['T', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'Time'], sep=',', skiprows=10)
            lowBack        = loadCsv(data=data,  
                            resample = False,
                            )
        lowBack.gyroscope -= 0, 0, 0
        leftFoot.gyroscope -= 0, 0, 0
        rightFoot.gyroscope -= 0, 0, 0
   

        leftFoot.walkSig, leftFoot.startEnd     = commonFunctions.startEnd(leftFoot, 
																		'foot', 
																		minDiff = MinExpSigLen,
																		maxDiff = 13000)
	
        rightFoot.walkSig, rightFoot.startEnd   = commonFunctions.startEnd(rightFoot, 
                                                                            'foot', 
                                                                            minDiff = MinExpSigLen,
                                                                            maxDiff = 13000)
        
        lowBack.walkSig, lowBack.startEnd     = commonFunctions.startEnd(lowBack, 
                                                                            'lowback', 
                                                                            minDiff = MinExpSigLen,
                                                                            maxDiff = 13500)
        
        try:    
            if ((len(lowBack.startEnd[0]) > 100) & (len(lowBack.startEnd[1]) > 100)):
                lowBack.acceleration = advancedFunctions.localToGlobal(lowBack, 
                                                                        stationary = True,
                                                                        plotje = False)
        except TypeError:
            pass
       
        try:
            if ((leftFoot is not None and (len(leftFoot.walkSig) < 11000 or len(leftFoot.walkSig) > 13000)) or
                (rightFoot is not None and (len(rightFoot.walkSig) < 11000 or len(rightFoot.walkSig) > 13000)) or
                (lowBack is not None and (len(lowBack.walkSig) < 11000 or len(lowBack.walkSig) > 13500))):
                st.error('Warning: the signal length is too short! Cannot be included.')
            else:
                st.success('Checkpoint 1: signal length is good.')
        except AttributeError as e:
            st.error(f"An error occurred while checking signal lengths: {e}")

        '''
        This is the second check. The difference between strides detected
        should not exceed 3. If it does exceed 3 we try the other foot
        first. Else the participant cannot be included.
        '''
        def stepdetectRightLeft(plotje, printje, maxStrideFreq):
            Stepdetect.stepFrequency(rightFoot, 
                                        maxStrideFreq = maxStrideFreq, 
                                        plotje = plotje)
            if printje:
                print('Right foot: ' , '\n')
            
            Stepdetect.stepdetection_foot(rightFoot,
                                            devideInparts = strides,
                                            plotje = plotje, 
                                            printje = printje) 
            
            Stepdetect.stepFrequency(leftFoot, 
                                        rightFoot.fftFoot.stridefreq, 
                                        plotje = plotje)
            if printje:
                print('Left foot: ' , '\n')
            Stepdetect.stepdetection_foot(leftFoot,  
                                        devideInparts = strides,
                                        plotje = plotje , 
                                        printje = printje) 
            differenceRL =  np.abs(leftFoot.stridenum -rightFoot.stridenum)
            return differenceRL

        def stepdetectLeftRight(plotje, printje, maxStrideFreq):
            Stepdetect.stepFrequency(leftFoot, 
                                    maxStrideFreq = maxStrideFreq, 
                                    plotje = plotje)
            if printje:
                print('Left foot: ' , '\n')
            
            Stepdetect.stepdetection_foot(leftFoot,
                                            devideInparts = strides,
                                            plotje = plotje, 
                                            printje = printje )
            
            Stepdetect.stepFrequency(rightFoot, 
                                        leftFoot.fftFoot.stridefreq, 
                                        plotje = plotje)
            if printje:
                print('Right foot: ' , '\n')

            Stepdetect.stepdetection_foot(rightFoot,  
                                            devideInparts = strides,
                                            plotje = plotje, 
                                            printje = printje) 
            
            differenceLR = np.abs(leftFoot.stridenum -rightFoot.stridenum)
            return differenceLR

        differenceRL = stepdetectRightLeft(plotje = False, printje = False,maxStrideFreq = 1)

        differenceLR = stepdetectLeftRight(plotje = False, printje = False, maxStrideFreq = 1)


        try:
            if ((differenceRL <= differenceLR) & (differenceRL < 8)):
                print('Right foot first')
                stepdetectRightLeft(plotje = False,
                printje = True,
                maxStrideFreq = 1)
                if ((leftFoot.stridenum < 12) or (rightFoot.stridenum < 12) or (np.isnan(leftFoot.stridetimemean)) or (np.isnan(rightFoot.stridetimemean))):
                    print('(leftFoot.stridenum < 12) or (rightFoot.stridenum < 12)')
                    raise Exception

            elif ((differenceLR <= differenceRL) & (differenceLR < 8)):
                print('Left foot first')
                stepdetectLeftRight(plotje = False,
                printje = True,
                maxStrideFreq = 1)
                if ((leftFoot.stridenum < 12) or (rightFoot.stridenum < 12) or (np.isnan(leftFoot.stridetimemean)) or (np.isnan(rightFoot.stridetimemean))):
                    raise Exception
            else:
                raise Exception('problem with selecting front foot')
        except: # This when people walk faster than 1 stride p.s.,
            differenceRL = stepdetectRightLeft(plotje = False,
            printje = True,
            maxStrideFreq = 2)
            differenceLR = stepdetectLeftRight(plotje = False,
            printje = False,
            maxStrideFreq = 2)
            if ((differenceRL <= differenceLR) & (differenceRL < 8)):
                print('Right foot first')
                stepdetectRightLeft(plotje = False,
                printje = True,
                maxStrideFreq = 2)
                if ((leftFoot.stridenum < 12) or (rightFoot.stridenum < 12) or (np.isnan(leftFoot.stridetimemean)) or (np.isnan(rightFoot.stridetimemean))):
                        raise Exception
            elif ((differenceLR <= differenceRL) & (differenceLR < 8)):
                print('Left foot first')
                stepdetectLeftRight(plotje = False,
                printje = True,
                maxStrideFreq = 2)
                if ((leftFoot.stridenum < 12) or (rightFoot.stridenum < 12) or (np.isnan(leftFoot.stridetimemean)) or (np.isnan(rightFoot.stridetimemean))):
                    raise Exception
            else:
                print('We cannot detect the same amount of strides left ',
                'and right. Cannot be included!')
                raise Exception("We cannot detect the same amount of strides left and right. Cannot be included!")
        
        rightFoot.fusion                                    = sensorFusion(rightFoot)
        rightFoot.globalacceleration, rightFoot.quaternion   = sensorFusion.LinearAcceleration(rightFoot, 
                                                                                                plotje = False)
        rightFoot.velocity, rightFoot.position              = sensorFusion.VelocityPosition(rightFoot, 
                                                                                            rotate = False,
                                                                                            veldriftcorrection = True)
        
        sensorFusion.spatTempoutcome(rightFoot,  
                                        devideInparts = strides,
                                        printje = True, 
                                        plotje = False)
        
        
        print('\n', 'SensorFusion Left foot: ' , '\n')
        
        leftFoot.fusion                                     = sensorFusion(leftFoot)
        leftFoot.globalacceleration,rightFoot.quaternion    = sensorFusion.LinearAcceleration(leftFoot, 
                                                                                                plotje = False)
        leftFoot.velocity, leftFoot.position                = sensorFusion.VelocityPosition(leftFoot, 
                                                                                            rotate = False,
                                                                                            veldriftcorrection = True)
        
        sensorFusion.spatTempoutcome(leftFoot, 
                                        devideInparts = strides,
                                        printje = True, 
                                        plotje = False)

        '''
        Distance between feet should not exceed 10m. 
        '''
        if (np.abs(leftFoot.totdist -rightFoot.totdist) > 25):
            print('Difference between distance too large. We cannot include',
                    'this participant')
            raise Exception("Difference between distance too large. We cannot include this participant")
        
        print('\n', 'checkpoint 3 passed','\n')
        
        
        '''
        Returns: Dominant frequenty, width, slope, amplitude, density, stride and 
        step frequency, Harmonic ratio and harmonic index 
        
        IMPORTANT:
        - Calculate the mediolateral first to determine the correct HZ settings for AP and VT 
        '''
        
        # FFT ML
        print('Low Back: ' , '\n')
        lowBack.accFT                  = np.array(lowBack.acceleration[lowBack.walkSig,:])               
        lowBack.fftsignalML             = fastfourierTransform(lowBack, 
                                                                direction = 'ML', 
                                                                position = 'lowBack',
                                                                lineartaper = False,
                                                                plotje = False) 
                                                
        lowBack.fftsignalML.FFTGait(hzMax = 1.5, hzMin = 0.2, printje = False)
        
        # FFT VT
        lowBack.fftsignalVT             = fastfourierTransform(lowBack, 
                                                                direction = 'VT', 
                                                                position = 'lowBack',
                                                                lineartaper = False,
                                                                plotje = False)    
        lowBack.fftsignalVT.FFTGait(hzMax = lowBack.fftsignalML.hzMax , 
                                    hzMin = lowBack.fftsignalML.hzMin,
                                    printje = False)
        
        # FFT AP
        lowBack.fftsignalAP             = fastfourierTransform(lowBack, 
                                                                direction = 'AP', 
                                                                position = 'lowBack', 
                                                                lineartaper = False,
                                                                plotje = False)    
        lowBack.fftsignalAP.FFTGait(hzMax = lowBack.fftsignalML.hzMax , 
                                    hzMin = lowBack.fftsignalML.hzMin,
                                    printje = False)
                                
        '''
        Stepdetection of the right and leftfoot.
        Returns: Array of peaks VT, Mean/STD stride time
        '''
        
        Stepdetect.stepdetection_lowback_HC(lowBack, 
                                            leftFoot, 
                                            rightFoot, 
                                            devideInparts = steps, 
                                            plotje = False, 
                                            printje = False)
        ############################### Checkpoint 4 ################################
        '''
        the difference between peaks found in feet in low back should be < 10.
        '''
        
        if (np.abs(len(lowBack.peaks) - (leftFoot.stridenum + rightFoot.stridenum)) > 10):
            print('We cannot find the same amount of peaks in LB as in feet sensors.',
                    'Participant cannot be included')
                                                                        
            
        print('\n', 'checkpoint 4 passed','\n')
        
        '''
        Returns: Autocorrelation and autocovariance
        '''
        
        
        commonFunctions.autocovariance(leftFoot, sig = 'gacc', plotje = False)  
        commonFunctions.autocorrelation(leftFoot,sig = 'gacc', plotje = False)  
        commonFunctions.autocovariance(rightFoot, sig = 'gacc',plotje = False)  
        commonFunctions.autocorrelation(rightFoot, sig = 'gacc',plotje = False)  
        commonFunctions.autocovariance(lowBack, sig = 'acc',plotje = False)  
        commonFunctions.autocorrelation(lowBack, sig = 'acc', plotje = False)  
        
        '''
        Returns: range, std, rms
        '''
        
        commonFunctions.desciptives_acc(leftFoot, 
                                        leftFoot.globalacceleration,
                                        leftFoot.peaksVT,
                                        strides,
                                        firststride = 5
                                        )
        commonFunctions.desciptives_acc(rightFoot, 
                                        rightFoot.globalacceleration,
                                        rightFoot.peaksVT,
                                        strides,
                                        firststride = 5
                                        )
        commonFunctions.desciptives_acc(lowBack, 
                                        lowBack.acceleration,
                                        lowBack.peaks,
                                        steps,
                                        firststride = 10
                                        )
        commonFunctions.desciptives_gyr(leftFoot, 
                                        leftFoot.gyroscope,
                                        leftFoot.peaksVT,
                                        strides,
                                        firststride = 5  
                                        )
        commonFunctions.desciptives_gyr(rightFoot,
                                        rightFoot.gyroscope,
                                        rightFoot.peaksVT,
                                        strides,
                                        firststride = 5                                           
                                        )
        commonFunctions.desciptives_gyr(lowBack,
                                        lowBack.gyroscope,
                                        lowBack.peaks,
                                        steps,
                                        firststride = 10
                                        )
        '''
        First the signal is resampled to 50 strides and 100 steps before doing
        lyapunov. 
        Returns: Short term local divergence exponent
        '''
        
        lowBack.lde = Lyapunovmean(lowBack,
                                                lowBack.peaks,
                                                NumbOfSteps = 50,
                                                new_sf = 50,
                                                firststride = 8,
                                                ndim = 5,
                                                delay = 15,
                                                nnbs = 5
                                                ) 

        
        leftFoot.lde = Lyapunovmean(leftFoot,
                                                leftFoot.peaksVT,
                                                NumbOfSteps = 25,
                                                new_sf = 100,
                                                firststride = 4,					
                                                ndim = 5,
                                                delay = 15,
                                                nnbs = 5
                                                )   
        rightFoot.lde = Lyapunovmean(rightFoot,
                                                rightFoot.peaksVT,
                                                NumbOfSteps = 25,
                                                new_sf = 100,
                                                firststride = 4,					
                                                ndim = 5,
                                                delay = 15,
                                                nnbs = 5
                                                )
        asymmetryResults    = Asymmetry(leftFoot, rightFoot, lowBack, parFoot, 
                                        printje = False)
        commonFunctions.normalised(asymmetryResults, leftFoot, rightFoot,
                                    lowBack, lengte)

        Entropy(lowBack, 
                                lowBack.acceleration, 
                                lowBack.peaks, 
                                NumbOfSteps = 50,
                                firststride = 10,
                                m = 2,
                                r=0.25
                                )
                
        Entropy(leftFoot, 
                            leftFoot.globalacceleration,
                            leftFoot.peaksVT, 
                            NumbOfSteps = 25,
                            firststride = 5,
                            m = 2,
                            r=0.25
                            )  
        
        Entropy(rightFoot, 
                            rightFoot.globalacceleration, 
                            rightFoot.peaksVT, 
                            NumbOfSteps = 25,
                            firststride = 5,
                            m = 2,
                            r=0.25
                        )  

        data = {
            "Parameter": [
                "Left Foot Stride Time Mean",
                "Left Foot Stride Time STD",
                # ... (repeat for all left foot parameters)
                "Right Foot Stride Time Mean",
                "Right Foot Stride Time STD",
                # ... (repeat for all right foot parameters)
                "Low Back Stride Time Mean",
                "Low Back Stride Time STD",
                # ... (repeat for all low back parameters)
                # Include all other parameters here...
            ],
            "Value": [
                leftFoot.stridetimemean,
                leftFoot.stridetimeSTD,
                # ... (repeat for all left foot values)
                rightFoot.stridetimemean,
                rightFoot.stridetimeSTD,
                # ... (repeat for all right foot values)
                lowBack.stridetimemean,
                lowBack.stridetimeSTD,
                # ... (repeat for all low back values)
                # Include all other values here...
            ]
        }
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(data)

        # Display the DataFrame as a table in Streamlit
        st.dataframe(df)

        st.success("All files have been uploaded and processed.")
    else:
        st.warning("Please upload all required files to continue.")

    # # Create file uploaders with specific names
    # for i, name in enumerate(upload_names):
    #     uploaded_file = st.file_uploader(f"Upload {name}", type=["csv"])
    #     if uploaded_file is not None:
    #         uploaded_files.append(uploaded_file)

    # # Process uploaded files
    # for i, uploaded_file in enumerate(uploaded_files):
    #     if uploaded_file is not None:
    #         st.header(f"Uploaded File {i+1}")
    #         data = pd.read_csv(uploaded_file, names=['T', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'Time'], sep=',', skiprows=10)
    #         lowBack        = loadCsv(data=data, 
    #                         resample = False,
    #                         )     
            

if __name__ == '__main__':
    main()
