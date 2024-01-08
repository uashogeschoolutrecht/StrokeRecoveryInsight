import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

from balance.app.pages.Scripts.complexityFeatures import *
from balance.app.pages.Scripts.errors import *
from balance.app.pages.Scripts.frequencyFeatures import *
from balance.app.pages.Scripts.proces import *
from balance.app.pages.Scripts.loadFiles import loadCsv
from balance.app.pages.Scripts.spatioTemporalFeatures import *

from scipy.interpolate import interp1d


def check_and_trim_data(i, data):
    # Assuming 'i' is a numeric value that corresponds to the condition numbers 'C1', 'C2', etc.
    # If 'i' is actually a string like 'C1', then adjust the conditions accordingly.
    
    if i in [1, 2, 3]:  # Corresponding to 'C1', 'C2', and 'C3'
        
        if len(data.acceleration) < 5600:
            raise Exception("Included file too short")
        elif len(data.acceleration) > 7500:
            raise Exception("Included file too long")
        else:
            data.acceleration = data.acceleration[0:5600]
            data.gyroscope = data.gyroscope[0:5600]
            
    else:
        
        if len(data.acceleration) < 2600:
            raise Exception("Included file too short")
        elif len(data.acceleration) > 6500:
            raise Exception("Included file too long")
        else:
            data.acceleration = data.acceleration[0:2600]
            data.gyroscope = data.gyroscope[0:2600]
            
    return data

def calculate_data_range(data):
    # Assuming 'data' has an attribute 'filtAcc'
    xyAccData = np.delete(data.filtAcc, 0, 1)
    xyAccDataSmall = xyAccData[0::10]
    return np.max(np.abs(xyAccDataSmall)) * 1.1 if xyAccDataSmall.size > 0 else 0


# Main code
def main():
    st.title("Processing your own collected data")

    # Introduction
    st.header("Introduction")
    st.markdown("""
    On this page, we enable you to upload your own collected 
    balance data.
    """)

    uploaded_files = []
    all_data_ranges = []
    all_xyAccDataSmall = []
    all_lowBack_paths = []

    for i in range(5):
        uploaded_file = st.file_uploader(f"Upload File {i+1}", type=["csv"])

        if uploaded_file is not None:
            # st.header(f"Uploaded File {i+1}")
            data = pd.read_csv(uploaded_file, names=['T', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'Time'], sep=',', skiprows=10)
            lowBack        = loadCsv(data=data, 
                            resample = False,
                            )     
            resp = check_and_trim_data(i, lowBack)
            lowBack.gyroscope -= 0, 0, 0


            ######################### Cartesian coordinate system #########################
            # '''
            # Signal is rotated to a global axis using the method decscribed
            # by Moe Nillsen (1988)and filtered butter, bandpass 1-3.5 Hz.
            # '''
            lowBack.rotAcc, lowBack.rotGyro = Processing.bruijn_rotation(lowBack,   
                                                                        plotje = False
                                                                        )
            # Third order butterworth highpass 0.4 Hz 
            lowBack.filtAcc = Processing.filt_high(lowBack.rotAcc,  
                                                            lowBack.sampleFreq,   
                                                            cutoff = 0.4, 
                                                            order = 3,
                                                            n_input = 3
                                                            )  
            
            xyAccData = np.delete(lowBack.filtAcc, 0, 1)
            xyAccDataSmall = xyAccData[0::10]
            data_range = calculate_data_range(lowBack)
            all_data_ranges.append(data_range)
            all_xyAccDataSmall.append(xyAccDataSmall)
            # Calculate lowBack.path and store it
            lowBack.path = spatioTemporal.path(lowBack.filtAcc)
            all_lowBack_paths.append(lowBack.path)
    max_range = max(all_data_ranges) if all_data_ranges else 0
    
    # Create columns for plots
    num_files = len(all_xyAccDataSmall)

    if num_files > 0:
        # Create columns for plots
        columns = st.columns(num_files)

        for i, (xyAccDataSmall, lowBack_path) in enumerate(zip(all_xyAccDataSmall, all_lowBack_paths)):
            if xyAccDataSmall.size > 0:
                with columns[i]:  
                    fig, ax = plt.subplots()

                    # Extract x and y coordinates
                    x_coords = xyAccDataSmall[:, 0]
                    y_coords = xyAccDataSmall[:, 1]

                    # Plotting a line plot
                    ax.plot(x_coords, y_coords, color='blue')  # You can change the color if needed

                    ax.axhline(y=0, color='k', linestyle='--')
                    ax.axvline(x=0, color='k', linestyle='--')

                    # Set uniform axis limits
                    ax.set_xlim(-max_range, max_range)
                    ax.set_ylim(-max_range, max_range)

                    ax.set_title(f'Meting {i}')
                    ax.set_xlabel('X-Axis')
                    ax.set_ylabel('Y-Axis')

                    st.pyplot(fig)
                    lowBack.path            = spatioTemporal.path(lowBack.filtAcc)
                    st.write(f"Path for file {i+1}: {round(lowBack_path,2)}")
        # lowBack.accMag          = spatioTemporal.magnitude(lowBack.filtAcc) 
        # lowBack.accMagMean      = np.mean(lowBack.accMag)     
                        
        # lowBack.jerkAP          = spatioTemporal.jerk(lowBack.filtAcc[:,0], 
        #                                             lowBack.sampleFreq)               
        # lowBack.jerkML          = spatioTemporal.jerk(lowBack.filtAcc[:,1], 
        #                                             lowBack.sampleFreq)                   
        # lowBack.jerktot         = spatioTemporal.Jerktot(lowBack.filtAcc, 
        #                                             lowBack.sampleFreq)                     
        
        # lowBack.jerkAPrms       = spatioTemporal.RMS(lowBack.jerkAP)
        # lowBack.jerkAPrange     = spatioTemporal.RANGE(lowBack.jerkAP)
        
        # lowBack.jerkMLrms       = spatioTemporal.RMS(lowBack.jerkML)
        # lowBack.jerkMLrange     = spatioTemporal.RANGE(lowBack.jerkML)
        
        # lowBack.accAPrms        = spatioTemporal.RMS(lowBack.filtAcc[:,0])
        # lowBack.accAPrange      = spatioTemporal.RANGE(lowBack.filtAcc[:,0])
        
        # lowBack.accMLrms        = spatioTemporal.RMS(lowBack.filtAcc[:,1])
        # lowBack.accMLrange      = spatioTemporal.RANGE(lowBack.filtAcc[:,1])
        
        # lowBack.gyrAPrms        = spatioTemporal.RMS(lowBack.rotGyro[:,0])
        # lowBack.gyrAPrange      = spatioTemporal.RANGE(lowBack.rotGyro[:,0])
        
        # lowBack.gyrMLrms        = spatioTemporal.RMS(lowBack.rotGyro[:,1])
        # lowBack.gyrMLrange      = spatioTemporal.RANGE(lowBack.rotGyro[:,1])
                                                
        # lowBack.dist            = spatioTemporal.dist(lowBack.filtAcc) 
        # lowBack.path            = spatioTemporal.path(lowBack.filtAcc)
        # st.header(f"lowBack.path: {lowBack.path}")   
        # lowBack.disp            = spatioTemporal.displacement(lowBack.filtAcc)   
        # lowBack.meanVelocity    = 0
        # lowBack.meanFrequency   = spatioTemporal.meanfreq(lowBack)
        # lowBack.area            = spatioTemporal.area(lowBack.jerktot)
        
        # lowBack.circleArea = spatioTemporal.within_cirkel(lowBack.filtAcc, 
        #                                                 percentage = 95,         
        #                                                 plotje = False,
        #                                                 circarea = 0.001)
        
        # lowBack.ellipseArea  = spatioTemporal.within_ellipse(lowBack.filtAcc, 
        #                                                     percentage = 90,   
        #                                                     plotje = False,
        #                                                     ellipl = 0.002,
        #                                                     ellipw = 0.001)
    else:
        # Handle the case when there are no files
        st.write("No files to display.")

# Todo:
# pca score berekenen per conditie
# PLotten onder grafiek ofwel tabel
# MDC laten zien in staaf diagram met error bar. 
# Pca score voor gait
# plotjes voor gait
# pca voor gait
# mdc voor gait

# Alle papers van Richard in balans en gait. net zoals vae 
# youtube filmpje er bij 
# tekst mooi maken, Michiel schrijft about the project/. 
# Github links en DANS repo



if __name__ == '__main__':
    main()
