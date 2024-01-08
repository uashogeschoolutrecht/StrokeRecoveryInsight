from . import advancedFunctions
import numpy as np

def Entropy(obj, signal, peaks, NumbOfSteps, firststride, m, r):            
    '''
    add descriptives to object
    Range, std and rms
    '''     
    select = signal[peaks[firststride]:peaks[NumbOfSteps+firststride]]
    
    obj.approxentropy_accX = advancedFunctions.aproximateEntropy(m, r = (r * np.std(select[:,0] )), 
                                                       signal = select[:,0])
    obj.approxentropy_accY = advancedFunctions.aproximateEntropy(m,r = (r * np.std(select[:,1] )), 
                                                       signal = select[:,1])
    obj.approxentropy_accZ = advancedFunctions.aproximateEntropy(m,r = (r * np.std(select[:,2] )), 
                                                       signal = select[:,2])
    
    obj.sampleEntropy_accX = advancedFunctions.aproximateEntropy(m,r = (r * np.std(select[:,0] )), 
                                                       signal = select[:,0])
    obj.sampleEntropy_accY = advancedFunctions.aproximateEntropy(m,r = (r * np.std(select[:,1] )), 
                                                       signal = select[:,1])
    obj.sampleEntropy_accZ = advancedFunctions.aproximateEntropy(m,r = (r * np.std(select[:,2] )), 
                                                       signal = select[:,2])
    