from . import commonFunctions, advancedFunctions

def Lyapunovmean(obj, peaks, NumbOfSteps, new_sf, firststride, ndim, delay, nnbs):
        obj.resamp50, obj.new_sf = commonFunctions.resample(obj, 
                                                           peaks, 
                                                           NumbOfSteps, 
                                                           firststride, 
                                                           new_sf, 
                                                           sig = 'acceleration')   
        obj.state = advancedFunctions.statespace(obj, 
                                                    sig = obj.resamp50, 
                                                    ndim = ndim, 
                                                    delay = delay, 
                                                    plotje = False)      
        divergence, lde = advancedFunctions.rosenstein(obj,
                                                              obj.state, 
                                                              period = 1, 
                                                              ws = 0.5, 
                                                              nnbs = nnbs, 
                                                              plot = False)

        return  lde
        
    
    
    
# Use code below if you would like to include a x amount of steps multiple
# times and calculate the mean of these values -> np.mean[50,50,50,50]
'''
    count = int(np.floor(len(peaks[firststride:-firststride])/therapistinput))
    slde = []
    for i in range(count):
        stride = i * firststride + firststride
        obj.resamp50, obj.new_sf = commonfunction.resample(obj, 
                                                           peaks, 
                                                           therapistinput, 
                                                           stride, 
                                                           new_sf, 
                                                           sig = 'acceleration')   
        obj.state = advancedfunction.statespace(obj, 
                                                    sig = obj.resamp50, 
                                                    ndim = 5, 
                                                    delay = 10, 
                                                    plotje = False)      
        divergence, lde = advancedfunction.rosenstein(obj,
                                                              obj.state, 
                                                              period = 1, 
                                                              ws = 0.5, 
                                                              nnbs = 5, 
                                                              plot = False)
        slde.append(lde)

    return  np.mean(slde, axis = 0)
    
'''