import numpy as np
from . import commonFunctions

def localToGlobal(lowBack, stationary = False, plotje = None):
        '''
         Input: 
         3D-acceleration time series with intended orientation:
         Low back sensor:
         forward/backward:       acceleration[:,2]
         left/right:             acceleration[:,1]
         up/down:                acceleration[ ,0]    
        
         Output: 
         
         RotationMatrixT: Transpose of rotation matrix used for realignment 
         Define VT direction (RVT) as direction of mean acceleration
         
         Theory
         The mean of AP and ML will with a large by default be an estimate of the
         gravity component in the signal.
         AP: positive direction forward, positive rotation upward
         VT: positive direction upward
         ML: positive direction right, positive rotation upward
         
         Move2data sensor on lowerback:
        Positive is backward
        positive is upward
        positive is left

         Moe-Nilssen, R. (1998). A new method for evaluating motor control in gait under 
         real-life environmental conditions. Part 1: The instrument. Clinical Biomechanics, 
         13(4-5), 320–327. doi:10.1016/s0268-0033(98)00089-8 
        '''
        if stationary:
            stationary      = np.concatenate((lowBack.startEnd))
            APaccel         = lowBack.acceleration[stationary,2]
            MLaccel         = lowBack.acceleration[stationary,1]
        else:
            APaccel         = lowBack.acceleration[:,2]
            MLaccel         = lowBack.acceleration[:,1]        
        APbar           = np.mean(APaccel)
        MLbar           = np.mean(MLaccel)
        
        sin_theta_AP    = APbar
        cos_theta_AP    = np.sqrt(1-(np.power(APbar,2)))
        sin_theta_ML    = MLbar
        cos_theta_ML    = np.sqrt(1-(np.power(MLbar,2)))
        
        APraw           =lowBack.acceleration[:,2]
        MLraw           =lowBack.acceleration[:,1]
        VTraw           =lowBack.acceleration[:,0]

        AP              =(APraw*cos_theta_AP)-(VTraw*sin_theta_AP)
        VT1             =(APraw*sin_theta_AP)+(VTraw*cos_theta_AP)
        ML              =(MLraw*cos_theta_ML)-(VT1*sin_theta_ML)
        VT              =(MLraw*sin_theta_ML)+(VT1*cos_theta_ML)-1        
        
        trueAcc         = np.array((VT, ML, AP)).transpose()
        
        if plotje:
            commonFunctions.plot1(trueAcc)
        
        return trueAcc
          

        

def statespace(obj, sig, ndim = 5, delay = 10, plotje = None):
    try:
        m,n = sig.shape
        state = np.zeros((m - delay * (ndim ), ndim,n))

    except ValueError:
        m = len(sig)
        state = np.zeros((m - delay * (ndim ), ndim,1))
        
    for i_dim in range(ndim):          
        state[:,i_dim,:] = sig[i_dim*delay:len(sig)-(ndim - i_dim) * delay]
     
    if plotje:
         '''
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(state[:,0], state2[:,0], state3[:,0]  )
        ax.plot(state[:,1], state2[:,1], state3[:,1]  )
        ax.plot(state[:,2], state2[:,2], state3[:,2]  )
        ax.plot(state[:,3], state2[:,3], state3[:,3]  )
        ax.plot(state[:,4], state2[:,4], state3[:,4]  )
        '''  
     
    # create state space

    return state   
    # State is a ndim x normalised length x n matrix
    
        
def rosenstein(obj, state, period = 1, ws = 0.5, nnbs = 5, plot = None):
    # ws: wiundow size over which divergence should be calculated
    # fs: sample frequentie
    try:
        fs = obj.new_sf
    except AttributeError:
        fs = int(1 / obj.sampleFreq)
        
    m,n,o = state.shape    
    ws = int(ws * fs)
    emptyarray = np.empty((ws,n,o))
    emptyarray[:] = np.nan
    state = np.concatenate((state, emptyarray),axis = 0) #we extend the state space with NaN, so that we don't run into problems later
    L1 = 0.5 * period * fs
    divergence = np.zeros((m*nnbs,ws,o))
    difference = np.zeros((m+ws,n,o))
    lde = np.zeros(o)
    
    for i_t in range(0,o):
        counter = 0
        for i in range(0,m): # loop over time samples
            for j in range(0,n): # loop over dimensions -> Efficienter
                difference[:,j,i_t] = np.subtract(state[:,j,i_t],state[i,j,i_t]) ** 2
                
            start_index         = int(np.max([0,i-int(0.5*fs*period)])) #find point half a period befor current point   
            stop_index          = int(np.min([m,i+int(0.5*fs*period)])) #find point half a period past current point     
            difference[start_index:stop_index,:,i_t] = np.nan# discard data within one period from sample i_t putting it to n an            
            index          = np.sum(difference[:,:,i_t],axis = 1).argsort()
            
            for k in range(0,nnbs):
                div_tmp = np.subtract(state[i:i+ws,:,i_t],state[index[k]:index[k]+ws,:,i_t])
                divergence[counter,:,i_t] =  np.sqrt(np.sum(div_tmp**2,axis = 1)) 
                counter += 1 # @ Sjoerd
                
        divmat =np.nanmean(np.log(divergence[:,:,i_t]),0); # calculate average for output      
        xdiv =  np.linspace(1,len(divmat), num = int(0.5*fs*period))
        Ps = np.polynomial.polynomial.polyfit(xdiv, divmat[0:int(np.floor(L1))],1)
            
        sfit = np.polynomial.Polynomial(Ps)
        lde[i_t] = Ps[1]
        
    
        if plot:
            import commonfunction
            commonfunction.plot3(divmat, sfit(xdiv),title = 'Lyapunov Rosenstein' ,
                                 xlabel = 'time [s]', ylabel = 'divergence',
                                 legend1 = 'divergence', legend2 = 'lde short'
                                 )
            
    return divergence, lde


def count_vectors(m, r, N, signal):
    C = np.zeros((len(signal)-m +1,1))
    for i in range(N-(m-1)):
        tmpSignal = signal[i:]
        checkValues = tmpSignal
        for j in range(m):
            refValue = signal[i+j]
            minVal = refValue - r
            maxVal = refValue + r
            matchesIDX = np.where((checkValues >= minVal) 
                                    & (checkValues <= maxVal))[0]
            if j == 0:
                initmatchesIDX = matchesIDX
                folNumbIDX = matchesIDX + 1
            else:
                initmatchesIDX = initmatchesIDX[matchesIDX]
                folNumbIDX = initmatchesIDX + 1 + j
            try:
                checkValues = tmpSignal[folNumbIDX]
            except IndexError: 
                checkValues = tmpSignal[folNumbIDX[:-1]]
                
        C[i] += len(matchesIDX)    
        
        try:
            C[(folNumbIDX[1:]-1 - j + i)] += 1
        except:
            continue
        
    return C

def aproximateEntropy(m, r, signal):
    m_2 = m + 1
    N = len(signal)
    
    firstResult = count_vectors(m, r, N, signal)
    logFR = np.sum(np.log(firstResult/ (N - m + 1)))
    phi = logFR / (N - m + 1)
    
    secondResult = count_vectors(m_2, r, N, signal)
    logSR = np.sum(np.log(secondResult / (N - m)))
    phi_2 = logSR / (N - m)
    
    entropy = phi - phi_2
    return entropy


def sampleEntropy(m, r, signal):
    m_2 = m + 1
    N = len(signal)
    
    firstResult = count_vectors(m, r, N, signal) - 1 
    logFR = np.sum((firstResult / (N - m))) / (N- m)
    phi = logFR * (( (N-m - 1)* (N-m)) / 2)
    
    secondResult = count_vectors(m_2, r, N, signal) - 1
    logSR = np.sum(secondResult / (N - m_2))  / (N- m)
    phi_2 = logSR * (( (N-m - 1)* (N-m)) / 2)
    
    entropy =  -np.log( phi_2 /phi) # Returns values between 0 and 2
    return entropy